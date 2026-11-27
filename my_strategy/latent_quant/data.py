"""Read-only Latent-Quant panel with a fixed universe and trading calendar."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import random
import sqlite3

import numpy as np
import pandas as pd

from my_strategy.core.paths import DATA_ROOT
from my_strategy.research_workbench.data import (
    ResearchData,
    _attach_labels,
    _fingerprint,
    _load_benchmark,
    _local_naive,
    _read_bars,
    attach_point_in_time_classifications,
    load_classification_history,
)
from my_strategy.research_workbench.universe import load_universe, members, restrict_bars


_RAW_DIR = DATA_ROOT / "raw"
_STAGE = _RAW_DIR / "workbench_universe" / "20261120"
_UPDATE = _RAW_DIR / "workbench_updates" / "20261120"
_HFQ = _STAGE / "hfq-market"
_PRICES = ("open", "high", "low", "close")
_TRADE_PRICES = tuple(f"trade_{column}" for column in _PRICES)
_KEYS = ["stock", "date"]
_BOARD_POLICY = "main_star"
_PRICE_TOLERANCE = 0.011
_VOLUME_TOLERANCE = 0.01
_ATR_PERIOD = 14
_SUPERTREND_MULTIPLE = 3.0

BASE_FEATURES = (
    "return_1d", "return_5d", "open_gap", "intraday_return", "range_pct",
    "volume_change_5d", "log_amount", "volatility_20d", "breadth",
)
EXTRA_FEATURES = ("st_direction", "atr_pct", "st_distance")


@dataclass(frozen=True)
class LatentData(ResearchData):
    benchmark_open: pd.Series
    benchmark_close: pd.Series
    calendar: pd.DatetimeIndex
    market_status: pd.DataFrame
    roster: pd.DataFrame


def _sample_universe(
    connection: sqlite3.Connection,
    roster: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    anchor: pd.Timestamp,
    max_stocks: int,
    seed: int,
) -> tuple[list[str], pd.Timestamp]:
    session = calendar[calendar < anchor][-1]
    active = members(roster, session, session)
    traded = [stock for stock, in connection.execute(
        """SELECT stock FROM stock_daily_normalized
           WHERE date = ? AND has_trade_price = 1 AND open > 0 AND volume > 0
           ORDER BY stock""",
        (session.date().isoformat(),),
    ) if stock in active.index]
    if not traded:
        raise ValueError(f"No eligible stocks with raw prices on anchor session {session.date()}")
    rng = random.Random(seed)
    pools = [pool.index.tolist() for _, pool in active.loc[traded].groupby("board", sort=True)]
    rng.shuffle(pools)
    for pool in pools:
        rng.shuffle(pool)
    ranked = [stock for row in zip_longest(*pools) for stock in row if stock is not None]
    return ranked[:max_stocks], session


def _read_market(path: Path, required: Sequence[str], stocks: Sequence[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        filters = [("stock", "in", list(stocks))] if stocks is not None else None
        data = pd.read_parquet(path, filters=filters)
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path, encoding="utf-8-sig")
        if stocks is not None:
            data = data.loc[data["stock"].isin(stocks)].copy()
    else:
        raise ValueError(f"Expected CSV or Parquet input at {path}")
    if missing := set(required).difference(data.columns):
        raise ValueError(f"Missing fields {sorted(missing)} in {path}")
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    if data["date"].isna().any() or data["date"].dt.tz is not None or data.duplicated(_KEYS).any():
        raise ValueError(f"Invalid or duplicate stock/date keys in {path}")
    return data


def _market_adjusted(
    path: Path, roster: pd.DataFrame, calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    data = _read_market(path, (*_KEYS, *_PRICES))
    effective_dates = "factor_effective_date" in data
    if effective_dates:
        effective = pd.to_datetime(data["factor_effective_date"], errors="raise")
        if effective.isna().any() or effective.gt(data["date"]).any():
            raise ValueError(f"Invalid or future-effective factor in {path}")
    if "available_at" in data:
        cutoff = data["date"] + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        available = _local_naive(data["available_at"], day_end_for_date=True)
        data = data.loc[available.le(cutoff)].copy()
    for column in _PRICES:
        values = pd.to_numeric(data[column], errors="coerce")
        data[column] = values.where(np.isfinite(values) & values.gt(0))
    scoped = data.loc[data["date"].isin(calendar) & data["stock"].isin(roster.index)]
    adjusted = restrict_bars(scoped[[*_KEYS, *_PRICES]], roster).drop(columns="exchange")
    ordered = adjusted.sort_values(_KEYS)
    previous_dates = pd.Series(calendar, index=calendar).shift()
    grouped = ordered.groupby("stock", sort=False)
    adjacent = grouped["date"].shift().eq(ordered["date"].map(previous_dates))
    gains = ordered["close"].div(grouped["close"].shift()).sub(1).where(adjacent)
    up = gains.gt(0).where(gains.notna()).astype(float)
    breadth = up.groupby(ordered["date"]).mean().reindex(calendar)
    return adjusted, breadth, effective_dates


def _match_execution_reference(bars: pd.DataFrame, path: Path, stocks: Sequence[str]) -> pd.DataFrame:
    reference = _read_market(path, (*_KEYS, *_PRICES, "volume", "amount"), stocks)
    fields = [*_PRICES, "volume", "amount"]
    if "preclose" not in reference:
        reference["preclose"] = np.nan
    fields.append("preclose")
    if "available_at" in reference:
        cutoff = reference["date"] + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        available = _local_naive(reference["available_at"], day_end_for_date=True)
        reference = reference.loc[available.le(cutoff)]
    reference = reference[[*_KEYS, *fields]].rename(columns={column: f"ref_{column}" for column in fields})
    for column in fields:
        reference[f"ref_{column}"] = pd.to_numeric(reference[f"ref_{column}"], errors="coerce")
    result = bars.merge(reference, on=_KEYS, how="left", validate="one_to_one")
    price_match = result["has_trade_price"].copy()
    for column in _PRICES:
        price_match &= result[f"trade_{column}"].sub(result[f"ref_{column}"]).abs().le(_PRICE_TOLERANCE)
    ratio = result["ref_volume"].div(result["volume"].where(result["volume"].gt(0)))
    same_volume = ratio.sub(1).abs().le(_VOLUME_TOLERANCE)
    lot_volume = ratio.div(100).sub(1).abs().le(_VOLUME_TOLERANCE) | ratio.mul(100).sub(1).abs().le(_VOLUME_TOLERANCE)
    verified = price_match & (same_volume | lot_volume)
    result["reference_verified"] = verified
    result["normalized_volume_units"] = price_match & lot_volume
    result["feature_volume"] = result["ref_volume"].where(verified)
    from_reference = verified & result["ref_amount"].gt(0) & ~result["amount"].gt(0)
    result["amount"] = result["amount"].mask(from_reference, result["ref_amount"])
    result["amount_from_reference"] = from_reference
    result["provider_preclose"] = result["ref_preclose"].where(verified & result["ref_preclose"].gt(0))
    return result.drop(columns=[f"ref_{column}" for column in fields])


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mid = (high + low) / 2
    upper = (mid + _SUPERTREND_MULTIPLE * atr).to_numpy(copy=True)
    lower = (mid - _SUPERTREND_MULTIPLE * atr).to_numpy(copy=True)
    prices = close.to_numpy()
    direction = np.full(len(close), np.nan)
    trend = np.full(len(close), np.nan)
    for i in range(len(close)):
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]) or not np.isfinite(prices[i]):
            continue
        if i == 0 or not np.isfinite(direction[i - 1]):
            direction[i] = 1 if prices[i] >= mid.iat[i] else -1
        else:
            if prices[i - 1] <= upper[i - 1]:
                upper[i] = min(upper[i], upper[i - 1])
            if prices[i - 1] >= lower[i - 1]:
                lower[i] = max(lower[i], lower[i - 1])
            if direction[i - 1] < 0:
                direction[i] = 1 if prices[i] > upper[i] else -1
            else:
                direction[i] = -1 if prices[i] < lower[i] else 1
        trend[i] = lower[i] if direction[i] > 0 else upper[i]
    return direction, trend


def _stock_features(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    # Reindex on genuine sessions so a missing day cannot masquerade as yesterday.
    observed = daily.set_index("date")
    days = observed.reindex(calendar)
    close = days["close"]
    previous_close = close.shift()
    previous_volume = days["feature_volume"].shift().rolling(5).median()
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame({
        "return_1d": returns,
        "return_5d": close.div(close.shift(5)).sub(1).where(returns.rolling(5).count().eq(5)),
        "open_gap": days["open"].div(previous_close).sub(1),
        "intraday_return": close.div(days["open"]).sub(1),
        "range_pct": days["high"].sub(days["low"]).div(previous_close),
        "volume_change_5d": days["feature_volume"].div(previous_volume).sub(1),
        "log_amount": np.log(days["amount"].where(days["amount"].gt(0))),
        "volatility_20d": returns.rolling(20).std(ddof=0),
    }, index=calendar)
    true_range = pd.concat([
        days["high"].sub(days["low"]),
        days["high"].sub(previous_close).abs(),
        days["low"].sub(previous_close).abs(),
    ], axis=1).max(axis=1, skipna=False)
    atr = true_range.rolling(_ATR_PERIOD).mean()
    direction, trend = _supertrend(days["high"], days["low"], close, atr)
    features["st_direction"] = direction
    features["atr_pct"] = atr.div(close)
    features["st_distance"] = (close - trend).div(close)
    return daily.join(features, on="date")


def _status_evidence(paths: Sequence[Path], stocks: Sequence[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for path in paths:
        data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
        if "market_status" not in data:
            data["market_status"] = data["trade_status"].eq(0).map({True: "suspended"})
        if "source_url" not in data:
            data["source_url"] = data.get("evidence_url", "")
        parts.append(data.loc[
            data["stock"].isin(stocks) & data["date"].isin(dates)
            & data["market_status"].isin(("suspended", "delisted")),
            [*_KEYS, "market_status", "source", "source_url"],
        ])
    if not parts:
        return pd.DataFrame({
            "stock": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
            "market_status": pd.Series(dtype=str), "source": pd.Series(dtype=str), "source_url": pd.Series(dtype=str),
        })
    return pd.concat(parts, ignore_index=True).drop_duplicates(_KEYS).sort_values(_KEYS)


def load_panel(
    start: str = "2024-01-02",
    end: str = "2026-11-20",
    max_stocks: int = 120,
    seed: int = 42,
    *,
    raw_db: str | Path = _RAW_DIR / "khquant_raw.db",
    roster_path: str | Path = _STAGE / "roster.csv",
    calendar_path: str | Path = _STAGE / "calendar.csv",
    benchmark_path: str | Path = _STAGE / "csi300.csv",
    adjusted_path: str | Path = _HFQ / "verified_market_hfq.parquet",
    execution_reference_path: str | Path = _HFQ / "sina_klc_market_reference.parquet",
    classification_path: str | Path | None = _STAGE / "industries" / "capco-industries.csv",
    status_paths: Sequence[str | Path] = (
        _STAGE / "verified_nontrading_evidence.csv", _UPDATE / "verified_nontrading.csv",
        _UPDATE / "star_suspensions.csv",
    ),
    warmup_start: str = "2023-09-01",
    universe_anchor: str = "2024-01-02",
) -> LatentData:
    """Use dated roster and raw warehouse, then build features and T+1 to T+6 labels."""
    paths = {
        name: Path(path).resolve(strict=True) for name, path in {
            "raw_db": raw_db, "roster": roster_path, "calendar": calendar_path, "benchmark": benchmark_path,
            "adjusted": adjusted_path, "execution_reference": execution_reference_path,
        }.items()
    }
    statuses = [Path(path).resolve(strict=True) for path in status_paths]
    first, last, warm, anchor = map(pd.Timestamp, (start, end, warmup_start, universe_anchor))
    if not warm < anchor <= first <= last or max_stocks < 1:
        raise ValueError("Require warmup_start < universe_anchor <= start <= end and a positive stock cap")
    def as_date(day: pd.Timestamp) -> str:
        return day.date().isoformat()

    session_file = pd.read_csv(paths["calendar"], parse_dates=["date"])
    calendar = pd.DatetimeIndex(session_file.loc[session_file["is_trading_day"].eq(1), "date"]).sort_values()
    if (calendar.empty or calendar.hasnans or calendar.has_duplicates or calendar.tz is not None
            or calendar.day_of_week.isin((5, 6)).any() or not calendar[0] <= warm < first <= last <= calendar[-1]):
        raise ValueError("Invalid verified calendar or requested dates outside its coverage")
    sessions = calendar[(calendar >= first) & (calendar <= last)]
    feature_calendar = calendar[(calendar >= warm) & (calendar <= last)]
    if len(sessions) <= 6 or not len(calendar[(calendar >= warm) & (calendar < anchor)]):
        raise ValueError("Need an anchor session in the warmup and at least seven study sessions")
    roster = load_universe(paths["roster"], _BOARD_POLICY)
    benchmark = _load_benchmark(paths["benchmark"])
    classification = (
        load_classification_history(classification_path).assign(board="") if classification_path is not None else None
    )
    adjusted, breadth, effective_dates = _market_adjusted(paths["adjusted"], roster, feature_calendar)

    with closing(sqlite3.connect(paths["raw_db"].as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        stocks, anchor_session = _sample_universe(connection, roster, feature_calendar, anchor, max_stocks, seed)
        raw = _read_bars(connection, stocks, as_date(warm), as_date(last))
    raw_digest = sha256(raw.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    selected = roster.loc[stocks]
    bars = restrict_bars(raw.loc[raw["date"].isin(feature_calendar)], selected)
    for column in (*_TRADE_PRICES, "volume", "amount", "turnover"):
        values = pd.to_numeric(bars[column], errors="coerce")
        bars[column] = values.where(np.isfinite(values) & values.gt(0))
    bars["has_trade_price"] = bars["has_trade_price"].eq(1)
    bars[list(_TRADE_PRICES)] = bars[list(_TRADE_PRICES)].where(bars["has_trade_price"], axis=0)
    bars["prev_trade_close"] = bars.groupby("stock", sort=False)["trade_close"].transform(lambda values: values.ffill().shift())
    bars = _match_execution_reference(bars, paths["execution_reference"], stocks)
    bars = bars.merge(adjusted.loc[adjusted["stock"].isin(stocks)], on=_KEYS, how="left", validate="one_to_one")
    if bars.empty:
        raise ValueError("No observed rows for the selected historical universe")
    features = [_stock_features(daily, feature_calendar) for _, daily in bars.groupby("stock", sort=False)]
    frame = pd.concat(features, ignore_index=True)
    frame = frame.loc[frame["date"].isin(sessions)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No observed rows in the requested study dates")
    frame["breadth"] = frame["date"].map(breadth)
    frame = attach_point_in_time_classifications(frame, classification, board_reference=selected["board"])
    frame["list_date"] = frame["stock"].map(selected["list_date"])
    frame["delist_date"] = frame["stock"].map(selected["delist_date"])
    status = _status_evidence(statuses, stocks, sessions)
    frame = frame.merge(status[[*_KEYS, "market_status"]], on=_KEYS, how="left", validate="one_to_one")
    frame["suspended"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    traded = frame["has_trade_price"] & frame["trade_open"].gt(0) & frame["volume"].gt(0)
    frame.loc[traded, "suspended"] = False
    frame.loc[frame["market_status"].eq("suspended"), "suspended"] = True
    _attach_labels(frame, sessions, None, benchmark, horizon=5)
    frame["signal_eligible"] &= (
        np.isfinite(frame[[*BASE_FEATURES, *EXTRA_FEATURES]]).all(axis=1)
        & ~frame["suspended"].fillna(False)
    )

    expected_keys = sum(len(members(selected, day, day)) for day in sessions)
    valid_trade_keys = pd.MultiIndex.from_frame(frame.loc[traded, _KEYS])
    explanatory = status.loc[status["market_status"].eq("suspended")]
    explanatory = restrict_bars(explanatory, selected)
    explained = (~pd.MultiIndex.from_frame(explanatory[_KEYS]).isin(valid_trade_keys)).sum()
    labeled = frame["target"].notna()
    eligible = frame["signal_eligible"]
    eligible_labeled = eligible & labeled
    benchmark_open = benchmark["open"].reindex(calendar)
    benchmark_close = benchmark["close"].reindex(calendar)
    file_inputs = {
        name: {"path": str(path), "sha256": _fingerprint(path)} for name, path in paths.items() if name != "raw_db"
    }
    file_inputs["raw_db"] = {
        "path": str(paths["raw_db"]), "sha256": raw_digest,
        "scope": "UTF-8 CSV of ordered selected raw OHLCV rows and provenance including warmup, read in one SQLite transaction",
    }
    file_inputs["market_status"] = [{"path": str(path), "sha256": _fingerprint(path)} for path in statuses]
    if classification is not None:
        file_inputs["classification"] = {key: classification.attrs[key] for key in ("path", "sha256")}
    eligible_dates = frame.loc[eligible_labeled, "date"]
    classified = frame["industry"].ne("unknown")
    classified_dates = frame.loc[classified, "date"]
    coverage = {
        "first_session": as_date(sessions[0]), "last_session": as_date(sessions[-1]),
        "warmup_start": as_date(feature_calendar[0]), "horizon": 5, "benchmark": "000300.SH",
        "benchmark_sessions_missing": int(benchmark.reindex(sessions)[["open", "close"]].isna().any(axis=1).sum()),
        "universe_policy": _BOARD_POLICY, "anchor_session": as_date(anchor_session), "seed": seed,
        "stocks": stocks, "requested_stocks": max_stocks, "loaded_stocks": int(frame["stock"].nunique()),
        "stock_counts_by_board": selected["board"].value_counts().to_dict(),
        "historical_delistings": int(selected["delist_date"].le(last).sum()),
        "rows": len(frame), "expected_active_stock_days": expected_keys,
        "raw_trade_rows": len(valid_trade_keys), "adjusted_rows": int(frame[list(_PRICES)].notna().all(axis=1).sum()),
        "known_suspended_stock_days": int(explained),
        "missing_trade_status_stock_days": int(expected_keys - len(valid_trade_keys) - explained),
        "execution_reference_verified_rows": int(frame["reference_verified"].sum()),
        "execution_reference_volume_unit_repairs": int(frame["normalized_volume_units"].sum()),
        "provider_preclose_rows": int(frame["provider_preclose"].notna().sum()),
        "amount_from_reference_rows": int(frame["amount_from_reference"].sum()),
        "industry_classification": {
            "known_rows": int(classified.sum()), "unknown_rows": int((~classified).sum()),
            "known_eligible_rows": int((eligible & classified).sum()),
            "first_known_session": as_date(classified_dates.min()) if not classified_dates.empty else None,
            "usage": "execution concentration control only",
        },
        "signal_eligible_rows": int(eligible.sum()), "labeled_rows": int(labeled.sum()),
        "eligible_labeled_rows": int(eligible_labeled.sum()),
        "eligible_label_first_signal": as_date(eligible_dates.min()) if not eligible_dates.empty else None,
        "eligible_label_last_signal": as_date(eligible_dates.max()) if not eligible_dates.empty else None,
        "price_basis": "effective_date_hfq",
        "breadth_scope": "Fraction rising among historical roster names with verified adjacent-session HFQ closes",
        "point_in_time": {
            "verified": False,
            "adjusted_effective_dates_verified": effective_dates,
            "historical_publication_timestamps_verified": False,
            "vendor_revisions_verified": False,
            "qualification": "Frozen retrospective archive; daily availability and vintage revisions are unverified.",
        },
        "inputs": file_inputs,
    }
    return LatentData(
        frame=frame, feature_columns=[*BASE_FEATURES, *EXTRA_FEATURES], coverage=coverage, sessions=sessions,
        benchmark_returns=benchmark_close.pct_change(fill_method=None).reindex(sessions),
        benchmark_open=benchmark_open, benchmark_close=benchmark_close, calendar=calendar, market_status=status,
        roster=selected,
    )
