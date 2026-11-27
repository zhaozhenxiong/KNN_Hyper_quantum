"""Read-only, point-in-time inputs for the A-share research workbench."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable

import numpy as np
import pandas as pd

from my_strategy.agents.factor_agent import FactorAgent
from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS, build_ml_features
from my_strategy.storage.db_paths import raw_db_path
from my_strategy.research_workbench.universe import load_universe, members, restrict_bars, target_coverage


_PRICE_COLUMNS = ("open", "high", "low", "close")
_TRADE_COLUMNS = tuple(f"trade_{name}" for name in _PRICE_COLUMNS)
_STOCK = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_INSTANT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}.*(?:Z|[+-][0-9]{2}:[0-9]{2})")
_TZ = "Asia/Shanghai"
_CATEGORIES = ("board", "industry", "concepts")
_CLASS_FIELDS = ("stock", "effective_from", "available_at", "source", "version")
_BENCHMARK_FILES = {"000300.SH": "csi300"}
_SEED_DATE = pd.Timestamp("1900-01-01")
WORKBENCH_FEATURE_COLUMNS = (
    *DEFAULT_FEATURE_COLUMNS, "mean_reversion_zscore", "atr_pct", "amihud_illiquidity",
)


@dataclass(frozen=True)
class ResearchData:
    """Daily panel, executable calendar, benchmark series and input lineage."""

    frame: pd.DataFrame
    feature_columns: list[str]
    coverage: dict[str, Any]
    sessions: pd.DatetimeIndex
    benchmark_returns: pd.Series


def _date(value: str) -> pd.Timestamp:
    if not _DATE.fullmatch(value):
        raise ValueError(f"Expected an ISO calendar date, got {value!r}")
    return pd.Timestamp(value)


def _local_naive(series: pd.Series, *, day_end_for_date: bool = False) -> pd.Series:
    """Parse explicit offsets; date-only evidence becomes known at day's end."""
    values = series.astype(str)
    days = values.str.fullmatch(_DATE)
    if not (days | values.str.fullmatch(_INSTANT)).all():
        raise ValueError("Availability requires an ISO date or an ISO timestamp with timezone offset")
    local_days = pd.to_datetime(values.where(days), format="%Y-%m-%d", errors="raise")
    if day_end_for_date:
        local_days += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    instants = pd.to_datetime(values.mask(days), format="mixed", utc=True).dt.tz_convert(_TZ).dt.tz_localize(None)
    return instants.where(~days, local_days)


def _fingerprint(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_classification_history(path: str | Path) -> pd.DataFrame:
    """Load a versioned PIT CSV. End dates are exclusive; concepts use `|`."""
    path = Path(path).resolve(strict=True)
    history = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = set(_CLASS_FIELDS).difference(history.columns)
    if required:
        raise ValueError(f"Classification history missing columns: {sorted(required)}")
    history = history[[*_CLASS_FIELDS, *(col for col in ("effective_to", *_CATEGORIES) if col in history)]].copy()
    for column in history.columns:
        history[column] = history[column].str.strip()
    if not history["stock"].str.fullmatch(_STOCK).all() or history[["source", "version"]].eq("").any().any():
        raise ValueError("Classification history requires exchange-suffixed stock codes, source, and version")
    for column in ("effective_to", *_CATEGORIES):
        if column not in history:
            history[column] = ""
    for column in ("effective_from", "effective_to"):
        values = history[column].replace("", None)
        if not values.dropna().str.fullmatch(_DATE).all():
            raise ValueError(f"Classification history requires ISO dates in {column}")
        history[column] = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    invalid_interval = history["effective_to"].notna() & history["effective_to"].le(history["effective_from"])
    if history["effective_from"].isna().any() or invalid_interval.any():
        raise ValueError("Classification history has a missing effective_from or an invalid effective interval")
    history["available_at"] = _local_naive(history["available_at"], day_end_for_date=True)
    if history.duplicated(["stock", "effective_from", "available_at"]).any():
        raise ValueError("Ambiguous multiple classification records for the same stock, effective date and availability")
    history["concepts"] = history["concepts"].map(
        lambda cell: tuple(sorted({token.strip() for token in cell.split("|") if token.strip().lower() not in ("", "unknown")}))
    )
    history.attrs.update(path=str(path), sha256=_fingerprint(path))
    return history.sort_values(["stock", "effective_from", "available_at"]).reset_index(drop=True)


def _known_board(stock: str, exchange: str) -> str:
    code = stock.split(".")[0]
    if exchange == "SH":
        if code.startswith(("688", "689")):
            return "star"
        if code.startswith(("600", "601", "603", "605")):
            return "sh_main"
    if exchange == "SZ":
        if code.startswith(("300", "301")):
            return "chinext"
        if code.startswith(("000", "001")):
            return "sz_main"
        if code.startswith(("002", "003")):
            return "sz_002_003"
    if exchange == "BJ":
        return "bse"
    return "unknown"


def attach_point_in_time_classifications(
    frame: pd.DataFrame, history: pd.DataFrame | None = None, *, board_reference: pd.Series | None = None,
) -> pd.DataFrame:
    """Assign the newest effective classification observable by signal close."""
    out = frame.reset_index(drop=True).copy()
    symbols = out[["stock", "exchange"]].drop_duplicates()
    inferred = {row.stock: _known_board(row.stock, row.exchange) for row in symbols.itertuples(index=False)}
    board = out["stock"].map(inferred if board_reference is None else board_reference).fillna("unknown").to_numpy(
        dtype=object, copy=True,
    )
    industry = np.full(len(out), "unknown", dtype=object)
    concepts = np.empty(len(out), dtype=object)
    concepts.fill(())
    versions = np.full(len(out), "", dtype=object)
    sources = np.full(len(out), "", dtype=object)
    if history is not None and not history.empty:
        groups = out.groupby("stock", sort=False).indices
        dates = out["date"].to_numpy(dtype="datetime64[ns]")
        # Mutate writable arrays one stock at a time. Assigning each interval to
        # an Arrow-backed string column would rebuild that entire column.
        for stock, records in history.groupby("stock", sort=False):
            positions = groups.get(stock)
            if positions is None:
                continue
            stock_dates = dates[positions]
            cutoff = stock_dates + np.timedelta64(15, "h")
            unmatched = np.ones(len(positions), dtype=bool)
            for row in records.iloc[::-1].itertuples(index=False):
                active = unmatched & (stock_dates >= row.effective_from.to_datetime64())
                active &= cutoff >= row.available_at.to_datetime64()
                if pd.notna(row.effective_to):
                    active &= stock_dates < row.effective_to.to_datetime64()
                matches = positions[active]
                if not len(matches):
                    continue
                if row.board.lower() not in ("", "unknown"):
                    board[matches] = row.board
                if row.industry.lower() not in ("", "unknown"):
                    industry[matches] = row.industry
                values = np.empty(len(matches), dtype=object)
                values.fill(row.concepts)
                concepts[matches] = values
                versions[matches] = row.version
                sources[matches] = row.source
                unmatched[active] = False
                if not unmatched.any():
                    break
    out["board"] = board
    out["industry"] = industry
    out["concepts"] = pd.Series(concepts, dtype=object)
    out["classification_version"] = versions
    out["classification_source"] = sources
    return out


def _market_file(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Market prices require a CSV or Parquet file: {path}")
    missing = set(columns).difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty market price file or missing columns {sorted(missing)}: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any() or frame["date"].dt.tz is not None or frame["date"].dt.day_of_week.ge(5).any():
        raise ValueError(f"Market prices require valid, timezone-free trading dates: {path}")
    for column in set(columns).intersection(_PRICE_COLUMNS):
        values = pd.to_numeric(frame[column], errors="raise")
        frame[column] = values.where(np.isfinite(values) & values.gt(0))
    return frame


def _load_benchmark(path: Path) -> pd.DataFrame:
    data = _market_file(path, ("date", "open", "close"))
    if data.duplicated("date").any():
        raise ValueError(f"Benchmark has duplicate trading dates: {path}")
    # The original Tencent loader persisted date/open/close as date/close/open.
    legacy = data["source"].eq("tencent_cn_index") if "source" in data else pd.Series(False, index=data.index)
    data.loc[legacy, ["open", "close"]] = data.loc[legacy, ["close", "open"]].to_numpy()
    data.attrs["legacy_tencent_repaired_rows"] = int(legacy.sum())
    if "available_at" in data:
        available = _local_naive(data["available_at"], day_end_for_date=True)
        late = available.gt(data["date"] + pd.Timedelta(hours=15))
        data.loc[late, ["open", "close"]] = np.nan
    return data.set_index("date").sort_index()


def _read_bars(connection: sqlite3.Connection, stocks: Sequence[str], first: str, last: str) -> pd.DataFrame:
    parts = []
    for offset in range(0, len(stocks), 500):
        chunk = stocks[offset : offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            SELECT n.stock, n.date, s.exchange, n.volume, n.amount, n.turnover,
                   n.source, n.has_trade_price,
                   r.trade_open, r.trade_high, r.trade_low, r.trade_close
            FROM stock_daily_normalized AS n
            LEFT JOIN securities AS s ON s.stock = n.stock
            LEFT JOIN stock_daily_raw AS r ON r.rowid = (
                SELECT rowid FROM stock_daily_raw
                WHERE stock = n.stock AND date = n.date AND source_file = n.source_file
                      AND (n.has_trade_price <> 1 OR trade_close = n.close)
                ORDER BY (trade_close > 0) DESC, row_number DESC
                LIMIT 1
            )
            WHERE n.stock IN ({placeholders}) AND n.date BETWEEN ? AND ?
            ORDER BY n.stock, n.date
        """
        parts.append(pd.read_sql_query(sql, connection, params=[*chunk, first, last], parse_dates=["date"]))
    return pd.concat(parts, ignore_index=True)


def _add_research_factors(factors: pd.DataFrame) -> None:
    """Compute causal factors on one stock in ascending session order.

    The reversal z-score compares today's close with the prior 20 observations.
    ATR uses the trailing 14 true ranges as a fraction of today's close.
    Amihud uses the trailing 20 absolute close returns per unit of traded amount.
    Incomplete windows, zero variance and nonpositive traded amounts stay missing.
    """
    close = factors["close"]
    previous = close.shift()
    reference = previous.rolling(20)
    factors["mean_reversion_zscore"] = (close - reference.mean()) / reference.std(ddof=0).where(lambda value: value.gt(0))

    high, low = factors["high"], factors["low"]
    true_ranges = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1)
    true_range = true_ranges.max(axis=1, skipna=False).where(high.ge(low))
    factors["atr_pct"] = true_range.rolling(14).mean() / close.where(close.gt(0))

    amount = factors["amount"]
    valid_amount = amount.where(np.isfinite(amount) & amount.gt(0) & factors["volume"].gt(0))
    daily_illiquidity = (close / previous - 1).abs() / valid_amount
    factors["amihud_illiquidity"] = daily_illiquidity.rolling(20).mean()


def _reference_amounts(bars: pd.DataFrame, path: Path, *, normalize_volume: bool = False) -> pd.DataFrame:
    """Enrich only bars whose native prices agree with the unadjusted reference."""
    reference = _market_file(path, ("stock", "date", *_PRICE_COLUMNS, "amount"))
    if reference.duplicated(["stock", "date"]).any():
        raise ValueError("Execution reference requires unique stock/date rows")
    fields = (*_PRICE_COLUMNS, "amount")
    if normalize_volume and "volume" in reference:
        fields = (*fields, "volume")
    reference = reference[["stock", "date", *fields]].rename(columns={name: f"reference_{name}" for name in fields})
    merged = bars.merge(reference, on=["stock", "date"], how="left", validate="one_to_one")
    matched = pd.Series(True, index=merged.index)
    for name in _PRICE_COLUMNS:
        matched &= merged[f"reference_{name}"].sub(merged[f"trade_{name}"]).abs().le(0.02001)
    amount = pd.to_numeric(merged["reference_amount"], errors="coerce")
    from_reference = matched & amount.gt(0) & np.isfinite(amount) & ~merged["amount"].gt(0)
    merged["amount"] = merged["amount"].mask(from_reference, amount)
    merged["execution_reference_available"] = merged[[f"reference_{name}" for name in _PRICE_COLUMNS]].notna().all(axis=1)
    merged["execution_reference_matched"] = matched
    merged["amount_from_reference"] = from_reference
    if "reference_volume" in merged:
        volume = pd.to_numeric(merged["reference_volume"], errors="coerce")
        use_volume = matched & volume.gt(0) & np.isfinite(volume) & volume.ne(merged["volume"])
        merged["volume"] = merged["volume"].mask(use_volume, volume)
        merged["volume_from_reference"] = use_volume
    return merged.drop(columns=[f"reference_{name}" for name in fields])


def _make_features(
    bars: pd.DataFrame, price_basis: str, adjusted_path: Path | None, *, partial_reference: bool = False,
    feature_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if price_basis == "adjusted":
        prices = _market_file(adjusted_path, ("stock", "date", *_PRICE_COLUMNS))
        if not prices["stock"].str.fullmatch(_STOCK).fillna(False).all() or prices.duplicated(["stock", "date"]).any():
            raise ValueError("Adjusted history requires canonical stock codes and unique stock/date keys")
        if "available_at" in prices:
            available = _local_naive(prices["available_at"], day_end_for_date=True)
            prices = prices.loc[available.le(prices["date"] + pd.Timedelta(hours=15))]
        data = bars.merge(prices[["stock", "date", *_PRICE_COLUMNS]], on=["stock", "date"], how="left", validate="one_to_one")
        if "execution_reference_matched" in data:
            incompatible = ~data["execution_reference_matched"]
            if partial_reference:
                # Roster experiments use a separately reconciled adjusted input. An
                # optional liquidity feed can add evidence or flag a price conflict.
                incompatible &= data["execution_reference_available"]
            data.loc[incompatible, list(_PRICE_COLUMNS)] = np.nan
        if partial_reference:
            # Unreconciled source days are not observations for an adjusted
            # feature window, including its volume and liquidity history.
            data = data.loc[data[list(_PRICE_COLUMNS)].gt(0).all(axis=1)]
    else:
        data = bars.assign(**{price: bars[f"trade_{price}"] for price in _PRICE_COLUMNS})
    if data.empty:
        raise ValueError("No verified prices remain for workbench feature calculation.")
    parts = []
    feature_columns = list(WORKBENCH_FEATURE_COLUMNS)
    fields = list(data.columns) + feature_columns
    groups = data.groupby("stock", sort=False)
    total = groups.ngroups
    for completed, (_, daily) in enumerate(groups, 1):
        daily = daily.sort_values("date").reset_index(drop=True)
        factors = build_ml_features(FactorAgent.add_factors(daily))
        _add_research_factors(factors)
        for column in ("macd", "macd_signal", "macd_hist", "macd_5_10_6", "macd_5_10_6_signal", "macd_5_10_6_hist"):
            factors[column] = factors[column] / daily["close"].where(daily["close"].gt(0))
        factors[feature_columns] = factors[feature_columns].astype(np.float32)
        parts.append(factors[fields])
        if feature_progress is not None and (completed == 1 or completed % 200 == 0 or completed == total):
            feature_progress(completed, total)
    return pd.concat(parts, ignore_index=True)


def _attach_labels(frame: pd.DataFrame, sessions: pd.DatetimeIndex, next_session: pd.Timestamp | None, benchmark: pd.DataFrame, horizon: int) -> None:
    calendar = pd.Series(sessions, index=sessions)
    trading_dates = calendar.shift(-1)
    if next_session is not None:
        trading_dates.iloc[-1] = next_session
    frame["trade_date"] = frame["date"].map(trading_dates)
    frame["label_end"] = frame["date"].map(calendar.shift(-(horizon + 1)))
    frame["signal_eligible"] = frame[[*_PRICE_COLUMNS, *_TRADE_COLUMNS]].gt(0).all(axis=1) & frame["has_trade_price"] & frame["volume"].gt(0)
    exposure = frame[["stock", "date", "open"]].copy()
    execution_price = frame["trade_open"].gt(0) & frame["has_trade_price"] & frame["volume"].gt(0)
    exposure["open"] = exposure["open"].where(execution_price)
    open_prices = exposure.set_index(["stock", "date"])["open"]
    entry = open_prices.reindex(pd.MultiIndex.from_frame(frame[["stock", "trade_date"]])).to_numpy()
    exit_ = open_prices.reindex(pd.MultiIndex.from_frame(frame[["stock", "label_end"]])).to_numpy()
    index_entry = frame["trade_date"].map(benchmark["open"]).to_numpy()
    index_exit = frame["label_end"].map(benchmark["open"]).to_numpy()
    stock_return = exit_ / entry - 1
    reference_return = index_exit / index_entry - 1
    frame["target"] = pd.Series(stock_return - reference_return).where(frame["signal_eligible"])


def _select_anchored_universe(
    connection: sqlite3.Connection, anchor: str, cap: int, allowed: pd.Series | None = None,
) -> list[str]:
    session = connection.execute(
        "SELECT date FROM stock_daily_normalized WHERE date <= ? ORDER BY date DESC LIMIT 1", (anchor,),
    ).fetchone()
    if session is None:
        return []
    rows = connection.execute(
        "SELECT stock FROM stock_daily_normalized WHERE date = ? AND has_trade_price = 1 ORDER BY stock", session,
    )
    board_pools: dict[str, list[str]] = defaultdict(list)
    for (stock,) in rows:
        if _STOCK.fullmatch(stock) and (allowed is None or stock in allowed.index):
            board = _known_board(stock, stock[-2:]) if allowed is None else allowed[stock]
            board_pools[board].append(stock)
    ranked = [stock for level in zip_longest(*board_pools.values()) for stock in level if stock]
    return ranked[:cap]


def load_research_data(
    start: str,
    end: str,
    *,
    raw_db: str | Path | None = None,
    classification_csv: str | Path | None = None,
    stocks: Sequence[str] | None = None,
    max_stocks: int | None = None,
    benchmark: str = "000300.SH",
    benchmark_path: str | Path | None = None,
    price_basis: str = "raw_research",
    adjusted_path: str | Path | None = None,
    execution_reference_path: str | Path | None = None,
    universe_anchor: str | None = None,
    universe_policy: str = "legacy",
    universe_path: str | Path | None = None,
    horizon: int = 5,
    lookback: int = 60,
    feature_progress: Callable[[int, int], None] | None = None,
) -> ResearchData:
    """Load a local panel with T+1 to T+1+horizon open excess return labels."""
    if (universe_policy == "legacy") != (universe_path is None):
        raise ValueError("A workbench universe policy needs a roster, and legacy uses the existing stock pool.")
    roster = load_universe(universe_path, universe_policy) if universe_path else None
    if max_stocks == 0 and roster is not None:
        max_stocks = None
    first, last = _date(start), _date(end)
    anchor = _date(universe_anchor) if universe_anchor else first
    if not (_SEED_DATE <= anchor <= first <= last) or min(horizon, lookback) < 1 or (max_stocks is not None and max_stocks < 1):
        raise ValueError("Invalid research date range, universe anchor, horizon, lookback, or stock cap")
    if price_basis not in {"adjusted", "raw_research"} or (price_basis == "adjusted") != (adjusted_path is not None):
        raise ValueError("Select raw_research without adjusted_path or adjusted with a verified adjusted price file")
    db_path = Path(raw_db or raw_db_path()).resolve(strict=True)
    if benchmark_path is None:
        instrument = _BENCHMARK_FILES.get(benchmark)
        if not instrument:
            raise ValueError(f"Pass an explicit benchmark_path for {benchmark}")
        benchmark_path = db_path.parent / "external_market" / f"{instrument}.parquet"
    index_path = Path(benchmark_path).resolve(strict=True)
    index_prices = _load_benchmark(index_path)
    price_path = Path(adjusted_path).resolve(strict=True) if adjusted_path is not None else None
    warmup = (first - pd.Timedelta(days=max(90, 2 * lookback))).strftime("%Y-%m-%d")
    scoped = members(roster, first, last) if roster is not None else None
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        warehouse_dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM stock_daily_normalized WHERE date BETWEEN ? AND ? ORDER BY date",
            connection, params=(warmup, end), parse_dates=["date"],
        )["date"]
        next_raw = connection.execute(
            "SELECT date FROM stock_daily_normalized WHERE date > ? ORDER BY date LIMIT 1", (end,),
        ).fetchone()
        sessions = pd.DatetimeIndex(warehouse_dates).union(index_prices.loc[warmup:end].index).sort_values()
        sessions = sessions[(sessions >= first) & (sessions <= last)]
        if len(sessions) < 2:
            raise ValueError(f"Fewer than two local market sessions between {start} and {end}")
        index_next = index_prices.index[index_prices.index > last]
        next_dates = ([pd.Timestamp(next_raw[0])] if next_raw else []) + (list(index_next[:1]) if len(index_next) else [])
        next_session = min(next_dates, default=None)
        if scoped is not None:
            universe_evidence = target_coverage(connection, roster, sessions[-1])
            universe_evidence["historical_universe_stocks"] = len(scoped)
            if stocks is None:
                stocks = scoped.index.tolist() if max_stocks is None else _select_anchored_universe(
                    connection, anchor.date().isoformat(), max_stocks, members(scoped, anchor, anchor)["board"],
                )
        if stocks is None:
            if max_stocks is None:
                rows = connection.execute("SELECT stock FROM securities WHERE first_date <= ? ORDER BY stock", (end,))
                stocks = [row[0] for row in rows]
            else:
                stocks = _select_anchored_universe(connection, anchor.strftime("%Y-%m-%d"), max_stocks)
        requested = list(dict.fromkeys(stock.upper() for stock in stocks))
        if not requested or any(not _STOCK.fullmatch(stock) for stock in requested):
            raise ValueError("Research needs canonical stock codes such as 600000.SH; no universe resolved")
        if scoped is not None:
            outside = sorted(set(requested) - set(scoped.index))
            if outside:
                raise ValueError(f"Stocks outside the dated {universe_policy} universe: {', '.join(outside[:8])}")
        selected = requested[:max_stocks]
        bars = _read_bars(connection, selected, warmup, end)
    if scoped is not None:
        universe_evidence["rows_before_filter"] = len(bars)
        bars = restrict_bars(bars, scoped)
        universe_evidence["rows_after_filter"] = len(bars)
    if bars.empty:
        raise ValueError("Local raw warehouse has no matching stock bars for this research window")

    numeric = [*_TRADE_COLUMNS, "volume", "amount", "turnover"]
    bars[numeric] = bars[numeric].apply(pd.to_numeric, errors="coerce")
    bars["has_trade_price"] = bars["has_trade_price"].eq(1)
    prices = bars[list(_TRADE_COLUMNS)]
    bars[list(_TRADE_COLUMNS)] = prices.where(np.isfinite(prices) & prices.gt(0)).where(bars["has_trade_price"], axis=0)
    exchange = bars["exchange"].fillna("").str.upper()
    bars["exchange"] = exchange.where(exchange.isin(("SH", "SZ", "BJ")), bars["stock"].str[-2:])
    bars["prev_trade_close"] = bars.groupby("stock", sort=False)["trade_close"].ffill().groupby(bars["stock"], sort=False).shift()

    reference_path = Path(execution_reference_path).resolve(strict=True) if execution_reference_path else None
    if reference_path is not None:
        bars = _reference_amounts(bars, reference_path, normalize_volume=scoped is not None)
    frame = _make_features(
        bars, price_basis, price_path, partial_reference=scoped is not None, feature_progress=feature_progress,
    )
    frame = frame.loc[frame["date"].between(first, last)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Local raw warehouse has no stock bars during requested dates")
    history = load_classification_history(classification_csv) if classification_csv is not None else None
    frame = attach_point_in_time_classifications(
        frame, history, board_reference=scoped["board"] if scoped is not None else None,
    )
    _attach_labels(frame, sessions, next_session, index_prices, horizon)
    aligned = index_prices.reindex(sessions)
    coverage = {
        "raw_db": str(db_path), "requested_start": start, "requested_end": end,
        "first_session": sessions[0].date().isoformat(), "last_session": sessions[-1].date().isoformat(),
        "next_session": next_session.date().isoformat() if next_session is not None else None,
        "benchmark": benchmark, "benchmark_path": str(index_path), "benchmark_sha256": _fingerprint(index_path),
        "benchmark_first_session": index_prices.index[0].date().isoformat(),
        "benchmark_last_session": index_prices.index[-1].date().isoformat(),
        "benchmark_legacy_tencent_repaired_rows": index_prices.attrs.get("legacy_tencent_repaired_rows", 0),
        "benchmark_sessions_missing": int(aligned[["open", "close"]].isna().any(axis=1).sum()),
        "benchmark_missing_dates": [value.date().isoformat() for value in aligned.index[aligned[["open", "close"]].isna().any(axis=1)]],
        "requested_stocks": len(selected), "loaded_stocks": int(frame["stock"].nunique()),
        "rows": int(len(frame)), "labeled_rows": int(frame["target"].notna().sum()),
        "raw_trade_rows": int(frame[list(_TRADE_COLUMNS)].gt(0).all(axis=1).sum()),
        "data_last_session": frame["date"].max().date().isoformat(),
        "price_basis": price_basis, "adjusted_path": str(price_path) if price_path is not None else None,
        "adjusted_sha256": _fingerprint(price_path) if price_path is not None else None,
        "adjusted_rows": int(frame[list(_PRICE_COLUMNS)].gt(0).all(axis=1).sum()) if price_path is not None else 0,
        "execution_reference_path": str(reference_path) if reference_path else None,
        "execution_reference_sha256": _fingerprint(reference_path) if reference_path else None,
        "execution_reference_available_rows": int(frame["execution_reference_available"].sum()) if reference_path else 0,
        "execution_reference_matched_rows": int(frame["execution_reference_matched"].sum()) if reference_path else 0,
        "volume_from_reference_rows": int(frame["volume_from_reference"].sum()) if "volume_from_reference" in frame else 0,
        "amount_from_reference_rows": int(frame["amount_from_reference"].sum()) if reference_path else 0,
        "classification_rows": int(frame["classification_version"].ne("").sum()),
        "classification_counts": {
            "board": int(frame["board"].ne("unknown").sum()), "exchange": int(frame["exchange"].ne("unknown").sum()),
            "industry": int(frame["industry"].ne("unknown").sum()), "concepts": int(frame["concepts"].map(bool).sum()),
        },
        "classification_versions": sorted(set(frame["classification_version"]) - {""}),
        "classification_sources": sorted(set(frame["classification_source"]) - {""}),
        "classification_path": history.attrs["path"] if history is not None else None,
        "classification_sha256": history.attrs["sha256"] if history is not None else None,
        "stock_universe_anchor": anchor.date().isoformat() if max_stocks is not None else None,
        "limit_data": "raw_open_and_previous_close_without_authoritative_limits",
        "holding_sessions": horizon,
    }
    if scoped is not None:
        universe_evidence["selected_stocks"] = len(selected)
        universe_evidence["feature_rows_in_requested_period"] = len(frame)
        universe_evidence["selected_latest_trading_rows"] = int(
            (frame["date"].eq(sessions[-1]) & frame["has_trade_price"]).sum()
        )
        coverage["universe"] = universe_evidence
        coverage["universe_path"] = roster.attrs["path"]
    return ResearchData(frame, list(WORKBENCH_FEATURE_COLUMNS), coverage, sessions, aligned["close"].pct_change(fill_method=None))
