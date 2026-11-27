"""Point-in-time inputs for an out-of-universe single-stock replay."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from my_strategy.core.run_context import stable_hash
from my_strategy.latent_quant.data import (
    BASE_FEATURES,
    EXTRA_FEATURES,
    LatentData,
    _BOARD_POLICY,
    _HFQ,
    _PRICES,
    _RAW_DIR,
    _STAGE,
    _TRADE_PRICES,
    _UPDATE,
    _market_adjusted,
    _match_execution_reference,
    _status_evidence,
    _stock_features,
)
from my_strategy.research_workbench.data import (
    _STOCK,
    _attach_labels,
    _date,
    _fingerprint,
    _load_benchmark,
    _read_bars,
    attach_point_in_time_classifications,
    load_classification_history,
)
from my_strategy.research_workbench.universe import load_universe, restrict_bars


class StockReplayUnavailable(ValueError):
    """The local archive does not support the same research price convention."""

    def __init__(self, symbol: str, reason: str, *, code: str = "replay_inputs_unavailable") -> None:
        super().__init__(f"{symbol} {reason}")
        self.symbol = symbol
        self.reason = reason
        self.code = code


class _WindowPrices(BytesIO):
    """In-memory Parquet path accepted by the unchanged Latent-Quant helpers."""

    suffix = ".parquet"

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source

    def __str__(self) -> str:
        return str(self.source)


def _day(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _bounded_prices(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, *, stock: str | None = None,
) -> tuple[_WindowPrices, dict]:
    """Read closed date predicates before validating or hashing price values."""
    if path.suffix.lower() == ".parquet":
        date_type = pq.read_schema(path).field("date").type
        lower, upper = start, end
        if pa.types.is_date(date_type):
            lower, upper = start.date(), end.date()
        elif pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
            lower, upper = _day(start), _day(end)
        # Frozen files use ISO dates or midnight timestamps. The half-open bound
        # also handles an input timestamp on the last requested session.
        if pa.types.is_timestamp(date_type):
            upper = end + pd.Timedelta(days=1)
            filters = [("date", ">=", lower), ("date", "<", upper)]
        else:
            filters = [("date", ">=", lower), ("date", "<=", upper)]
        if stock is not None:
            filters.append(("stock", "=", stock))
        data = pq.read_table(path, filters=filters)
    elif path.suffix.lower() == ".csv":
        first, last = _day(start), _day(end)
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            rows = [row for row in reader if first <= row["date"][:10] <= last
                    and (stock is None or row["stock"] == stock)]
            frame = pd.DataFrame(rows, columns=reader.fieldnames, dtype="string")
        data = pa.Table.from_pandas(frame, preserve_index=False)
    else:
        raise ValueError(f"Expected CSV or Parquet price source: {path}")
    output = _WindowPrices(path)
    pq.write_table(data, output, compression="zstd")
    evidence = {"path": str(path), "sha256": sha256(output.getbuffer()).hexdigest(),
                "scope": "in-memory bounded parquet", "start": _day(start), "end": _day(end), "rows": data.num_rows}
    if stock is not None:
        evidence["stock"] = stock
    output.seek(0)
    return output, evidence


def _calendar(path: Path, warm: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    archive = pd.read_csv(path, parse_dates=["date"])
    calendar = pd.DatetimeIndex(archive.loc[archive["is_trading_day"].eq(1), "date"]).sort_values()
    if (calendar.empty or calendar.hasnans or calendar.has_duplicates or calendar.tz is not None
            or calendar.day_of_week.isin((5, 6)).any() or not calendar[0] <= warm <= end <= calendar[-1]):
        raise ValueError("回放日期超出真实交易日历，或日历存在无效记录。")
    return calendar[(calendar >= warm) & (calendar <= end)]


def load_stock_replay_data(
    symbol: str,
    start: str,
    end: str,
    *,
    raw_db: str | Path = _RAW_DIR / "khquant_raw.db",
    roster_path: str | Path = _STAGE / "roster.csv",
    calendar_path: str | Path = _STAGE / "calendar.csv",
    benchmark_path: str | Path = _STAGE / "csi300.csv",
    adjusted_path: str | Path = _HFQ / "verified_market_hfq.parquet",
    execution_reference_path: str | Path = _HFQ / "sina_klc_market_reference.parquet",
    classification_path: str | Path | None = _STAGE / "industries" / "capco-industries.csv",
    status_paths: Sequence[str | Path] = (
        _STAGE / "verified_nontrading_evidence.csv", _UPDATE / "verified_nontrading.csv", _UPDATE / "star_suspensions.csv",
    ),
    warmup_start: str = "2023-09-01",
) -> LatentData:
    """Return one stock with identical feature definitions and prices bounded by end.

    Warmup bars are used only for features, previous close, and benchmark return.
    The market-wide breadth still uses the frozen historical roster. The loader
    reads existing local files, leaves the frozen 120-stock experiment intact,
    and does not substitute unadjusted OHLC for unavailable research prices.
    """
    code = symbol.strip().upper()
    if _STOCK.fullmatch(code) is None:
        raise StockReplayUnavailable(code, "不是带交易所后缀的有效股票代码。", code="invalid_symbol")
    first, last, warm = map(_date, (start, end, warmup_start))
    if not warm <= first <= last:
        raise StockReplayUnavailable(code, "请求日期不在所选预热区间之后。", code="invalid_range")
    paths = {key: Path(value).resolve(strict=True) for key, value in {
        "raw_db": raw_db, "roster": roster_path, "calendar": calendar_path, "benchmark": benchmark_path,
        "adjusted": adjusted_path, "execution_reference": execution_reference_path,
    }.items()}
    calendar = _calendar(paths["calendar"], warm, last)
    sessions = calendar[calendar >= first]
    if sessions.empty:
        raise StockReplayUnavailable(code, "所选日期区间没有已核验的交易日。", code="no_sessions")
    roster = load_universe(paths["roster"], _BOARD_POLICY)
    if code not in roster.index:
        raise StockReplayUnavailable(
            code, "缺少与冻结模型口径一致的沪深主板／科创板历史股票名单和复权证据。",
            code="unverified_roster",
        )
    selected = roster.loc[[code]]
    with closing(sqlite3.connect(paths["raw_db"].as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        raw = _read_bars(connection, (code,), _day(warm), _day(last))
    bars = restrict_bars(raw.loc[raw["date"].isin(calendar)], selected)
    if bars.empty or not bars["date"].isin(sessions).any():
        raise StockReplayUnavailable(code, "当地仓在上市有效期及请求范围内缺少原始日线。", code="missing_raw_bars")
    for column in (*_TRADE_PRICES, "volume", "amount", "turnover"):
        values = pd.to_numeric(bars[column], errors="coerce")
        bars[column] = values.where(np.isfinite(values) & values.gt(0))
    bars["has_trade_price"] = bars["has_trade_price"].eq(1)
    bars[list(_TRADE_PRICES)] = bars[list(_TRADE_PRICES)].where(bars["has_trade_price"], axis=0)
    bars["prev_trade_close"] = bars["trade_close"].ffill().shift()

    inputs = {role: {"path": str(paths[role]), "sha256": _fingerprint(paths[role])} for role in ("roster", "calendar")}
    inputs["raw_db"] = {"path": str(paths["raw_db"]), "scope": "ordered SQLite stock rows through input_cutoff",
                        "sha256": sha256(raw.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()}
    adjusted_source, inputs["adjusted"] = _bounded_prices(paths["adjusted"], warm, last)
    with adjusted_source:
        # Numeric values after the cutoff never reach the shared HFQ validator.
        adjusted, breadth, effective_dates = _market_adjusted(adjusted_source, roster, calendar)
    research_prices = adjusted.loc[adjusted["stock"].eq(code)]
    if not effective_dates or not research_prices["date"].isin(sessions).any():
        raise StockReplayUnavailable(code, "所选窗口没有带有效因子日期的同口径复权行情。", code="missing_adjusted_bars")
    reference, inputs["execution_reference"] = _bounded_prices(paths["execution_reference"], warm, last, stock=code)
    with reference:
        if not inputs["execution_reference"]["rows"]:
            raise StockReplayUnavailable(code, "缺少可核对真实成交量与成交额的原价参考。", code="missing_execution_reference")
        bars = _match_execution_reference(bars, reference, (code,))
    bars = bars.merge(research_prices, on=["stock", "date"], how="left", validate="one_to_one")
    frame = _stock_features(bars, calendar)
    frame = frame.loc[frame["date"].isin(sessions)].reset_index(drop=True)
    frame["breadth"] = frame["date"].map(breadth)

    classification = load_classification_history(classification_path).assign(board="") if classification_path is not None else None
    if classification is not None:
        inputs["classification"] = {key: classification.attrs[key] for key in ("path", "sha256")}
    frame = attach_point_in_time_classifications(frame, classification, board_reference=selected["board"])
    frame["list_date"] = selected["list_date"].iloc[0]
    frame["delist_date"] = selected["delist_date"].iloc[0]
    status_files = [Path(path).resolve(strict=True) for path in status_paths]
    status = _status_evidence(status_files, (code,), sessions)
    inputs["market_status"] = [{"path": str(path), "sha256": _fingerprint(path)} for path in status_files]
    frame = frame.merge(status[["stock", "date", "market_status"]], on=["stock", "date"], how="left", validate="one_to_one")
    frame["suspended"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    traded = frame["has_trade_price"] & frame["trade_open"].gt(0) & frame["volume"].gt(0)
    frame.loc[traded, "suspended"] = False
    frame.loc[frame["market_status"].eq("suspended"), "suspended"] = True

    benchmark_source, inputs["benchmark"] = _bounded_prices(paths["benchmark"], warm, last)
    with benchmark_source:
        benchmark = _load_benchmark(benchmark_source)
    _attach_labels(frame, sessions, None, benchmark, horizon=5)
    feature_columns = [*BASE_FEATURES, *EXTRA_FEATURES]
    frame["signal_eligible"] &= np.isfinite(frame[feature_columns]).all(axis=1) & ~frame["suspended"].fillna(False)
    eligible = frame.loc[frame["signal_eligible"], "date"]
    if eligible.empty:
        raise StockReplayUnavailable(
            code, "可核验的复权价格或匹配量价不足，所选窗口没有完成预热的共同特征。",
            code="insufficient_feature_history",
        )
    benchmark_open = benchmark["open"].reindex(calendar)
    benchmark_close = benchmark["close"].reindex(calendar)
    covered = frame.loc[frame[list(_PRICES)].notna().all(axis=1), "date"]
    coverage = {
        "symbol": code, "name_scope": "single_stock_replay", "input_cutoff": end, "price_basis": "effective_date_hfq",
        "first_session": _day(sessions[0]), "last_session": _day(sessions[-1]), "requested_start": start, "requested_end": end,
        "warmup_start": _day(calendar[0]), "warmup_sessions": int((calendar < sessions[0]).sum()),
        "first_adjusted_bar_with_warmup": _day(research_prices["date"].min()), "eligible_first_session": _day(eligible.min()),
        "eligible_last_session": _day(eligible.max()), "adjusted_first_session": _day(covered.min()),
        "adjusted_last_session": _day(covered.max()), "sessions": len(sessions), "rows": len(frame), "raw_trade_rows": int(traded.sum()),
        "adjusted_rows": len(covered), "signal_eligible_rows": len(eligible),
        "execution_reference_verified_rows": int(frame["reference_verified"].sum()),
        "benchmark_sessions_missing": int(benchmark.reindex(sessions)[["open", "close"]].isna().any(axis=1).sum()),
        "benchmark": "000300.SH", "horizon": 5, "inputs": inputs,
        "point_in_time": {"verified": False, "adjusted_effective_dates_verified": True,
                          "historical_publication_timestamps_verified": False, "vendor_revisions_verified": False},
    }
    coverage["input_version"] = stable_hash({"symbol": code, "cutoff": end, "inputs": inputs, "feature_columns": feature_columns})
    return LatentData(
        frame=frame, feature_columns=feature_columns, coverage=coverage, sessions=sessions,
        benchmark_returns=benchmark_close.pct_change(fill_method=None).reindex(sessions), benchmark_open=benchmark_open,
        benchmark_close=benchmark_close, calendar=calendar, market_status=status, roster=selected,
    )
