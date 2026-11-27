"""Deterministic point-in-time sector features and selection ledgers.

The stock frame must already contain point-in-time industry membership and
adjusted prices for active stock-days.  Optional ``list_date``/``delist_date``
fields are honored (the delisting date is exclusive), as are ``suspended``,
``is_suspended`` and ``market_status``.  No input data is read or written here.

Synthetic sector OHLC is an equal-weight research index, not a tradable price.
Each constituent is normalized by its own previous valid adjusted close.  The
first valid sector observation uses a synthetic previous close of 1.0; later
observations recurse from the previous valid synthetic close.  Consequently,
a newly listed stock without a previous close cannot participate in that day's
synthetic bar.  Such a stock is included in ``missing_price_count`` because it
lacks usable price history, even when its current OHLC is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PRICES = ("open", "high", "low", "close")
_PRICE_COLUMNS = ("date", "stock", "industry", *_PRICES, "amount", "turnover")
_CANDIDATE_CONTRACT = (
    "date", "stock", "industry", "score", "global_rank", "qualified",
    "purchase_candidate", "current_shares", "target_weight", "intent_side",
    "selection_reason", "order_count", "fill_count",
)
_COUNT_COLUMNS = (
    "qualified_count", "purchase_candidate_count", "target_count",
    "buy_intent_count", "order_count", "fill_count",
)
_SECTOR_COLUMNS = (
    "date", "industry", "open", "high", "low", "close", "synthetic_price_reason",
    "constituent_count", "price_participant_count", "suspended_count",
    "missing_price_count", "unclassified_count",
    "return_1d", "return_5d", "return_20d", "return_60d",
    "benchmark_return_1d", "benchmark_return_5d", "benchmark_return_20d", "benchmark_return_60d",
    "relative_strength_1d", "relative_strength_5d", "relative_strength_20d", "relative_strength_60d",
    "ma20", "ma60", "ma20_slope_5d", "breadth_above_ma20", "advance_ratio",
    "new_high_20d_count", "new_low_20d_count", "volatility_20d", "drawdown_60d",
    "sector_amount", "market_amount_share", "amount_change_1d", "amount_change_5d",
    "amount_ma5", "amount_ma20", "activity_ratio", "turnover_median", "turnover_coverage",
    "price_trend", "breadth_trend", "relative_strength_trend", "market_trend_score",
    "strong_up_resonance", "strong_down_resonance", "resonance_state",
)
_CANDIDATE_COLUMNS = (
    "date", "industry", "candidate_row_count", *_COUNT_COLUMNS,
    *(f"market_{column}" for column in ("candidate_row_count", *_COUNT_COLUMNS)),
    "is_rebalance", "selection_reason",
)
_CONFIRMATION_COLUMNS = (
    "candidate_mean_5d", "candidate_median_60d", "confirmation_history_count",
    "confirmation_non_rebalance_history_count", "strategy_confirmation",
)
MODEL_FEATURES = (
    "return_1d", "return_5d", "return_20d", "return_60d",
    "relative_strength_1d", "relative_strength_5d", "relative_strength_20d", "relative_strength_60d",
    "breadth_above_ma20", "advance_ratio", "volatility_20d", "drawdown_60d",
    "market_amount_share", "amount_change_1d", "amount_change_5d", "activity_ratio", "turnover_median",
    "price_trend", "breadth_trend", "relative_strength_trend", "market_trend_score",
)


def attach_model_features(frame: pd.DataFrame, sector_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach only same-close market features; never attach selection/candidate counts."""
    if sector_daily.empty:
        raise ValueError("Sector model view has no daily feature rows")
    daily = sector_daily.copy()
    daily["date"] = _normalize_dates(daily["date"])
    daily["industry"] = _industry(daily["industry"])
    if daily.duplicated(["date", "industry"]).any():
        raise ValueError("Sector model view requires unique date/industry keys")
    daily = daily.loc[daily["industry"].ne("unknown"), ["date", "industry", *MODEL_FEATURES]]
    daily = daily.rename(columns={column: f"sector_{column}" for column in MODEL_FEATURES})
    result = frame.copy()
    result["date"] = _normalize_dates(result["date"])
    result["industry"] = _industry(result["industry"])
    if result.duplicated(["date", "stock"]).any():
        raise ValueError("Stock model frame requires unique date/stock keys")
    return result.merge(daily, on=["date", "industry"], how="left", validate="many_to_one", sort=False)


def _normalize_dates(values: pd.Series, name: str = "date") -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError(f"{name} must contain unique, naive, non-missing dates")
    return dates.dt.normalize()


def _session_index(sessions) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(sessions, errors="raise")).normalize()
    if index.hasnans or index.tz is not None or index.has_duplicates:
        raise ValueError("sessions must contain unique, naive, non-missing dates")
    return index.sort_values()


def _industry(values: pd.Series) -> pd.Series:
    result = values.astype("string").fillna("unknown").str.strip()
    unknown = result.eq("") | result.str.lower().isin(("unknown", "unclassified"))
    return result.mask(unknown, "unknown").astype(str)


def _true(values: pd.Series) -> pd.Series:
    """Interpret ordinary boolean/0/1 ledger flags without truthy NaN values."""
    return values.eq(True).fillna(False).astype(bool)


def _maximum_drawdown(values: np.ndarray) -> float:
    peaks = np.maximum.accumulate(values)
    return float(np.min(values / peaks - 1.0))


def _empty_sector() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_SECTOR_COLUMNS))


def _attach_stock_history(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Add causal per-stock history, retaining the current row's industry."""
    result = frame.sort_values(["stock", "date", "industry"]).reset_index(drop=True).copy()
    for column in ("previous_close", "above_ma20", "advance", "new_high_20d", "new_low_20d"):
        result[column] = np.nan
    for _, rows in result.groupby("stock", sort=False):
        close = rows.set_index("date")["current_close"].reindex(sessions)
        previous = close.ffill().shift(1)
        ma20 = close.rolling(20, min_periods=20).mean()
        high20 = close.rolling(20, min_periods=20).max()
        low20 = close.rolling(20, min_periods=20).min()
        above = close.gt(ma20).astype(float).where(close.notna() & ma20.notna())
        advance = close.gt(previous).astype(float).where(close.notna() & previous.notna())
        new_high = close.eq(high20).astype(float).where(close.notna() & high20.notna())
        new_low = close.eq(low20).astype(float).where(close.notna() & low20.notna())
        locations = sessions.get_indexer(rows["date"])
        for column, values in (
            ("previous_close", previous), ("above_ma20", above), ("advance", advance),
            ("new_high_20d", new_high), ("new_low_20d", new_low),
        ):
            result.loc[rows.index, column] = values.iloc[locations].to_numpy()
    return result


def _sector_base(history: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    ordered = history.sort_values(["industry", "date", "stock"])
    for industry, industry_rows in ordered.groupby("industry", sort=True):
        previous_synthetic_close = np.nan
        for day, group in industry_rows.groupby("date", sort=True):
            constituent_count = len(group)
            suspended = group["suspended"]
            participant = group["valid_ohlc"] & group["previous_close"].gt(0) & ~suspended
            participants = group.loc[participant]
            values = {column: np.nan for column in _PRICES}
            reason = "NO_VALID_CONSTITUENTS"
            if not participants.empty:
                ratios = participants.loc[:, list(_PRICES)].div(participants["previous_close"], axis=0)
                base = previous_synthetic_close if np.isfinite(previous_synthetic_close) else 1.0
                proposed = ratios.mean().to_numpy(dtype=float) * base
                if np.isfinite(proposed).all() and (proposed > 0).all():
                    open_, high, low, close = proposed
                    if low <= min(open_, close) <= max(open_, close) <= high:
                        values = dict(zip(_PRICES, proposed, strict=True))
                        reason = ""
                        previous_synthetic_close = float(close)
                    else:
                        reason = "INVALID_SYNTHETIC_BAR"
                else:
                    reason = "INVALID_SYNTHETIC_BAR"
            amount = group["amount"].sum(min_count=1)
            valid_turnover = group["turnover"].notna()
            records.append({
                "date": day,
                "industry": industry,
                **values,
                "synthetic_price_reason": reason,
                "constituent_count": int(constituent_count),
                "price_participant_count": int(participant.sum()),
                "suspended_count": int(suspended.sum()),
                "missing_price_count": int((~suspended & ~participant).sum()),
                "unclassified_count": int(constituent_count if industry == "unknown" else 0),
                "breadth_above_ma20": group["above_ma20"].mean(),
                "advance_ratio": group["advance"].mean(),
                "new_high_20d_count": int(group["new_high_20d"].sum()),
                "new_low_20d_count": int(group["new_low_20d"].sum()),
                "sector_amount": amount,
                "turnover_median": group["turnover"].median(),
                "turnover_coverage": float(valid_turnover.sum() / constituent_count),
            })
    base = pd.DataFrame.from_records(records)
    market_amount = history.groupby("date")["amount"].sum(min_count=1)
    total = base["date"].map(market_amount)
    base["market_amount_share"] = base["sector_amount"].div(total.where(total.gt(0)))
    return base


def _sector_history_features(
    base: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    benchmark: pd.Series,
) -> pd.DataFrame:
    pieces = []
    benchmark_returns = {
        horizon: benchmark.div(benchmark.shift(horizon)).sub(1.0)
        for horizon in (1, 5, 20, 60)
    }
    for industry, group in base.groupby("industry", sort=True):
        rows = group.set_index("date").reindex(sessions)
        close = rows["close"]
        amount = rows["sector_amount"]
        features = pd.DataFrame(index=sessions)
        for horizon in (1, 5, 20, 60):
            features[f"return_{horizon}d"] = close.div(close.shift(horizon)).sub(1.0)
            features[f"benchmark_return_{horizon}d"] = benchmark_returns[horizon]
            features[f"relative_strength_{horizon}d"] = (
                features[f"return_{horizon}d"] - features[f"benchmark_return_{horizon}d"]
            )
        features["ma20"] = close.rolling(20, min_periods=20).mean()
        features["ma60"] = close.rolling(60, min_periods=60).mean()
        features["ma20_slope_5d"] = features["ma20"].diff(5)
        returns = close.pct_change(fill_method=None)
        features["volatility_20d"] = returns.rolling(20, min_periods=20).std(ddof=0)
        features["drawdown_60d"] = close.rolling(60, min_periods=60).apply(
            _maximum_drawdown, raw=True,
        )
        for horizon in (1, 5):
            previous = amount.shift(horizon)
            features[f"amount_change_{horizon}d"] = amount.div(previous.where(previous.gt(0))).sub(1.0)
        features["amount_ma5"] = amount.rolling(5, min_periods=5).mean()
        features["amount_ma20"] = amount.rolling(20, min_periods=20).mean()
        features["activity_ratio"] = features["amount_ma5"].div(
            features["amount_ma20"].where(features["amount_ma20"].gt(0)),
        )
        features["date"] = sessions
        features["industry"] = industry
        pieces.append(features.reset_index(drop=True))
    history = pd.concat(pieces, ignore_index=True)
    return base.merge(history, on=["date", "industry"], how="left", validate="one_to_one")


def build_sector_daily(
    frame: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    sessions=None,
) -> pd.DataFrame:
    """Build one causal adjusted-price feature row per observed date/industry.

    ``sessions`` is the authoritative trading calendar.  Unknown industries are
    retained for coverage and accounting but no cross-sectional rank is created
    for them (or for any industry).  Rolling windows are observed-session
    windows and include the current session unless explicitly trailing.
    """
    missing = set(_PRICE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Stock daily frame missing columns: {sorted(missing)}")
    work = frame.copy()
    work["date"] = _normalize_dates(work["date"])
    calendar = _session_index(sessions) if sessions is not None else _session_index(work["date"].unique())
    work = work.loc[work["date"].isin(calendar)].reset_index(drop=True)
    if calendar.empty or work.empty:
        return _empty_sector()
    if work.duplicated(["date", "stock"]).any():
        raise ValueError("Stock daily frame requires unique date/stock keys")
    if work["stock"].isna().any() or work["stock"].astype("string").str.strip().eq("").any():
        raise ValueError("Stock daily frame requires non-missing stock identifiers")
    work["stock"] = work["stock"].astype(str)
    work["industry"] = _industry(work["industry"])

    benchmark = benchmark_close.copy()
    benchmark.index = pd.DatetimeIndex(pd.to_datetime(benchmark.index, errors="raise")).normalize()
    if benchmark.index.hasnans or benchmark.index.tz is not None or benchmark.index.has_duplicates:
        raise ValueError("benchmark_close needs unique, naive, non-missing trading dates")
    benchmark = pd.to_numeric(benchmark.sort_index(), errors="coerce").reindex(calendar)
    benchmark = benchmark.where(np.isfinite(benchmark) & benchmark.gt(0))

    status = work.get("market_status", pd.Series("", index=work.index)).astype("string").fillna("").str.strip().str.lower()
    suspended = status.eq("suspended")
    for column in ("suspended", "is_suspended"):
        if column in work:
            suspended |= _true(work[column])
    active = ~status.isin(("delisted", "delist"))
    if "list_date" in work:
        listed = pd.to_datetime(work["list_date"], errors="coerce").dt.normalize()
        active &= listed.isna() | work["date"].ge(listed)
    if "delist_date" in work:
        delisted = pd.to_datetime(work["delist_date"], errors="coerce").dt.normalize()
        active &= delisted.isna() | work["date"].lt(delisted)
    work = work.loc[active].copy()
    work["suspended"] = suspended.loc[work.index].to_numpy()
    if work.empty:
        return _empty_sector()

    for column in _PRICES:
        values = pd.to_numeric(work[column], errors="coerce")
        work[column] = values.where(np.isfinite(values) & values.gt(0))
    for column in ("amount", "turnover"):
        values = pd.to_numeric(work[column], errors="coerce")
        work[column] = values.where(np.isfinite(values) & values.ge(0))
    valid_ohlc = work.loc[:, list(_PRICES)].notna().all(axis=1)
    valid_ohlc &= work["low"].le(work[["open", "close"]].min(axis=1))
    valid_ohlc &= work["high"].ge(work[["open", "close"]].max(axis=1))
    work["valid_ohlc"] = valid_ohlc
    work["current_close"] = work["close"].where(~work["suspended"])

    history = _attach_stock_history(work, calendar)
    base = _sector_base(history)
    result = _sector_history_features(base, calendar, benchmark)

    return _finalize_sector(result, native_index=False)


def _finalize_sector(result: pd.DataFrame, *, native_index: bool) -> pd.DataFrame:
    complete_price = result[["close", "ma20", "ma60", "ma20_slope_5d"]].notna().all(axis=1)
    up_price = (
        result["close"].gt(result["ma20"]) & result["ma20"].gt(result["ma60"])
        & result["ma20_slope_5d"].gt(0)
    )
    down_price = (
        result["close"].lt(result["ma20"]) & result["ma20"].lt(result["ma60"])
        & result["ma20_slope_5d"].lt(0)
    )
    result["price_trend"] = np.select(
        [up_price.fillna(False).to_numpy(dtype=bool), down_price.fillna(False).to_numpy(dtype=bool)],
        [1, -1], default=0,
    ).astype(int)
    result["breadth_trend"] = np.select(
        [result["breadth_above_ma20"].ge(0.6).fillna(False).to_numpy(dtype=bool),
         result["breadth_above_ma20"].le(0.4).fillna(False).to_numpy(dtype=bool)],
        [1, -1], default=0,
    ).astype(int)
    result["relative_strength_trend"] = np.select(
        [result["relative_strength_20d"].gt(0).fillna(False).to_numpy(dtype=bool),
         result["relative_strength_20d"].lt(0).fillna(False).to_numpy(dtype=bool)],
        [1, -1], default=0,
    ).astype(int)
    result["market_trend_score"] = result[[
        "price_trend", "breadth_trend", "relative_strength_trend",
    ]].sum(axis=1).astype(int)
    active_resonance = result["activity_ratio"].ge(1.0)
    result["strong_up_resonance"] = result["market_trend_score"].eq(3) & active_resonance
    result["strong_down_resonance"] = result["market_trend_score"].eq(-3) & active_resonance
    complete_evidence = (
        complete_price
        & result[["breadth_above_ma20", "relative_strength_20d", "activity_ratio"]].notna().all(axis=1)
    )
    result["resonance_state"] = "insufficient"
    if not native_index:
        complete_up = complete_evidence & result["market_trend_score"].gt(0)
        complete_down = complete_evidence & result["market_trend_score"].lt(0)
        result.loc[complete_evidence & result["market_trend_score"].eq(0), "resonance_state"] = "neutral"
        result.loc[complete_up, "resonance_state"] = "up"
        result.loc[complete_down, "resonance_state"] = "down"
        result.loc[result["strong_up_resonance"], "resonance_state"] = "strong_up"
        result.loc[result["strong_down_resonance"], "resonance_state"] = "strong_down"
    else:
        result["strong_up_resonance"] = False
        result["strong_down_resonance"] = False
    columns = ["date", "industry", *(("industry_name",) if "industry_name" in result else ()), *[
        column for column in _SECTOR_COLUMNS if column not in {"date", "industry"}
    ]]
    return result.loc[:, columns].sort_values(["date", "industry"]).reset_index(drop=True)


def build_native_sector_daily(
    daily: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    sessions=None,
) -> pd.DataFrame:
    """Build features from native THS industry bars without inventing breadth.

    Current constituents cannot establish historical breadth, advance ratios or
    turnover distributions.  Those fields stay missing, breadth contributes no
    trend point, and resonance remains ``insufficient`` even when native price,
    activity and benchmark evidence are complete.
    """
    required = {"date", "industry", *_PRICES, "amount"}
    if not required.issubset(daily):
        raise ValueError(f"Native industry daily missing columns: {sorted(required - set(daily))}")
    work = daily.copy()
    work["date"] = _normalize_dates(work["date"])
    calendar = _session_index(sessions) if sessions is not None else _session_index(work["date"].unique())
    work = work.loc[work["date"].isin(calendar)].reset_index(drop=True)
    if work.empty or calendar.empty:
        return _empty_sector()
    if work.duplicated(["date", "industry"]).any() or work["industry"].isna().any():
        raise ValueError("Native industry daily requires unique date/industry keys")
    work["industry"] = _industry(work["industry"])
    work = work.loc[work["industry"].ne("unknown")].copy()
    for column in _PRICES:
        values = pd.to_numeric(work[column], errors="coerce")
        work[column] = values.where(np.isfinite(values) & values.gt(0))
    amount = pd.to_numeric(work["amount"], errors="coerce")
    work["amount"] = amount.where(np.isfinite(amount) & amount.ge(0))
    valid = work[list(_PRICES)].notna().all(axis=1)
    valid &= work["low"].le(work[["open", "close"]].min(axis=1))
    valid &= work["high"].ge(work[["open", "close"]].max(axis=1))
    work = work.loc[valid].copy()
    if work.empty:
        return _empty_sector()

    benchmark = benchmark_close.copy()
    benchmark.index = pd.DatetimeIndex(pd.to_datetime(benchmark.index, errors="raise")).normalize()
    if benchmark.index.hasnans or benchmark.index.tz is not None or benchmark.index.has_duplicates:
        raise ValueError("benchmark_close needs unique, naive, non-missing trading dates")
    benchmark = pd.to_numeric(benchmark.sort_index(), errors="coerce").reindex(calendar)
    benchmark = benchmark.where(np.isfinite(benchmark) & benchmark.gt(0))
    base = pd.DataFrame({
        "date": work["date"], "industry": work["industry"],
        **({} if "industry_name" not in work else {"industry_name": work["industry_name"].astype(str)}),
        **{column: work[column] for column in _PRICES},
        "synthetic_price_reason": "", "sector_amount": work["amount"],
        "constituent_count": np.nan, "price_participant_count": np.nan, "suspended_count": np.nan,
        "missing_price_count": np.nan, "unclassified_count": np.nan, "breadth_above_ma20": np.nan,
        "advance_ratio": np.nan, "new_high_20d_count": np.nan, "new_low_20d_count": np.nan,
        "turnover_median": np.nan, "turnover_coverage": np.nan,
    })
    totals = base.groupby("date")["sector_amount"].sum(min_count=1)
    base["market_amount_share"] = base["sector_amount"].div(base["date"].map(totals).where(totals.gt(0)))
    result = _sector_history_features(base, calendar, benchmark)
    return _finalize_sector(result, native_index=True)


def _nonnegative_counts(values: pd.Series, name: str) -> pd.Series:
    counts = pd.to_numeric(values, errors="coerce").fillna(0)
    if not np.isfinite(counts).all() or counts.lt(0).any() or counts.mod(1).ne(0).any():
        raise ValueError(f"{name} must contain finite, non-negative integer counts")
    return counts.astype(int)


def summarize_selection_candidates(
    candidates: pd.DataFrame,
    *,
    sessions=None,
    rebalance_dates=None,
) -> pd.DataFrame:
    """Aggregate the stock ledger without expanding order/fill records.

    ``qualified`` and ``purchase_candidate`` are authoritative ledger flags.
    Purchase candidates alone are forced to zero off rebalance.  Order and fill
    fields are counts already stored on each unique stock row, so they are
    summed once rather than inferred from the number of rows.  Repeated
    ``market_*`` columns reconcile each date's sector sums to the global ledger.
    """
    missing = set(_CANDIDATE_CONTRACT).difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate ledger missing columns: {sorted(missing)}")
    work = candidates.copy()
    work["date"] = _normalize_dates(work["date"])
    calendar = _session_index(sessions) if sessions is not None else _session_index(work["date"].unique())
    work = work.loc[work["date"].isin(calendar)].reset_index(drop=True)
    if calendar.empty or work.empty:
        return pd.DataFrame(columns=list(_CANDIDATE_COLUMNS))
    if work.duplicated(["date", "stock"]).any():
        raise ValueError("Candidate ledger requires one row per date/stock")
    work["industry"] = _industry(work["industry"])
    reason = work["selection_reason"].astype("string").fillna("").str.strip()
    if rebalance_dates is None:
        work["is_rebalance"] = ~reason.str.upper().eq("NOT_REBALANCE_DAY")
    else:
        rebalances = _session_index(rebalance_dates)
        work["is_rebalance"] = work["date"].isin(rebalances)
    work["qualified_count"] = _true(work["qualified"]).astype(int)
    work["purchase_candidate_count"] = (
        _true(work["purchase_candidate"]) & work["is_rebalance"]
    ).astype(int)
    target_weight = pd.to_numeric(work["target_weight"], errors="coerce")
    work["target_count"] = target_weight.gt(0).astype(int)
    side = work["intent_side"].astype("string").fillna("").str.strip().str.upper()
    work["buy_intent_count"] = side.isin(("BUY", "ADD")).astype(int)
    work["order_count"] = _nonnegative_counts(work["order_count"], "order_count")
    work["fill_count"] = _nonnegative_counts(work["fill_count"], "fill_count")
    work["candidate_row_count"] = 1

    grouped = work.groupby(["date", "industry"], sort=True, as_index=False).agg({
        "candidate_row_count": "sum", **{column: "sum" for column in _COUNT_COLUMNS},
        "is_rebalance": "first",
    })
    totals = work.groupby("date")[["candidate_row_count", *_COUNT_COLUMNS]].sum()
    for column in ("candidate_row_count", *_COUNT_COLUMNS):
        grouped[f"market_{column}"] = grouped["date"].map(totals[column]).astype(int)

    reasons = []
    for row in grouped.itertuples(index=False):
        if not row.is_rebalance:
            reasons.append("NOT_REBALANCE_DAY")
            continue
        if row.purchase_candidate_count:
            reasons.append("")
            continue
        rows = work.loc[work["date"].eq(row.date) & work["industry"].eq(row.industry)]
        unique = sorted(set(rows["selection_reason"].astype("string").fillna("").str.strip()) - {"", "NOT_REBALANCE_DAY"})
        reasons.append("|".join(unique) if unique else "NO_PURCHASE_CANDIDATES")
    grouped["selection_reason"] = reasons
    return grouped.loc[:, list(_CANDIDATE_COLUMNS)].sort_values(["date", "industry"]).reset_index(drop=True)


def strategy_confirmation(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    """Add strategy-only confirmation without changing any market score.

    The current five-session mean is compared with the median of the preceding
    60 sessions (current excluded).  Evidence is sufficient only with at least
    20 non-rebalance observations in that trailing window.  Missing industry
    rows on an observed market date represent zero candidates for this
    calculation; the returned table retains exactly the original row keys.
    """
    required = {"date", "industry", "purchase_candidate_count"}
    missing = required.difference(candidate_daily.columns)
    if missing:
        raise ValueError(f"Candidate daily frame missing columns: {sorted(missing)}")
    result = candidate_daily.copy().reset_index(drop=True)
    for column in _CONFIRMATION_COLUMNS:
        result[column] = pd.Series(dtype=float if column != "strategy_confirmation" else str)
    if result.empty:
        return result
    result["date"] = _normalize_dates(result["date"])
    if result.duplicated(["date", "industry"]).any():
        raise ValueError("Candidate daily frame requires unique date/industry keys")
    result["industry"] = _industry(result["industry"])
    result["purchase_candidate_count"] = _nonnegative_counts(
        result["purchase_candidate_count"], "purchase_candidate_count",
    )
    if "is_rebalance" in result:
        result["is_rebalance"] = _true(result["is_rebalance"])
    elif "selection_reason" in result:
        result["is_rebalance"] = ~result["selection_reason"].astype("string").fillna("").str.upper().eq(
            "NOT_REBALANCE_DAY",
        )
    else:
        result["is_rebalance"] = True
    if result.groupby("date")["is_rebalance"].nunique().gt(1).any():
        raise ValueError("is_rebalance must be consistent across industries on each date")
    calendar = _session_index(result["date"].unique())
    date_flags = result.groupby("date")["is_rebalance"].first().reindex(calendar)

    for column in _CONFIRMATION_COLUMNS:
        if column == "strategy_confirmation":
            result[column] = "insufficient"
        else:
            result[column] = np.nan
    for _, rows in result.groupby("industry", sort=False):
        counts = rows.set_index("date")["purchase_candidate_count"].reindex(calendar).fillna(0)
        mean5 = counts.rolling(5, min_periods=5).mean()
        past = counts.shift(1)
        median60 = past.rolling(60, min_periods=1).median()
        history_count = past.rolling(60, min_periods=1).count()
        non_rebalance_count = (~date_flags).astype(int).shift(1).rolling(60, min_periods=1).sum()
        sufficient = non_rebalance_count.ge(20) & mean5.notna() & median60.notna()
        confirmed = sufficient & counts.gt(0) & mean5.ge(median60)
        state = pd.Series("insufficient", index=calendar, dtype=object)
        state.loc[sufficient] = "unconfirmed"
        state.loc[confirmed] = "confirmed"
        locations = calendar.get_indexer(rows["date"])
        for column, values in (
            ("candidate_mean_5d", mean5), ("candidate_median_60d", median60),
            ("confirmation_history_count", history_count),
            ("confirmation_non_rebalance_history_count", non_rebalance_count),
        ):
            result.loc[rows.index, column] = values.iloc[locations].to_numpy()
        result.loc[rows.index, "strategy_confirmation"] = state.iloc[locations].to_numpy()
    return result
