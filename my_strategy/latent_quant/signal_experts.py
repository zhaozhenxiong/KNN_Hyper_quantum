"""Causal signal experts with a common excess-return quantile contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .knn_backend import KNNBackend


QUANTILE_LEVELS = (0.10, 0.50, 0.90)
MIN_GROUP = 20
BASE_EXPERTS = (
    "donchian_breakout",
    "rsi_bollinger_mean_reversion",
    "ma_adx_trend",
    "momentum_rotation",
)
ALL_EXPERTS = (*BASE_EXPERTS, "majority_vote", "online_knn_volume_trend")


@dataclass(frozen=True)
class ExpertForecast:
    quantiles: np.ndarray
    raw_score: pd.Series
    state: pd.Series
    evidence: dict


def _rolling(frame: pd.DataFrame, column: str, window: int, method: str) -> pd.Series:
    return frame.groupby("stock", sort=False)[column].transform(
        lambda values: getattr(values.rolling(window, min_periods=window), method)(),
    )


def _ema(frame: pd.DataFrame, column: str, span: int, *, wilder: bool = False) -> pd.Series:
    alpha = 1 / span if wilder else 2 / (span + 1)
    return frame.groupby("stock", sort=False)[column].transform(
        lambda values: values.ewm(alpha=alpha, adjust=False, min_periods=span).mean(),
    )


def _sequential_states(
    frame: pd.DataFrame,
    *,
    upper: pd.Series,
    lower: pd.Series,
    basis: pd.Series,
    rsi: pd.Series,
    atr: pd.Series,
    kind: str,
) -> pd.Series:
    result = np.full(len(frame), np.nan)
    close = frame["close"].to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    basis_values = basis.to_numpy(dtype=float)
    rsi_values = rsi.to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    for positions in frame.groupby("stock", sort=False).indices.values():
        state = 0
        trailing = np.nan
        for position in positions:
            price = close[position]
            if kind == "breakout":
                high, low, volatility = upper_values[position], lower_values[position], atr_values[position]
                if not np.isfinite([price, high, low, volatility]).all():
                    state, trailing = 0, np.nan
                    continue
                if state == 1:
                    trailing = max(trailing, price - 2 * volatility)
                    if price < low or price < trailing:
                        state, trailing = -1, np.nan
                elif state == -1:
                    state = 0
                if state == 0 and price > high:
                    state, trailing = 1, price - 2 * volatility
            else:
                middle, oscillator = basis_values[position], rsi_values[position]
                if not np.isfinite([price, lower_values[position], middle, oscillator]).all():
                    state = 0
                    continue
                if state == 1 and (oscillator > 55 or price > middle):
                    state = -1
                elif state == -1:
                    state = 0
                if state == 0 and oscillator < 35 and price < lower_values[position]:
                    state = 1
            result[position] = state
    return pd.Series(result, index=frame.index, dtype=float)


def _signals(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["stock", "date"], kind="stable")
    close, high, low, volume = (ordered[column].astype(float) for column in ("close", "high", "low", "volume"))
    previous_close = close.groupby(ordered["stock"], sort=False).shift()
    returns = close / previous_close - 1
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1,
    ).max(axis=1)
    work = ordered[["stock", "date"]].copy()
    work["close"], work["high"], work["low"], work["volume"] = close, high, low, volume
    work["true_range"] = true_range
    work["returns"] = returns
    atr = _ema(work, "true_range", 14, wilder=True)

    ema_fast, ema_slow = _ema(work, "close", 20), _ema(work, "close", 60)
    up = high.groupby(ordered["stock"], sort=False).diff()
    down = -low.groupby(ordered["stock"], sort=False).diff()
    plus_dm = up.where(up.gt(down) & up.gt(0), 0.0)
    minus_dm = down.where(down.gt(up) & down.gt(0), 0.0)
    work["plus_dm"], work["minus_dm"] = plus_dm, minus_dm
    plus_di = 100 * _ema(work, "plus_dm", 14, wilder=True) / atr
    minus_di = 100 * _ema(work, "minus_dm", 14, wilder=True) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    work["dx"] = dx
    adx = _ema(work, "dx", 14, wilder=True)
    trend_ready = ema_slow.notna() & adx.notna() & atr.gt(0)
    trend_direction = np.sign(ema_fast - ema_slow)
    trend_state = trend_direction * (1 + adx.ge(20).astype(int))
    trend_state = trend_state.where(trend_ready)
    trend_raw = (ema_fast / ema_slow - 1) * (1 + adx.fillna(0) / 100)

    work["gain"] = returns.clip(lower=0)
    work["loss"] = -returns.clip(upper=0)
    average_gain, average_loss = _ema(work, "gain", 14, wilder=True), _ema(work, "loss", 14, wilder=True)
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).where(average_loss.gt(0), 100.0).where(average_gain.notna())
    basis = _rolling(work, "close", 20, "mean")
    deviation = _rolling(work, "close", 20, "std")
    bollinger_lower = basis - 2 * deviation
    mean_state = _sequential_states(
        work, upper=basis, lower=bollinger_lower, basis=basis, rsi=rsi, atr=atr, kind="mean_reversion",
    )
    mean_raw = (basis - close) / deviation.replace(0, np.nan) - rsi / 100

    work["channel_high"] = high.groupby(ordered["stock"], sort=False).transform(
        lambda values: values.rolling(55, min_periods=55).max().shift(),
    )
    work["channel_low"] = low.groupby(ordered["stock"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=20).min().shift(),
    )
    breakout_state = _sequential_states(
        work, upper=work["channel_high"], lower=work["channel_low"], basis=basis, rsi=rsi, atr=atr, kind="breakout",
    )
    breakout_raw = close / work["channel_high"] - 1 + breakout_state.fillna(0) * 0.001

    momentum = close / close.groupby(ordered["stock"], sort=False).shift(20) - 1
    work["volatility"] = returns.groupby(ordered["stock"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=20).std(),
    )
    momentum_raw = momentum / work["volatility"].replace(0, np.nan)
    momentum_rank = momentum_raw.groupby(ordered["date"], sort=False).rank(pct=True)
    momentum_state = pd.Series(
        np.select([momentum_rank.ge(0.90), momentum_rank.ge(0.75), momentum_rank.le(0.25)], [2, 1, -1], default=0),
        index=ordered.index, dtype=float,
    ).where(momentum_rank.notna())

    typical = (high + low + close) / 3
    money = typical * volume
    direction = typical.groupby(ordered["stock"], sort=False).diff()
    work["positive_money"] = money.where(direction.gt(0), 0.0).where(direction.notna())
    work["negative_money"] = money.where(direction.lt(0), 0.0).where(direction.notna())
    positive_money = _rolling(work, "positive_money", 14, "sum")
    negative_money = _rolling(work, "negative_money", 14, "sum")
    mfi = 100 - 100 / (1 + positive_money / negative_money.replace(0, np.nan))
    work["price_volume"] = close * volume
    vwma = _rolling(work, "price_volume", 20, "sum") / _rolling(work, "volume", 20, "sum").replace(0, np.nan)
    sma = _rolling(work, "close", 20, "mean")
    difference = vwma - sma
    prior_difference = difference.groupby(ordered["stock"], sort=False).shift()
    knn_state = pd.Series(0.0, index=ordered.index).where(vwma.notna())
    knn_state = knn_state.mask(prior_difference.le(0) & difference.gt(0), 1.0)
    knn_state = knn_state.mask(prior_difference.ge(0) & difference.lt(0), -1.0)
    work["volume_mean"] = _rolling(work, "volume", 5, "mean")
    volume_change = volume / work["volume_mean"].replace(0, np.nan) - 1
    knn_raw = knn_state.fillna(0) * (adx.fillna(0) / 100 + (vwma / sma - 1).abs().fillna(0))
    knn_raw = knn_raw.where(knn_state.notna())

    result = pd.DataFrame({
        "donchian_breakout_state": breakout_state,
        "donchian_breakout_raw": breakout_raw,
        "rsi_bollinger_mean_reversion_state": mean_state,
        "rsi_bollinger_mean_reversion_raw": mean_raw,
        "ma_adx_trend_state": trend_state,
        "ma_adx_trend_raw": trend_raw.where(trend_ready),
        "momentum_rotation_state": momentum_state,
        "momentum_rotation_raw": momentum_raw,
        "online_knn_volume_trend_state": knn_state,
        "online_knn_volume_trend_raw": knn_raw,
        "adx": adx,
        "mfi": mfi,
        "volume_change": volume_change,
    }, index=ordered.index)
    majority_states = [result[f"{name}_state"].clip(-1, 1) for name in BASE_EXPERTS]
    majority_ready = sum(state.notna().astype(int) for state in majority_states)
    result["majority_vote_state"] = sum(state.fillna(0) for state in majority_states).clip(-2, 2).where(majority_ready.ge(2))
    ranks = [result[f"{name}_raw"].groupby(ordered["date"], sort=False).rank(pct=True) for name in BASE_EXPERTS]
    result["majority_vote_raw"] = pd.concat(ranks, axis=1).mean(axis=1).where(majority_ready.ge(2))
    return result


def _conditional_quantiles(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    signals: pd.DataFrame,
    name: str,
) -> ExpertForecast:
    pooled = train["target"].quantile(list(QUANTILE_LEVELS)).to_numpy(dtype=float)
    states = signals[f"{name}_state"]
    raw = signals[f"{name}_raw"]
    train_state = states.loc[train.index]
    valid = train["target"].notna() & train_state.notna()
    evidence_states = train_state.loc[valid]
    target = train.loc[valid, "target"]

    def grouped(values: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        table = target.groupby(values, observed=True)
        return table.quantile(list(QUANTILE_LEVELS)).unstack(), table.size()

    if valid.any():
        exact, exact_count = grouped(evidence_states)
        signed, signed_count = grouped(evidence_states.map(lambda value: int(np.sign(value))))
    else:
        exact = signed = pd.DataFrame(columns=QUANTILE_LEVELS, dtype=float)
        exact_count = signed_count = pd.Series(dtype=int)
    row_states = states.loc[rows.index]
    exact_values = exact.reindex(row_states).to_numpy(dtype=float)
    exact_ok = exact_count.reindex(row_states).fillna(0).to_numpy() >= MIN_GROUP
    signed_values = signed.reindex(row_states.map(lambda value: int(np.sign(value)) if pd.notna(value) else np.nan)).to_numpy(dtype=float)
    signed_ok = signed_count.reindex(row_states.map(lambda value: int(np.sign(value)) if pd.notna(value) else np.nan)).fillna(0).to_numpy() >= MIN_GROUP
    values = np.where(exact_ok[:, None], exact_values, np.where(signed_ok[:, None], signed_values, pooled))
    if not np.isfinite(pooled).all():
        raise ValueError(f"{name} lacks finite pooled training quantiles")
    values = np.where(np.isfinite(values), values, pooled)
    evidence = {
        "expert": name, "score_semantics": "excess_return_quantiles", "min_group_rows": MIN_GROUP,
        "train_state_rows": int(valid.sum()), "evaluation_rows": len(rows),
        "warmup_rows": int(row_states.isna().sum()), "unique_states": int(evidence_states.nunique()),
        "exact_group_rows": int(exact_ok.sum()), "signed_fallback_rows": int((~exact_ok & signed_ok).sum()),
        "pooled_fallback_rows": int((~exact_ok & ~signed_ok).sum()),
        "positive_q50_rate": float(np.mean(values[:, 1] > 0)),
    }
    return ExpertForecast(
        values,
        pd.Series(raw.loc[rows.index].to_numpy(dtype=float), index=rows.index, name=name),
        row_states.copy(),
        evidence,
    )


def _knn_forecast(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    device: str = "cpu",
    query_chunk: int = 256,
) -> ExpertForecast:
    name = "online_knn_volume_trend"
    base = _conditional_quantiles(train, rows, signals, name)
    combined = pd.concat([train, rows]).sort_values(["date", "stock"], kind="stable")
    states = signals[f"{name}_state"]
    features = signals[["adx", "mfi", "volume_change"]]
    finite = pd.Series(np.isfinite(features.to_numpy(dtype=float)).all(axis=1), index=features.index)
    events = states.loc[combined.index].isin((-1, 1)) & finite.loc[combined.index]
    samples = combined.loc[events & combined["target"].notna() & combined["label_end"].notna()].copy()
    sample_features = features.loc[samples.index].to_numpy(dtype=float)
    initial = samples.index.isin(train.index)
    if initial.sum() < MIN_GROUP:
        base.evidence.update(knn_rows=0, online_updates=0, max_available_samples=int(initial.sum()), minimum_samples=MIN_GROUP)
        return base
    backend = KNNBackend(device, query_chunk=query_chunk)
    center = sample_features[initial].mean(axis=0)
    scale = sample_features[initial].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    sample_features = (sample_features - center) / scale
    sample_targets = samples["target"].to_numpy(dtype=float)
    sample_ends = pd.to_datetime(samples["label_end"]).to_numpy()
    sample_dates = pd.to_datetime(samples["date"]).to_numpy()
    sample_states = states.loc[samples.index].to_numpy(dtype=float)
    values = base.quantiles.copy()
    used = 0
    seen_updates: set[int] = set()
    row_positions = pd.Series(np.arange(len(rows)), index=rows.index)
    max_available = int(initial.sum())
    for day in pd.DatetimeIndex(rows["date"].unique()):
        available = (sample_ends < day.to_datetime64()) & (sample_dates < day.to_datetime64())
        available_indices = np.flatnonzero(available)[-2000:]
        max_available = max(max_available, len(available_indices))
        seen_updates.update(np.flatnonzero(available & ~initial).tolist())
        query_index = rows.index[rows["date"].eq(day)]
        query_index = query_index[states.loc[query_index].isin((-1, 1))]
        query_index = query_index[np.isfinite(features.loc[query_index].to_numpy(dtype=float)).all(axis=1)]
        if not len(query_index) or not len(available_indices):
            continue
        queries = (features.loc[query_index].to_numpy(dtype=float) - center) / scale
        query_states = states.loc[query_index].to_numpy(dtype=float)
        for state in (-1.0, 1.0):
            positions = np.flatnonzero(query_states == state)
            same = available_indices[sample_states[available_indices] == state]
            if not len(positions) or len(same) < MIN_GROUP:
                continue
            quantiles = backend.quantiles(sample_features[same], sample_targets[same], queries[positions])
            output_positions = row_positions.loc[query_index[positions]].to_numpy(dtype=int)
            values[output_positions] = quantiles
            used += len(positions)
    base.quantiles[:] = values
    base.evidence.update(
        knn_rows=used, online_updates=len(seen_updates), max_available_samples=max_available,
        minimum_samples=MIN_GROUP, neighbors=12, max_available_history=2000,
        scaler="initial_train_only", update_rule="label_end_strictly_before_signal_date",
        backend=backend.evidence,
    )
    return base


def defensive_exposure(benchmark_returns: pd.Series) -> pd.DataFrame:
    """Pre-registered causal market overlay; it never changes stock scores."""
    returns = benchmark_returns.copy()
    returns.index = pd.to_datetime(returns.index)
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Defensive exposure needs unique, ordered benchmark sessions")
    returns = pd.to_numeric(returns, errors="coerce")
    trend = (1 + returns.fillna(0)).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    volatility = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    reference = volatility.expanding(min_periods=60).median().shift()
    ready = trend.notna() & volatility.notna() & reference.notna()
    risks = trend.lt(0).astype(int) + volatility.gt(reference).astype(int)
    exposure = pd.Series(np.select([risks.eq(2), risks.eq(1)], [0.0, 0.5], default=1.0), index=returns.index)
    exposure = exposure.where(ready, 0.5)
    state = pd.Series(np.select([risks.eq(2), risks.eq(1)], ["defensive_cash", "reduced"], default="full"), index=returns.index)
    return pd.DataFrame({
        "target_exposure": exposure,
        "benchmark_trend_20": trend,
        "benchmark_volatility_20": volatility,
        "volatility_reference": reference,
        "defensive_state": state.where(ready, "warmup"),
    })


def forecast_experts(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    names: tuple[str, ...] = ALL_EXPERTS,
    *,
    knn_device: str = "cpu",
    knn_query_chunk: int = 256,
) -> dict[str, ExpertForecast]:
    """Fit only on mature prior labels and return forecasts in the original row order."""
    if not names or set(names).difference(ALL_EXPERTS):
        raise ValueError("Unknown or empty signal expert set")
    if train.empty or rows.empty or train.index.has_duplicates or rows.index.has_duplicates:
        raise ValueError("Signal experts need nonempty, uniquely indexed train and evaluation rows")
    if train.index.intersection(rows.index).any() or train[["date", "stock"]].duplicated().any() or rows[["date", "stock"]].duplicated().any():
        raise ValueError("Signal experts need disjoint, unique stock/date rows")
    if train["label_end"].max() >= rows["date"].min():
        raise ValueError("Training labels must mature before the first evaluation date")
    combined = pd.concat([train, rows])
    signals = _signals(combined)
    forecasts = {}
    for name in names:
        forecasts[name] = _knn_forecast(train, rows, signals, device=knn_device, query_chunk=knn_query_chunk) if name == "online_knn_volume_trend" else _conditional_quantiles(
            train, rows, signals, name,
        )
    return forecasts
