"""Common quantile scoring and paired, trading-day block uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = np.array([0.10, 0.50, 0.90])
QUANTILES = ["q10", "q50", "q90"]


def score_quantiles(rows: pd.DataFrame) -> dict:
    """Score complete, shared labels with equal weight per trading session."""
    scored = rows.dropna(subset=["target", *QUANTILES])
    if scored.empty:
        return {"rows": 0, "dates": 0, "pinball": None, "coverage_80": None, "width_80": None}
    pred = scored[QUANTILES].to_numpy()
    error = scored["target"].to_numpy()[:, None] - pred
    loss = np.maximum(ALPHAS * error, (ALPHAS - 1) * error)
    stats = pd.DataFrame(loss, columns=QUANTILES, index=scored.index)
    stats["date"] = scored["date"]
    stats["covered"] = (error[:, 0] >= 0) & (error[:, 2] <= 0)
    stats["width"] = pred[:, 2] - pred[:, 0]
    per_day = stats.groupby("date").mean()
    means = per_day.mean()
    return {
        "rows": len(scored),
        "dates": len(per_day),
        "pinball": float(means[QUANTILES].mean()),
        "pinball_by_quantile": {key: float(means[key]) for key in QUANTILES},
        "coverage_80": float(means["covered"]),
        "width_80": float(means["width"]),
    }


def equity_returns(daily: pd.DataFrame, initial_cash: float) -> pd.Series:
    values = daily.set_index(pd.to_datetime(daily["date"]))["equity"]
    prior = values.shift(1, fill_value=initial_cash)
    return values.div(prior).sub(1).rename("net_return")


def paired_block_interval(
    periods: list[tuple[pd.Series, pd.Series]], *, seed: int,
    block_size: int = 10, draws: int = 2_000,
) -> dict:
    """Resample common trading-day blocks within each fold and pair both arms."""
    rng = np.random.default_rng(seed)
    actual_log = np.zeros(2)
    samples_log = np.zeros((draws, 2))
    days = 0
    for candidate, reference in periods:
        if not candidate.index.equals(reference.index) or not candidate.index.is_unique:
            raise ValueError("Paired equity series must share unique trading dates")
        returns = np.column_stack([candidate, reference])
        if not np.isfinite(returns).all() or (returns <= -1).any():
            raise ValueError("Paired returns must be finite and greater than -1")
        count = len(returns)
        if count < block_size:
            raise ValueError("A fold must cover at least one full trading-day block")
        log_returns = np.log1p(returns)
        starts = rng.integers(0, count, size=(draws, (count + block_size - 1) // block_size))
        indices = ((starts[:, :, None] + np.arange(block_size)) % count).reshape(draws, -1)[:, :count]
        actual_log += log_returns.sum(axis=0)
        samples_log += log_returns[indices].sum(axis=1)
        days += count
    actual = np.expm1(actual_log)
    samples = np.expm1(samples_log)
    lower, upper = np.quantile(samples[:, 0] - samples[:, 1], [0.025, 0.975])
    return {
        "net_return_difference": float(actual[0] - actual[1]),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "days": days,
        "block_sessions": block_size,
        "draws": draws,
    }


def summarize_regime_returns(returns: pd.Series, regimes: pd.DataFrame) -> dict:
    """Attribute daily P&L to the regime observable at the prior close."""
    result = {}
    for factor in regimes.columns:
        result[factor] = {}
        for name in ("low", "high"):
            values = returns.loc[regimes[factor].reindex(returns.index).eq(name)]
            result[factor][name] = {
                "days": len(values),
                "net_return": float((1 + values).prod() - 1) if len(values) else None,
            }
    return result
