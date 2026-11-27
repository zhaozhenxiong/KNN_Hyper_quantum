"""Regularized expert stacking and shadow-only exposure learning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


QUANTILE_LEVELS = (0.10, 0.50, 0.90)
EXPOSURES = (0.0, 0.5, 1.0)


def pinball_loss(target: np.ndarray, predicted: np.ndarray, date_weights: np.ndarray | None = None) -> float:
    error = np.asarray(target, dtype=float)[:, None] - np.asarray(predicted, dtype=float)
    losses = np.maximum(np.asarray(QUANTILE_LEVELS) * error, (np.asarray(QUANTILE_LEVELS) - 1) * error).mean(axis=1)
    if date_weights is not None:
        return float(np.dot(losses, date_weights))
    return float(losses.mean())


def _date_weights(dates: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.Series(dates).reset_index(drop=True)
    return (1 / values.groupby(values).transform("size")).to_numpy(dtype=float) / len(values.unique())


def _sample_rows(dates: pd.Series, cap: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for _, positions in pd.Series(np.arange(len(dates))).groupby(dates.reset_index(drop=True), sort=True):
        values = positions.to_numpy()
        selected.extend(rng.choice(values, min(cap, len(values)), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _fit_weights(
    target: np.ndarray,
    forecasts: dict[str, np.ndarray],
    dates: pd.Series,
    anchor: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    names = tuple(forecasts)
    stack = np.stack([np.sort(forecasts[name], axis=1) for name in names], axis=1)
    sample = _sample_rows(dates)
    sampled_dates = dates.reset_index(drop=True).iloc[sample]
    weights_by_date = _date_weights(sampled_dates)
    sampled_target, sampled_stack = target[sample], stack[sample]

    def objective(weights: np.ndarray) -> float:
        predicted = np.einsum("e,neq->nq", weights, sampled_stack)
        return pinball_loss(sampled_target, predicted, weights_by_date) + regularization * float(np.sum((weights - anchor) ** 2))

    result = minimize(
        objective,
        anchor,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 100, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        weights, fallback = anchor.copy(), str(result.message)
    else:
        weights, fallback = np.clip(result.x, 0, 1), None
        weights /= weights.sum()
    predicted = np.einsum("e,neq->nq", weights, stack)
    evidence = {
        "optimizer": "SLSQP", "success": bool(result.success), "fallback": fallback,
        "regularization": regularization, "fit_rows": len(sample), "fit_dates": int(sampled_dates.nunique()),
        "all_rows_date_equal_pinball": pinball_loss(target, predicted, _date_weights(dates)),
        "objective": float(objective(weights)),
    }
    return weights, evidence


def fit_dynamic_fusion(
    rows: pd.DataFrame,
    regimes: pd.Series,
    forecasts: dict[str, np.ndarray],
) -> dict:
    """Fit only on supplied OOF rows; no chronological splitting happens here."""
    if len(forecasts) < 2 or rows.empty:
        raise ValueError("Dynamic fusion needs OOF labels and at least two experts")
    names = tuple(forecasts)
    length = len(rows)
    if any(np.asarray(values).shape != (length, 3) for values in forecasts.values()):
        raise ValueError("Every expert must supply one ordered quantile triplet per OOF row")
    regime = regimes.reindex(rows.index).fillna(False).astype(bool).to_numpy()
    target = rows["target"].to_numpy(dtype=float)
    finite = np.isfinite(target)
    for values in forecasts.values():
        finite &= np.isfinite(values).all(axis=1)
    if finite.sum() < 100 or rows.loc[finite, "date"].nunique() < 20:
        raise ValueError("Dynamic fusion has insufficient finite OOF evidence")
    fit_rows = rows.loc[finite]
    fit_target = target[finite]
    fit_regime = regime[finite]
    fit_forecasts = {name: np.sort(np.asarray(values)[finite], axis=1) for name, values in forecasts.items()}
    equal = np.full(len(names), 1 / len(names))
    global_weights, global_evidence = _fit_weights(fit_target, fit_forecasts, fit_rows["date"], equal, 0.01)
    risk_rows = fit_rows.loc[fit_regime]
    risk_dates = risk_rows["date"].nunique()
    risk_weights = global_weights.copy()
    risk_evidence = {"used": False, "reason": "fewer_than_20_oof_dates", "dates": int(risk_dates)}
    shrinkage = 0.0
    if risk_dates >= 20 and len(risk_rows) >= 500:
        learned, risk_evidence = _fit_weights(
            fit_target[fit_regime], {name: values[fit_regime] for name, values in fit_forecasts.items()},
            risk_rows["date"], global_weights, 0.05,
        )
        shrinkage = min(0.75, risk_dates / (risk_dates + 40))
        risk_weights = (1 - shrinkage) * global_weights + shrinkage * learned
        risk_evidence.update(used=True, reason="", dates=int(risk_dates), shrinkage_to_global=shrinkage)
    singles = {
        name: pinball_loss(fit_target, values, _date_weights(fit_rows["date"]))
        for name, values in fit_forecasts.items()
    }
    best_single = min(singles, key=singles.get)
    return {
        "version": "regularized_binary_gate_v1", "experts": list(names),
        "global_weights": dict(zip(names, global_weights.tolist(), strict=True)),
        "risk_off_weights": dict(zip(names, risk_weights.tolist(), strict=True)),
        "gate": {"feature": "causal_market_risk_off", "minimum_dates": 20, "minimum_rows": 500},
        "evidence": {
            "oof_rows": int(finite.sum()), "oof_dates": int(fit_rows["date"].nunique()),
            "dropped_nonfinite_rows": int((~finite).sum()), "global": global_evidence, "risk_off": risk_evidence,
            "single_expert_pinball": singles, "best_single_expert": best_single,
            "equal_weights": dict(zip(names, equal.tolist(), strict=True)),
            "fallback_order": ["global_weights", "finite_expert_equal", "current_fusion"],
        },
    }


def dynamic_predict(
    model: dict,
    rows: pd.DataFrame,
    regimes: pd.Series,
    forecasts: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    names = tuple(model["experts"])
    if set(forecasts) != set(names):
        raise ValueError("Dynamic prediction views do not match the frozen OOF model")
    risk = regimes.reindex(rows.index).fillna(False).astype(bool).to_numpy()
    global_weights = np.array([model["global_weights"][name] for name in names])
    risk_weights = np.array([model["risk_off_weights"][name] for name in names])
    weights = np.repeat(global_weights[None, :], len(rows), axis=0)
    weights[risk] = risk_weights
    stack = np.stack([np.asarray(forecasts[name], dtype=float) for name in names], axis=1)
    finite = np.isfinite(stack).all(axis=2)
    complete = finite.all(axis=1)
    values = np.einsum("ne,neq->nq", weights, np.nan_to_num(stack, nan=0.0))
    fallback_equal = missing_all = 0
    for position in np.flatnonzero(~complete):
        available = finite[position]
        if available.any():
            values[position] = stack[position, available].mean(axis=0)
            fallback_equal += 1
        elif "current_fusion" in names and np.isfinite(forecasts["current_fusion"][position]).all():
            values[position] = forecasts["current_fusion"][position]
        else:
            missing_all += 1
    if missing_all:
        raise ValueError("No finite expert or current-fusion fallback for evaluation rows")
    values = np.sort(values, axis=1)
    daily = pd.DataFrame({"date": rows["date"].to_numpy(), "risk_off": risk})
    for index, name in enumerate(names):
        daily[f"weight_{name}"] = weights[:, index]
    daily = daily.groupby("date", sort=True).first().reset_index()
    evidence = {
        "rows": len(rows), "risk_off_rows": int(risk.sum()), "risk_off_dates": int(daily["risk_off"].sum()),
        "finite_expert_equal_fallback_rows": fallback_equal, "missing_all_rows": missing_all,
    }
    return values, daily, evidence


def fit_shadow_exposure(rows: pd.DataFrame, regimes: pd.Series, quantiles: np.ndarray) -> dict:
    """Tabular Q-learning on OOF top-10 excess returns; never emits stock orders."""
    table = rows[["date", "stock", "target"]].copy()
    table["score"] = quantiles[:, 1]
    table["risk_off"] = regimes.reindex(rows.index).fillna(False).astype(bool).to_numpy()
    table["positive"] = table["score"].gt(0)
    daily = table.groupby("date").agg(state_risk=("risk_off", "first"), positive=("positive", "mean"))
    ordered = table.sort_values(["date", "score", "stock"], ascending=[True, False, True])
    top = ordered.groupby("date").head(10)
    complete = top.groupby("date")["target"].count().eq(10)
    daily["reward"] = top.groupby("date")["target"].mean().where(complete)
    daily = daily.dropna(subset=["reward"])
    if len(daily) < 20:
        return {"enabled": False, "reason": "fewer_than_20_mature_oof_days", "exposures": list(EXPOSURES)}
    states = daily["state_risk"].astype(int).to_numpy() * 2 + daily["positive"].gt(0.5).astype(int).to_numpy()
    rewards = daily["reward"].to_numpy(dtype=float)
    q = np.zeros((4, len(EXPOSURES)))
    rng = np.random.default_rng(42)
    for _ in range(20):
        previous = 2
        for index, state in enumerate(states):
            action = int(rng.integers(len(EXPOSURES))) if rng.random() < 0.10 else int(np.argmax(q[state]))
            reward = EXPOSURES[action] * rewards[index] - 0.0005 * abs(EXPOSURES[action] - EXPOSURES[previous])
            future = 0 if index + 1 == len(states) else float(np.max(q[states[index + 1]]))
            q[state, action] += 0.10 * (reward + 0.80 * future - q[state, action])
            previous = action
    return {
        "enabled": True, "q_table": q.tolist(), "exposures": list(EXPOSURES),
        "training_dates": len(daily), "episodes": 20, "learning_rate": 0.10, "discount": 0.80,
        "epsilon": 0.10, "turnover_penalty": 0.0005, "seed": 42,
        "state_definition": "2*risk_off + (median_daily_q50_positive_breadth>0.5)",
        "execution_role": "shadow_only_no_stock_orders",
    }


def shadow_exposure_actions(model: dict, rows: pd.DataFrame, regimes: pd.Series, quantiles: np.ndarray) -> pd.DataFrame:
    if not model.get("enabled"):
        return pd.DataFrame(columns=["date", "state", "shadow_exposure", "q_value"])
    table = rows[["date"]].copy()
    table["score"] = quantiles[:, 1]
    table["risk_off"] = regimes.reindex(rows.index).fillna(False).astype(bool).to_numpy()
    daily = table.groupby("date").agg(risk_off=("risk_off", "first"), positive=("score", lambda values: values.gt(0).mean()))
    q = np.asarray(model["q_table"])
    records = []
    for day, row in daily.iterrows():
        state = int(row.risk_off) * 2 + int(row.positive > 0.5)
        action = int(np.argmax(q[state]))
        records.append({"date": day, "state": state, "shadow_exposure": EXPOSURES[action], "q_value": q[state, action]})
    return pd.DataFrame(records)
