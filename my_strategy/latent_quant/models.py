"""Fit and replay the four Latent-Quant prediction arms on a purged window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from my_strategy.research_workbench.runner import Window, write_json
from my_strategy.validation.leakage import validate_feature_columns_no_lookahead


ARMS = ("historical_quantiles", "supertrend_atr", "catboost_base", "catboost_supertrend_atr")
QUANTILE_LEVELS = (0.10, 0.50, 0.90)
QUANTILE_COLUMNS = ("q10", "q50", "q90")
_TREND_COLUMNS = ("st_direction", "atr_pct", "st_distance")
_MIN_GROUP = 20
_THREADS = 4


def _group_quantiles(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    target = rows.groupby(groups, observed=True)["target"]
    quantiles = target.quantile(list(QUANTILE_LEVELS)).unstack().reindex(columns=QUANTILE_LEVELS)
    return quantiles.loc[target.size().ge(_MIN_GROUP)]


def _historical_quantiles(
    train: pd.DataFrame, rows: pd.DataFrame, pooled: np.ndarray,
) -> tuple[np.ndarray, dict]:
    stock_quantiles = _group_quantiles(train, ["stock"])
    values = stock_quantiles.reindex(rows["stock"]).to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, pooled), {
        "min_group_rows": _MIN_GROUP, "stock_groups": len(stock_quantiles),
    }


def _trend_groups(rows: pd.DataFrame, cuts: np.ndarray) -> pd.DataFrame:
    direction = rows["st_direction"].where(rows["st_direction"].isin((-1, 1)))
    atr = rows["atr_pct"].where(rows["atr_pct"].gt(0) & np.isfinite(rows["atr_pct"]))
    return rows.assign(direction=direction, atr_band=pd.cut(atr, [-np.inf, *cuts, np.inf], labels=False))


def _trend_quantiles(
    train: pd.DataFrame, rows: pd.DataFrame, pooled: np.ndarray,
) -> tuple[np.ndarray, dict]:
    atr = train.loc[train["atr_pct"].gt(0) & np.isfinite(train["atr_pct"]), "atr_pct"]
    cuts = np.unique(atr.quantile([1 / 3, 2 / 3]).to_numpy()) if len(atr) else np.empty(0)
    train_groups = _trend_groups(train, cuts)
    predicted_groups = _trend_groups(rows, cuts)
    conditional = _group_quantiles(train_groups, ["direction", "atr_band"])
    by_direction = _group_quantiles(train_groups, ["direction"])
    group_keys = pd.MultiIndex.from_frame(predicted_groups[["direction", "atr_band"]])
    values = conditional.reindex(group_keys).to_numpy(dtype=float)
    default = by_direction.reindex(predicted_groups["direction"]).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, default)
    return np.where(np.isfinite(values), values, pooled), {
        "min_group_rows": _MIN_GROUP, "atr_tertile_boundaries": cuts.tolist(),
        "direction_atr_groups": len(conditional), "direction_groups": len(by_direction),
    }


def _trend_score(rows: pd.DataFrame) -> np.ndarray:
    direction, atr, distance = (rows[column].to_numpy(dtype=float) for column in _TREND_COLUMNS)
    ready = np.isin(direction, (-1, 1)) & np.isfinite(atr) & (atr > 0) & np.isfinite(distance)
    score = np.zeros(len(rows))
    score[ready] = direction[ready] * (1 + np.abs(distance[ready]) / atr[ready])
    return score


def _matrix(rows: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    return rows[list(features)].astype(np.float32).replace([np.inf, -np.inf], np.nan)


def _fit_catboost(
    train: pd.DataFrame, valid: pd.DataFrame, rows: pd.DataFrame,
    features: Sequence[str], *, seed: int, iterations: int,
) -> tuple[np.ndarray, Any, dict]:
    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("Latent-Quant requires CatBoost in the Python environment") from exc

    params = {
        "loss_function": "MultiQuantile:alpha=0.1,0.5,0.9", "task_type": "CPU",
        "iterations": iterations, "depth": 5, "learning_rate": 0.05,
        "random_seed": seed, "thread_count": _THREADS, "allow_writing_files": False,
    }
    model = catboost.CatBoostRegressor(**params)
    model.fit(
        _matrix(train, features), train["target"],
        eval_set=(_matrix(valid, features), valid["target"]),
        use_best_model=True, early_stopping_rounds=20, verbose=False,
    )
    values = model.predict(_matrix(rows, features), thread_count=_THREADS)
    return values, model, {
        "catboost_version": catboost.__version__, "params": params,
        "features": list(features), "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_), "early_stopping_rounds": 20,
    }


def _crossing_rate(values: np.ndarray) -> float | None:
    return float((np.diff(values, axis=1) < 0).any(axis=1).mean()) if len(values) else None


def _calibrate(raw: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    count = len(labels)
    residuals = labels[:, None] - np.sort(raw[:count], axis=1)
    offsets = np.array([np.quantile(residuals[:, i], level) for i, level in enumerate(QUANTILE_LEVELS)])
    evaluation = raw[count:]
    shifted = np.sort(evaluation, axis=1) + offsets
    calibrated = np.sort(shifted, axis=1)
    return calibrated, {
        "method": "additive_residual_quantiles",
        "offsets": offsets.tolist(), "rows": count,
        "raw_crossing_rate": _crossing_rate(evaluation),
        "crossing_rate_after_offsets": _crossing_rate(shifted),
        "final_crossing_rate": _crossing_rate(calibrated),
    }


def _prediction_table(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    table = pd.DataFrame(values, index=index, columns=QUANTILE_COLUMNS)
    table["score"] = table["q50"]
    return table


def _sample_evidence(rows: pd.DataFrame) -> dict:
    return {
        "rows": len(rows), "dates": int(rows["date"].nunique()),
        "start": rows["date"].min().date().isoformat(),
        "end": rows["date"].max().date().isoformat(),
        "last_label_end": rows["label_end"].max().date().isoformat(),
    }


def predict_fold(
    frame: pd.DataFrame, window: Window, base_features: Sequence[str], extra_features: Sequence[str],
    *, seed: int, model_dir: Path | None, iterations: int = 100,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return all four arms on the same eligible evaluation index, even before labels settle."""
    features = [*base_features, *extra_features]
    reserved = {"date", "stock", "target", "label_end", "trade_date", "signal_eligible"}
    if not base_features or len(set(features)) != len(features) or reserved.intersection(features):
        raise ValueError("Latent-Quant features must be nonempty, distinct and exclude label fields")
    if not validate_feature_columns_no_lookahead(features)["ok"]:
        raise ValueError("Latent-Quant feature names cannot refer to future information")
    if not frame.index.is_unique or frame.duplicated(["stock", "date"]).any():
        raise ValueError("Latent-Quant requires one row per stock and signal date")

    eligible = frame["signal_eligible"].fillna(False).astype(bool)
    settled = eligible & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])
    train, valid, calibration = (frame.loc[mask & settled] for mask in window.masks(frame))
    if train.empty or valid.empty or calibration.empty:
        raise ValueError(f"{window.period} requires settled training, validation and calibration rows")
    evaluation = frame.loc[eligible & frame["date"].isin(window.evaluate)]
    if evaluation.empty:
        raise ValueError(f"{window.period} has no eligible evaluation rows")

    rows = pd.concat([calibration, evaluation])
    pooled = train["target"].quantile(list(QUANTILE_LEVELS)).to_numpy()
    historical, historical_evidence = _historical_quantiles(train, rows, pooled)
    trend, trend_evidence = _trend_quantiles(train, rows, pooled)
    base, base_model, base_evidence = _fit_catboost(train, valid, rows, base_features, seed=seed, iterations=iterations)
    extended, extended_model, extended_evidence = _fit_catboost(train, valid, rows, features, seed=seed, iterations=iterations)
    arms = zip(
        ARMS,
        (historical, trend, base, extended),
        (None, None, base_model, extended_model),
        (historical_evidence, trend_evidence, base_evidence, extended_evidence),
        strict=True,
    )
    predictions = {}
    model_evidence = {}
    labels = calibration["target"].to_numpy(dtype=float)
    for arm, raw, model, evidence in arms:
        calibrated, adjustments = _calibrate(raw, labels)
        table = _prediction_table(calibrated, evaluation.index)
        evidence["calibration"] = adjustments
        if arm == "supertrend_atr":
            table["score"] = _trend_score(evaluation)
        if model is not None and model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{arm}.cbm"
            temporary = model_path.with_suffix(".cbm.tmp")
            model.save_model(str(temporary), format="cbm")
            temporary.replace(model_path)
            metadata_path = model_path.with_suffix(".json")
            evidence["model_file"] = model_path.name
            evidence["metadata_file"] = metadata_path.name
            write_json(metadata_path, evidence)
        predictions[arm] = table
        model_evidence[arm] = evidence
    return predictions, {
        "train": _sample_evidence(train), "validation": _sample_evidence(valid),
        "calibration": _sample_evidence(calibration), "evaluation_rows": len(evaluation),
        "arms": model_evidence,
    }


def replay_catboost(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Reproduce a CatBoost arm's calibrated quantiles for the supplied feature rows."""
    from catboost import CatBoostRegressor

    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    raw = model.predict(_matrix(frame, metadata["features"]), thread_count=_THREADS)
    offsets = np.asarray(metadata["calibration"]["offsets"])
    return _prediction_table(np.sort(np.sort(raw, axis=1) + offsets, axis=1), frame.index)
