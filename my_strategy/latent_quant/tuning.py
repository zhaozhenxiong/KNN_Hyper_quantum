"""Train-only overseas PCA and chronological CatBoost model selection."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from my_strategy.research_workbench.runner import Window, write_json
from my_strategy.validation.leakage import validate_feature_columns_no_lookahead

from .evaluation import score_quantiles
from .models import QUANTILE_COLUMNS, _calibrate, _crossing_rate, _matrix, _prediction_table, _sample_evidence


GRID = tuple(product((4, 6), (0.03, 0.08)))
ITERATIONS = 180
EARLY_STOPPING_ROUNDS = 20
_THREADS = 4
_LOSS = "MultiQuantile:alpha=0.1,0.5,0.9"
_RESERVED = {"date", "stock", "target", "label_end", "trade_date", "signal_eligible"}


def _features(columns: Sequence[str]) -> list[str]:
    values = list(columns)
    if not values or len(set(values)) != len(values) or _RESERVED.intersection(values):
        raise ValueError("Feature columns must be distinct, nonempty and exclude label fields")
    if not validate_feature_columns_no_lookahead(values)["ok"]:
        raise ValueError("Feature names cannot refer to future information")
    return values


def _eligible(frame: pd.DataFrame) -> pd.Series:
    return frame["signal_eligible"].fillna(False).astype(bool)


def _settled(frame: pd.DataFrame) -> pd.Series:
    return _eligible(frame) & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])


def _daily_inputs(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if frame["date"].isna().any():
        raise ValueError("Overseas inputs require a date on every signal row")
    daily = frame[["date", *features]].drop_duplicates()
    if daily["date"].duplicated().any():
        raise ValueError("Overseas PCA needs a single shared observation per signal date")
    return daily.set_index("date").sort_index().replace([np.inf, -np.inf], np.nan)


def _standardized(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), (values - centers) / scales, 0.0)


def apply_fold_pca(frame: pd.DataFrame, metadata: Mapping) -> pd.DataFrame:
    """Apply a frozen fold transform without fitting or consulting targets."""
    daily = _daily_inputs(frame, list(metadata["input_features"]))
    values = daily.to_numpy(dtype=float, na_value=np.nan)
    standardized = _standardized(values, np.asarray(metadata["centers"]), np.asarray(metadata["scales"]))
    scores = standardized @ np.asarray(metadata["components"]).T
    scores[~np.isfinite(values).any(axis=1)] = np.nan
    components = pd.DataFrame(scores, index=daily.index, columns=metadata["output_features"])
    return frame.assign(**{name: frame["date"].map(components[name]) for name in components})


def add_fold_pca(
    frame: pd.DataFrame, window: Window, continuous: Sequence[str], *, n_components: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Fit standardized PCA on one overseas observation per purged training date."""
    features = _features(continuous)
    if not 1 <= n_components <= len(features):
        raise ValueError("PCA components must be between 1 and the number of inputs")
    train_mask, _, _ = window.masks(frame)
    daily = _daily_inputs(frame.loc[train_mask & _settled(frame)], features)
    observed = daily.notna().sum()
    if len(daily) <= n_components or observed.lt(2).any():
        raise ValueError(f"{window.period} lacks distinct training-date observations for overseas PCA")
    centers = daily.mean().to_numpy(dtype=float)
    scales = daily.std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    standardized = _standardized(daily.to_numpy(dtype=float, na_value=np.nan), centers, scales)
    _, singular, basis = np.linalg.svd(standardized, full_matrices=False)
    tolerance = max(standardized.shape) * np.finfo(float).eps * singular[0]
    if singular[n_components - 1] <= tolerance:
        raise ValueError(f"{window.period} overseas PCA has fewer independent inputs than components")
    components = basis[:n_components]
    pivots = np.abs(components).argmax(axis=1)
    components = components * np.sign(components[np.arange(n_components), pivots])[:, None]
    variance = singular**2
    metadata = {
        "input_features": features, "output_features": [f"overseas_pc{i + 1}" for i in range(n_components)],
        "train_dates": len(daily), "train_first_date": daily.index[0].date().isoformat(),
        "train_last_date": daily.index[-1].date().isoformat(), "observed_train_dates": observed.to_dict(),
        "centers": centers.tolist(), "scales": scales.tolist(), "components": components.tolist(),
        "explained_variance_ratio": (variance[:n_components] / variance.sum()).tolist(),
        "missing": "Training column mean for partially missing dates. All-missing dates remain missing.",
    }
    return apply_fold_pca(frame, metadata), metadata


def _tune_arm(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int) -> tuple[Any, dict]:
    import catboost

    x_train = _matrix(train, features)
    x_valid = _matrix(valid, features)
    date_weights = 1.0 / valid.groupby("date")["date"].transform("size")
    evaluation_set = catboost.Pool(x_valid, label=valid["target"], weight=date_weights)
    trials = []
    best_model = None
    winner = None
    for depth, rate in GRID:
        params = {
            "loss_function": _LOSS, "eval_metric": _LOSS, "task_type": "CPU", "iterations": ITERATIONS,
            "depth": depth, "learning_rate": rate, "random_seed": seed, "thread_count": _THREADS,
            "allow_writing_files": False,
        }
        model = catboost.CatBoostRegressor(**params)
        model.fit(
            x_train, train["target"], eval_set=evaluation_set,
            use_best_model=True, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False,
        )
        raw = model.predict(evaluation_set, thread_count=_THREADS)
        predicted = pd.DataFrame(raw, index=valid.index, columns=QUANTILE_COLUMNS)
        scores = score_quantiles(valid[["date", "target"]].join(predicted))
        stopping_score = float(model.get_best_score()["validation"][_LOSS])
        if scores["rows"] != len(valid) or not np.isfinite([scores["pinball"], stopping_score]).all():
            raise ValueError("Validation scoring requires finite predictions on the common sample")
        trial = {
            "params": params, "best_iteration": int(model.get_best_iteration()), "tree_count": int(model.tree_count_),
            "validation_pinball": stopping_score, "validation_recomputed_pinball": scores["pinball"],
            "validation_pinball_by_quantile": scores["pinball_by_quantile"],
            "validation_raw_crossing_rate": _crossing_rate(raw),
        }
        trials.append(trial)
        if winner is None or trial["validation_pinball"] < winner["validation_pinball"]:
            best_model, winner = model, trial
    return best_model, {
        "catboost_version": catboost.__version__, "features": features,
        "candidates": trials, "winner": winner, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "validation_selection": "CatBoost MultiQuantile on raw q10/q50/q90 with equal weight per session, also used for early stopping.",
    }


def tune_fold(
    frame: pd.DataFrame, window: Window, feature_sets: Mapping[str, Sequence[str]],
    *, seed: int, model_dir: Path | None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Select each arm on purged validation days, then calibrate and predict shared OOS rows."""
    if not feature_sets or any(not re.fullmatch(r"[a-z][a-z0-9_]*", arm) for arm in feature_sets):
        raise ValueError("Tuning requires one or more named arms using lowercase letters, digits and underscores")
    if not frame.index.is_unique or frame.duplicated(["date", "stock"]).any():
        raise ValueError("Tuning requires unique row indices and stock-date keys")
    settled = _settled(frame)
    train, valid, calibration = (frame.loc[mask & settled] for mask in window.masks(frame))
    evaluation = frame.loc[_eligible(frame) & frame["date"].isin(window.evaluate)]
    if any(rows.empty for rows in (train, valid, calibration, evaluation)):
        raise ValueError(f"{window.period} requires nonempty train, validation, calibration and evaluation samples")
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([calibration, evaluation])
    labels = calibration["target"].to_numpy(dtype=float)
    predictions = {}
    model_evidence = {}
    for arm, columns in feature_sets.items():
        features = _features(columns)
        model, evidence = _tune_arm(train, valid, features, seed)
        raw = model.predict(_matrix(rows, features), thread_count=_THREADS)
        if not np.isfinite(raw).all():
            raise ValueError(f"{arm} returned non-finite calibration or evaluation predictions")
        calibrated, evidence["calibration"] = _calibrate(raw, labels)
        predictions[arm] = _prediction_table(calibrated, evaluation.index)
        if model_dir is not None:
            model_path = model_dir / f"{arm}.cbm"
            temporary = model_path.with_suffix(".cbm.tmp")
            model.save_model(str(temporary), format="cbm")
            temporary.replace(model_path)
            metadata_path = model_path.with_suffix(".json")
            evidence.update(model_file=model_path.name, metadata_file=metadata_path.name)
            write_json(metadata_path, evidence)
        model_evidence[arm] = evidence
    return predictions, {
        "train": _sample_evidence(train), "validation": _sample_evidence(valid),
        "calibration": _sample_evidence(calibration), "evaluation_rows": len(evaluation),
        "arms": model_evidence,
    }
