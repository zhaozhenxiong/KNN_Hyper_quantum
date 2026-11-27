"""Promotion gates for candidate models and their point-in-time signals."""

from __future__ import annotations

from typing import Any, Mapping

from my_strategy.model_registry.schema import ModelEntry


def model_quality(entry: ModelEntry) -> float:
    """Return a compact out-of-sample quality score from registry metrics."""

    metrics = entry.metrics or {}
    if "score" in metrics:
        return _number(metrics.get("score"))
    test = metrics.get("test_metrics") if isinstance(metrics.get("test_metrics"), Mapping) else metrics
    return (
        0.60 * _number(test.get("daily_rank_ic_mean"))
        + 0.25 * _number(test.get("top_bottom_spread"))
        + 0.15 * max(-1.0, min(2.0, _number(test.get("stock_rank_ic_ir")))) / 2.0
    )


def assess_model_promotion(
    *,
    candidate: ModelEntry,
    champion: ModelEntry | None,
    signal_stability: Mapping[str, Any],
    min_test_stocks: int = 500,
    min_test_days: int = 60,
    min_quality_delta: float = 0.0,
    min_ic_delta: float = 0.0,
    min_matched_signals: int = 20,
    max_action_flip_rate: float = 0.35,
    min_walk_forward_folds: int = 2,
    min_walk_forward_positive_fold_rate: float = 0.55,
    min_walk_forward_mean_delta: float = 0.0,
    min_walk_forward_holdout_delta: float = 0.0,
    walk_forward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate without changing model lifecycle state."""

    candidate_metrics = _test_metrics(candidate)
    champion_metrics = _test_metrics(champion) if champion else {}
    candidate_quality = model_quality(candidate)
    champion_quality = model_quality(champion) if champion else 0.0
    candidate_ic = _number(candidate_metrics.get("daily_rank_ic_mean"))
    champion_ic = _number(champion_metrics.get("daily_rank_ic_mean"))
    matched_signals = int(signal_stability.get("matched", 0) or 0)
    action_flip_rate = _number(signal_stability.get("action_flip_rate"), 1.0)
    walk_forward_summary = _walk_forward_summary(walk_forward or candidate_metrics.get("walk_forward"))
    checks = {
        "candidate_is_not_champion": bool(champion is None or candidate.model_id != champion.model_id),
        "test_stock_coverage": int(_number(candidate_metrics.get("test_stocks", candidate_metrics.get("stocks")))) >= min_test_stocks,
        "test_day_coverage": int(_number(candidate_metrics.get("daily_rank_ic_days"))) >= min_test_days,
        "quality_delta": candidate_quality - champion_quality >= min_quality_delta,
        "rank_ic_delta": candidate_ic - champion_ic >= min_ic_delta,
        "signal_coverage": matched_signals >= min_matched_signals,
        "signal_stability": action_flip_rate <= max_action_flip_rate,
        "walk_forward_folds": walk_forward_summary["fold_count"] >= min_walk_forward_folds,
        "walk_forward_positive_fold_rate": walk_forward_summary["positive_fold_rate"] >= min_walk_forward_positive_fold_rate,
        "walk_forward_mean_delta": walk_forward_summary["mean_oos_delta"] >= min_walk_forward_mean_delta,
        "walk_forward_holdout_delta": walk_forward_summary["holdout_delta"] >= min_walk_forward_holdout_delta,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "candidate": {"model_id": candidate.model_id, "name": candidate.name, "quality": candidate_quality, "test_metrics": candidate_metrics},
        "champion": (
            {"model_id": champion.model_id, "name": champion.name, "quality": champion_quality, "test_metrics": champion_metrics}
            if champion
            else None
        ),
        "deltas": {"quality": candidate_quality - champion_quality, "daily_rank_ic": candidate_ic - champion_ic},
        "signal_stability": dict(signal_stability),
        "walk_forward": walk_forward_summary,
        "thresholds": {
            "min_test_stocks": min_test_stocks,
            "min_test_days": min_test_days,
            "min_quality_delta": min_quality_delta,
            "min_ic_delta": min_ic_delta,
            "min_matched_signals": min_matched_signals,
            "max_action_flip_rate": max_action_flip_rate,
            "min_walk_forward_folds": min_walk_forward_folds,
            "min_walk_forward_positive_fold_rate": min_walk_forward_positive_fold_rate,
            "min_walk_forward_mean_delta": min_walk_forward_mean_delta,
            "min_walk_forward_holdout_delta": min_walk_forward_holdout_delta,
        },
    }


def _test_metrics(entry: ModelEntry | None) -> dict[str, Any]:
    if entry is None:
        return {}
    metrics = entry.metrics or {}
    nested = metrics.get("test_metrics")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    # Candidate registration stores the full test block plus registry-level
    # coverage/selection fields.  Preserve both when evaluating a gate.
    for key, value in metrics.items():
        if key not in result:
            result[key] = value
    if "test_stocks" not in result and "stocks" not in result:
        data_range = entry.data_range or {}
        if "test_stocks" in data_range:
            result["test_stocks"] = data_range["test_stocks"]
    return result


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _walk_forward_summary(value: Any) -> dict[str, float | int]:
    """Normalize a model walk-forward report into promotion-gate evidence.

    The accepted schema deliberately matches the existing fusion walk-forward
    report: ``rolling_folds``, ``mean_oos_score_delta``,
    ``positive_fold_rate`` and ``holdout.score_delta``.  Model-specific
    runners may also use the shorter normalized keys below.
    """

    data = value if isinstance(value, Mapping) else {}
    folds = data.get("rolling_folds")
    fold_count = int(_number(data.get("fold_count"), len(folds) if isinstance(folds, list) else 0))
    return {
        "fold_count": fold_count,
        "positive_fold_rate": _number(data.get("positive_fold_rate")),
        "mean_oos_delta": _number(data.get("mean_oos_delta", data.get("mean_oos_score_delta"))),
        "holdout_delta": _number(
            data.get("holdout_delta", data.get("holdout_score_delta", (data.get("holdout") or {}).get("score_delta")))
            if isinstance(data.get("holdout"), Mapping)
            else data.get("holdout_delta", data.get("holdout_score_delta"))
        ),
    }
