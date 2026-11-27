"""Validation and comparison for point-in-time signal snapshots."""

from __future__ import annotations

from typing import Any, Mapping


COMPARISON_TYPES = {"exact_replay", "data_revision", "model_drift"}


def validate_signal_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum contract required for an auditable daily signal."""

    issues: list[str] = []
    signal = snapshot.get("signal") if isinstance(snapshot.get("signal"), Mapping) else {}
    lineage = snapshot.get("lineage") if isinstance(snapshot.get("lineage"), Mapping) else {}
    manifest = snapshot.get("data_manifest") if isinstance(snapshot.get("data_manifest"), Mapping) else {}
    for key in ["snapshot_id", "channel", "replay_fingerprint", "decision"]:
        if not snapshot.get(key):
            issues.append(f"missing:{key}")
    for key in ["symbol", "as_of_date", "signal", "data_version"]:
        if signal.get(key) in {None, ""}:
            issues.append(f"missing:signal.{key}")
    for key in ["as_of_date", "config_hash", "git_commit"]:
        if lineage.get(key) in {None, ""}:
            issues.append(f"missing:lineage.{key}")
    if not manifest.get("manifest_hash"):
        issues.append("missing:data_manifest.manifest_hash")
    if signal.get("as_of_date") and lineage.get("as_of_date") and signal["as_of_date"] != lineage["as_of_date"]:
        issues.append("as_of_date_mismatch")
    return {"ok": not issues, "issues": issues}


def compare_signal_snapshots(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    comparison_type: str,
) -> dict[str, Any]:
    """Compare a baseline and candidate without mutating either snapshot."""

    if comparison_type not in COMPARISON_TYPES:
        raise ValueError(f"unsupported comparison_type: {comparison_type}")
    baseline_validation = validate_signal_snapshot(baseline)
    candidate_validation = validate_signal_snapshot(candidate)
    baseline_signal = baseline.get("signal", {})
    candidate_signal = candidate.get("signal", {})
    baseline_manifest = baseline.get("data_manifest", {})
    candidate_manifest = candidate.get("data_manifest", {})
    baseline_models = baseline.get("model_versions", {})
    candidate_models = candidate.get("model_versions", {})

    action_changed = baseline_signal.get("signal") != candidate_signal.get("signal")
    score_delta = _number(candidate_signal.get("score")) - _number(baseline_signal.get("score"))
    weight_delta = _number(candidate_signal.get("target_weight")) - _number(baseline_signal.get("target_weight"))
    data_changed = baseline_manifest.get("manifest_hash") != candidate_manifest.get("manifest_hash")
    model_changed = baseline_models != candidate_models
    replay_equal = baseline.get("replay_fingerprint") == candidate.get("replay_fingerprint")
    same_signal_key = (
        baseline_signal.get("symbol") == candidate_signal.get("symbol")
        and baseline_signal.get("as_of_date") == candidate_signal.get("as_of_date")
    )

    if comparison_type == "exact_replay":
        status = "pass" if baseline_validation["ok"] and candidate_validation["ok"] and replay_equal else "failed"
    elif comparison_type == "data_revision":
        status = "changed" if data_changed else "unchanged"
    else:
        status = "changed" if model_changed or action_changed or score_delta != 0.0 or weight_delta != 0.0 else "unchanged"

    return {
        "comparison_type": comparison_type,
        "status": status,
        "same_signal_key": same_signal_key,
        "baseline_snapshot_id": baseline.get("snapshot_id", ""),
        "candidate_snapshot_id": candidate.get("snapshot_id", ""),
        "baseline_valid": baseline_validation,
        "candidate_valid": candidate_validation,
        "replay_equal": replay_equal,
        "data_manifest_changed": data_changed,
        "model_versions_changed": model_changed,
        "action_changed": action_changed,
        "score_delta": score_delta,
        "target_weight_delta": weight_delta,
    }


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
