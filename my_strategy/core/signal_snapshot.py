"""Point-in-time payload builders for immutable published trading signals."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from my_strategy.core.run_context import stable_hash


SNAPSHOT_SCHEMA_VERSION = 1
SMALL_SOURCE_HASH_BYTES = 1_000_000


def normalise_snapshot_value(value: Any) -> Any:
    """Convert runtime values into deterministic JSON-compatible signal metadata."""

    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): normalise_snapshot_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [normalise_snapshot_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return normalise_snapshot_value(item_method())
        except (TypeError, ValueError):
            pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except (TypeError, ValueError):
            pass
    return str(value)


def source_descriptor(path: str | Path, *, max_hash_bytes: int = SMALL_SOURCE_HASH_BYTES) -> dict[str, Any]:
    """Describe an input without expensive hashing of large market databases."""

    source = Path(path)
    descriptor: dict[str, Any] = {"path": str(source), "exists": source.exists()}
    if not source.exists() or not source.is_file():
        return descriptor
    stat = source.stat()
    descriptor.update({"size_bytes": int(stat.st_size), "modified_at_ns": int(stat.st_mtime_ns)})
    if stat.st_size <= max_hash_bytes:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        descriptor["sha256"] = digest.hexdigest()
    else:
        descriptor["sha256"] = ""
        descriptor["hash_skipped"] = "source_exceeds_limit"
    return descriptor


def build_data_manifest(
    *,
    data_version: str,
    source_paths: list[str | Path] | None = None,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, lightweight manifest for data available at signal time."""

    manifest = {
        "data_version": str(data_version or "unknown"),
        "sources": [source_descriptor(path) for path in sorted(source_paths or [], key=lambda item: str(item))],
        "context": normalise_snapshot_value(dict(source_context or {})),
    }
    # A content-hashed source may receive a new filesystem mtime during an
    # identical replay.  Preserve that mtime for audit display but exclude it
    # from the identity.  For large sources where hashing is intentionally
    # skipped, size plus mtime remains the revision detector.
    source_identity = []
    for source in manifest["sources"]:
        item = dict(source)
        if item.get("sha256"):
            item.pop("modified_at_ns", None)
        source_identity.append(item)
    manifest["manifest_hash"] = stable_hash(
        {"data_version": manifest["data_version"], "sources": source_identity, "context": manifest["context"]}
    )
    return manifest


def build_signal_snapshot(
    *,
    stock: str,
    decision: Mapping[str, Any],
    run_context: Mapping[str, Any],
    model_versions: Mapping[str, Any],
    source_paths: list[str | Path] | None = None,
    source_context: Mapping[str, Any] | None = None,
    channel: str = "official",
) -> dict[str, Any]:
    """Create one point-in-time signal payload without persisting it.

    The replay fingerprint deliberately excludes ``run_id`` and publication time:
    a rerun of the same information set must compare equal even when it receives
    a new run identifier.
    """

    clean_decision = normalise_snapshot_value(dict(decision))
    as_of_date = str(clean_decision.get("date") or run_context.get("as_of_date") or "")[:10]
    if not stock or not as_of_date:
        raise ValueError("stock and as_of_date are required for a signal snapshot")

    data_manifest = build_data_manifest(
        data_version=str(run_context.get("data_version") or "unknown"),
        source_paths=source_paths,
        source_context=source_context,
    )
    model_data = normalise_snapshot_value(dict(model_versions))
    signal = {
        "symbol": str(stock).upper(),
        "as_of_date": as_of_date,
        "score": clean_decision.get("final_score"),
        "confidence": clean_decision.get("entry_probability"),
        "signal": str(clean_decision.get("final_action") or "abstain").lower(),
        "target_weight": clean_decision.get("target_weight"),
        "reason_codes": clean_decision.get("reason_codes") or clean_decision.get("reason") or "",
        "model_version": model_data,
        "data_version": str(run_context.get("data_version") or "unknown"),
    }
    lineage = {
        "run_id": str(run_context.get("run_id") or ""),
        "as_of_date": as_of_date,
        "git_commit": str(run_context.get("git_commit") or ""),
        "config_hash": str(run_context.get("config_hash") or ""),
        "seed": normalise_snapshot_value(run_context.get("seed")),
    }
    snapshot_id = f"{channel}:{as_of_date}:{str(stock).upper()}"
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "channel": str(channel),
        "lineage": lineage,
        "model_versions": model_data,
        "data_manifest": data_manifest,
        "signal": signal,
        "decision": clean_decision,
        "execution": {
            "signal_time": f"{as_of_date}T15:00:00+08:00",
            "execution_time": "",
            "execution_assumption": "next_trading_open",
        },
    }
    replay_payload = dict(payload)
    replay_payload.pop("snapshot_id", None)
    replay_payload.pop("channel", None)
    replay_payload["lineage"] = {key: value for key, value in lineage.items() if key != "run_id"}
    replay_payload["data_manifest"] = {"manifest_hash": data_manifest["manifest_hash"]}
    payload["replay_fingerprint"] = stable_hash(replay_payload)
    return payload


def canonical_snapshot_json(payload: Mapping[str, Any]) -> str:
    """Render a stable JSON form used by the immutable file and SQLite record."""

    return json.dumps(normalise_snapshot_value(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True)
