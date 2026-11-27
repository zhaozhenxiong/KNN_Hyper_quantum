"""Immutable storage for official point-in-time signal publications."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from my_strategy.core.run_context import stable_hash
from my_strategy.core.signal_snapshot import build_signal_snapshot, canonical_snapshot_json
from my_strategy.storage.db_paths import processed_db_path
from my_strategy.storage.warehouse_schema import init_warehouse
from my_strategy.validation.signal_stability import compare_signal_snapshots, validate_signal_snapshot


class ImmutableSnapshotConflictError(RuntimeError):
    """Raised when a published official signal would be retroactively changed."""


class SignalSnapshotStore:
    """Append-only publication store backed by the unified processed SQLite DB."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else processed_db_path()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        init_warehouse(conn)
        return conn

    def publish(self, snapshot: Mapping[str, Any], *, artifact_path: str | Path | None = None) -> dict[str, Any]:
        validation = validate_signal_snapshot(snapshot)
        if not validation["ok"]:
            raise ValueError("invalid signal snapshot: " + ", ".join(validation["issues"]))
        payload = dict(snapshot)
        snapshot_id = str(payload["snapshot_id"])
        fingerprint = str(payload["replay_fingerprint"])
        conn = self.connect()
        try:
            with conn:
                existing = conn.execute(
                    "SELECT snapshot_id, replay_fingerprint, artifact_path, payload_json FROM signal_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["replay_fingerprint"]) != fingerprint:
                        raise ImmutableSnapshotConflictError(
                            f"published signal conflicts with immutable snapshot {snapshot_id}; use a non-official channel for research"
                        )
                    return {
                        "snapshot_id": snapshot_id,
                        "status": "existing",
                        "artifact_path": str(existing["artifact_path"] or ""),
                    }

                channel = str(payload["channel"])
                stock = str(payload["signal"]["symbol"])
                as_of_date = str(payload["signal"]["as_of_date"])
                channel_conflict = conn.execute(
                    "SELECT snapshot_id FROM signal_snapshots WHERE channel = ? AND stock = ? AND as_of_date = ?",
                    (channel, stock, as_of_date),
                ).fetchone()
                if channel_conflict is not None:
                    raise ImmutableSnapshotConflictError(
                        f"published signal already exists for {channel}:{stock}:{as_of_date} as {channel_conflict['snapshot_id']}"
                    )

                saved_path = self._write_artifact(payload, artifact_path)
                lineage = payload["lineage"]
                execution = payload.get("execution", {})
                conn.execute(
                    """
                    INSERT INTO signal_snapshots (
                        snapshot_id, channel, run_id, stock, as_of_date, published_at, signal_time, execution_time,
                        execution_assumption, final_action, final_score, target_weight, model_version, data_version,
                        config_hash, git_commit, seed, data_manifest_hash, replay_fingerprint, artifact_path, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        channel,
                        lineage.get("run_id", ""),
                        stock,
                        as_of_date,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                        execution.get("signal_time", ""),
                        execution.get("execution_time", ""),
                        execution.get("execution_assumption", ""),
                        payload["signal"].get("signal", ""),
                        payload["signal"].get("score"),
                        payload["signal"].get("target_weight"),
                        json.dumps(payload.get("model_versions", {}), ensure_ascii=False, sort_keys=True),
                        payload["signal"].get("data_version", ""),
                        lineage.get("config_hash", ""),
                        lineage.get("git_commit", ""),
                        lineage.get("seed"),
                        payload["data_manifest"].get("manifest_hash", ""),
                        fingerprint,
                        str(saved_path),
                        canonical_snapshot_json(payload),
                    ),
                )
        finally:
            conn.close()
        return {"snapshot_id": snapshot_id, "status": "published", "artifact_path": str(saved_path)}

    def load(self, snapshot_id: str) -> dict[str, Any]:
        conn = self.connect()
        try:
            row = conn.execute("SELECT payload_json FROM signal_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"signal snapshot not found: {snapshot_id}")
        return json.loads(str(row["payload_json"]))

    def compare(self, baseline_id: str, candidate_id: str, *, comparison_type: str) -> dict[str, Any]:
        report = compare_signal_snapshots(
            self.load(baseline_id), self.load(candidate_id), comparison_type=comparison_type
        )
        report["comparison_id"] = stable_hash(
            {"baseline": baseline_id, "candidate": candidate_id, "comparison_type": comparison_type}
        )[:32]
        conn = self.connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO signal_snapshot_comparisons (
                        comparison_id, created_at, comparison_type, baseline_snapshot_id, candidate_snapshot_id,
                        status, summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report["comparison_id"],
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                        comparison_type,
                        baseline_id,
                        candidate_id,
                        report["status"],
                        json.dumps(report, ensure_ascii=False, sort_keys=True),
                    ),
                )
        finally:
            conn.close()
        return report

    def summarize_channels(self, *, baseline_channel: str, candidate_channel: str, as_of_date: str) -> dict[str, Any]:
        """Summarize same-date action stability for promotion gates without mutation."""

        conn = self.connect()
        try:
            baseline_rows = conn.execute(
                "SELECT stock, payload_json FROM signal_snapshots WHERE channel = ? AND as_of_date = ?",
                (baseline_channel, as_of_date),
            ).fetchall()
            candidate_rows = conn.execute(
                "SELECT stock, payload_json FROM signal_snapshots WHERE channel = ? AND as_of_date = ?",
                (candidate_channel, as_of_date),
            ).fetchall()
        finally:
            conn.close()
        baseline = {str(row["stock"]): json.loads(str(row["payload_json"])) for row in baseline_rows}
        candidate = {str(row["stock"]): json.loads(str(row["payload_json"])) for row in candidate_rows}
        matched = sorted(set(baseline).intersection(candidate))
        action_flips = 0
        score_deltas: list[float] = []
        for stock in matched:
            report = compare_signal_snapshots(baseline[stock], candidate[stock], comparison_type="model_drift")
            action_flips += int(bool(report["action_changed"]))
            score_deltas.append(abs(float(report["score_delta"])))
        return {
            "as_of_date": as_of_date,
            "baseline_channel": baseline_channel,
            "candidate_channel": candidate_channel,
            "baseline_count": len(baseline),
            "candidate_count": len(candidate),
            "matched": len(matched),
            "baseline_only": sorted(set(baseline).difference(candidate)),
            "candidate_only": sorted(set(candidate).difference(baseline)),
            "action_flips": action_flips,
            "action_flip_rate": action_flips / len(matched) if matched else 1.0,
            "mean_abs_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
        }

    @staticmethod
    def _write_artifact(snapshot: Mapping[str, Any], artifact_path: str | Path | None) -> Path:
        if artifact_path is None:
            raise ValueError("artifact_path is required for an immutable signal snapshot")
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_snapshot_json(snapshot)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("replay_fingerprint") != snapshot.get("replay_fingerprint"):
                raise ImmutableSnapshotConflictError(f"artifact conflicts with immutable snapshot: {path}")
            return path
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
        return path


def publish_latest_signal(
    *,
    stock: str,
    decision: Mapping[str, Any],
    run_context: Mapping[str, Any],
    config: Mapping[str, Any],
    processed_db: str | Path | None,
    source_paths: list[str | Path] | None = None,
    source_context: Mapping[str, Any] | None = None,
    channel: str = "official",
) -> dict[str, Any]:
    """Build and publish an official latest signal from a runner output row."""

    agents = config.get("agents", {}) if isinstance(config.get("agents"), Mapping) else {}
    model_versions = {
        "ml": (agents.get("ml", {}) or {}).get("model_name", ""),
        "deep_learning": (agents.get("deep_learning", {}) or {}).get("model_name", ""),
        "probabilistic": (agents.get("probabilistic", {}) or {}).get("model_name", ""),
        "reinforcement_learning": (agents.get("reinforcement_learning", {}) or {}).get("model_name", ""),
        "fusion": stable_hash((config.get("decision", {}) or {}).get("fusion", {})),
    }
    snapshot = build_signal_snapshot(
        stock=stock,
        decision=decision,
        run_context=run_context,
        model_versions=model_versions,
        source_paths=source_paths,
        source_context=source_context,
        channel=channel,
    )
    artifact_root_value = str(run_context.get("artifact_root") or "").strip()
    if not artifact_root_value:
        raise ValueError("run_context.artifact_root is required for signal publication")
    artifact_root = Path(artifact_root_value)
    tag = stock.replace(".", "_")
    as_of_date = str(snapshot["signal"]["as_of_date"]).replace("-", "")
    artifact_path = artifact_root / "signals" / "published" / f"{channel}_{tag}_{as_of_date}.json"
    return SignalSnapshotStore(processed_db).publish(snapshot, artifact_path=artifact_path)
