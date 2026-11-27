"""Regression tests for immutable point-in-time signal publications."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from my_strategy.core.signal_snapshot import build_signal_snapshot
from my_strategy.storage.signal_snapshot_store import ImmutableSnapshotConflictError, SignalSnapshotStore, publish_latest_signal


def _context(root: Path, *, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "as_of_date": "2026-07-10",
        "git_commit": "abc123",
        "config_hash": "config-hash",
        "data_version": "2024-01-01:2026-07-10:local",
        "seed": 42,
        "artifact_root": str(root / run_id),
    }


def _decision(action: str = "buy", score: float = 0.71) -> dict[str, object]:
    return {
        "date": "2026-07-10",
        "final_action": action,
        "final_score": score,
        "entry_probability": 0.72,
        "target_weight": 0.5,
        "reason": "unit-test",
    }


class SignalSnapshotTests(unittest.TestCase):
    def _snapshot(self, root: Path, *, channel: str, run_id: str, decision: dict[str, object] | None = None, model: str = "ml-a", source: str = "v1") -> dict[str, object]:
        return build_signal_snapshot(
            stock="000001.SZ",
            decision=decision or _decision(),
            run_context=_context(root, run_id=run_id),
            model_versions={"ml": model, "fusion": "f1"},
            source_context={"raw_revision": source},
            channel=channel,
        )

    def test_official_snapshot_is_append_only_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SignalSnapshotStore(root / "processed.sqlite")
            snapshot = self._snapshot(root, channel="official", run_id="first")
            artifact = root / "artifacts" / "official.json"

            first = store.publish(snapshot, artifact_path=artifact)
            replay = self._snapshot(root, channel="official", run_id="second")
            second = store.publish(replay, artifact_path=root / "other" / "ignored.json")

            self.assertEqual(first["status"], "published")
            self.assertEqual(second["status"], "existing")
            self.assertTrue(artifact.exists())
            self.assertEqual(store.load(first["snapshot_id"])["replay_fingerprint"], snapshot["replay_fingerprint"])

    def test_content_hashed_source_ignores_mtime_in_replay_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "small_signal.csv"
            source.write_text("date,action\n2026-07-10,buy\n", encoding="utf-8")
            first = build_signal_snapshot(
                stock="000001.SZ",
                decision=_decision(),
                run_context=_context(root, run_id="first"),
                model_versions={"ml": "ml-a"},
                source_paths=[source],
            )
            source.write_text("date,action\n2026-07-10,buy\n", encoding="utf-8")
            replay = build_signal_snapshot(
                stock="000001.SZ",
                decision=_decision(),
                run_context=_context(root, run_id="replay"),
                model_versions={"ml": "ml-a"},
                source_paths=[source],
            )
            self.assertEqual(first["replay_fingerprint"], replay["replay_fingerprint"])

    def test_conflicting_official_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SignalSnapshotStore(root / "processed.sqlite")
            store.publish(self._snapshot(root, channel="official", run_id="first"), artifact_path=root / "first.json")
            changed = self._snapshot(root, channel="official", run_id="second", decision=_decision(action="sell", score=-0.4))
            with self.assertRaises(ImmutableSnapshotConflictError):
                store.publish(changed, artifact_path=root / "second.json")

    def test_exact_replay_data_revision_and_model_drift_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SignalSnapshotStore(root / "processed.sqlite")
            baseline = self._snapshot(root, channel="official", run_id="base")
            exact = self._snapshot(root, channel="research", run_id="replay")
            revised = self._snapshot(root, channel="revised", run_id="revision", source="v2")
            candidate = self._snapshot(root, channel="candidate", run_id="candidate", model="ml-b", decision=_decision(score=0.83))
            entries = {}
            for name, payload in {"baseline": baseline, "exact": exact, "revised": revised, "candidate": candidate}.items():
                entries[name] = store.publish(payload, artifact_path=root / f"{name}.json")["snapshot_id"]

            self.assertEqual(store.compare(entries["baseline"], entries["exact"], comparison_type="exact_replay")["status"], "pass")
            revision_report = store.compare(entries["baseline"], entries["revised"], comparison_type="data_revision")
            self.assertEqual(revision_report["status"], "changed")
            self.assertTrue(revision_report["data_manifest_changed"])
            drift_report = store.compare(entries["baseline"], entries["candidate"], comparison_type="model_drift")
            self.assertEqual(drift_report["status"], "changed")
            self.assertTrue(drift_report["model_versions_changed"])

    def test_channel_summary_counts_same_date_action_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SignalSnapshotStore(root / "processed.sqlite")
            store.publish(self._snapshot(root, channel="official", run_id="official"), artifact_path=root / "official.json")
            store.publish(
                self._snapshot(root, channel="candidate", run_id="candidate", decision=_decision(action="sell")),
                artifact_path=root / "candidate.json",
            )
            summary = store.summarize_channels(
                baseline_channel="official",
                candidate_channel="candidate",
                as_of_date="2026-07-10",
            )
            self.assertEqual(summary["matched"], 1)
            self.assertEqual(summary["action_flips"], 1)
            self.assertEqual(summary["action_flip_rate"], 1.0)

    def test_runner_helper_writes_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = publish_latest_signal(
                stock="000001.SZ",
                decision=_decision(),
                run_context=_context(root, run_id="publication-run"),
                config={"agents": {"ml": {"model_name": "ml-a"}}, "decision": {"fusion": {"weights": {"ml": 1.0}}}},
                processed_db=root / "processed.sqlite",
                source_context={"raw_revision": "v1"},
            )
            self.assertEqual(result["status"], "published")
            self.assertTrue(Path(result["artifact_path"]).exists())


if __name__ == "__main__":
    unittest.main()
