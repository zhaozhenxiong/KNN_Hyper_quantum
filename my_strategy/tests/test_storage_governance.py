from __future__ import annotations

from pathlib import Path
import gc
import sqlite3
import tempfile
import unittest

import pandas as pd

from my_strategy.storage.artifact_store import ArtifactStore, stable_artifact_id
from my_strategy.storage.processed_repository import ProcessedRepository
from my_strategy.storage.screen_sqlite import import_screen_outputs
from my_strategy.storage.warehouse_importer import _scan_output_artifacts
from my_strategy.storage.warehouse_integrity import audit_warehouse_integrity, ensure_run_detail_unique_indexes
from my_strategy.scripts.archive_run_details_to_parquet import archive_run


class StorageGovernanceTests(unittest.TestCase):
    def test_artifacts_are_external_and_legacy_blobs_can_migrate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ArtifactStore(root / "warehouse.db", content_root=root / "content")
            artifact_id = store.put_bytes("unit", "json", "external", b'{"ok":true}', {"source": "test"})
            with sqlite3.connect(root / "warehouse.db") as conn:
                row = conn.execute("SELECT payload, storage_backend, content_path FROM artifact_blobs WHERE artifact_id=?", (artifact_id,)).fetchone()
                self.assertEqual(row[0], b"")
                self.assertEqual(row[1], "content_addressed_file")
                self.assertTrue((root / "content" / row[2]).exists())
                legacy_id = stable_artifact_id("unit", "csv", "legacy")
                conn.execute(
                    "INSERT INTO artifact_blobs (artifact_id, namespace, kind, name, payload, size_bytes, sha256, metadata_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, '{}', 'now', 'now')",
                    (legacy_id, "unit", "csv", "legacy", b"a,b\n1,2\n", 8, "c7bd025bbad4df99b6b3f96d6ef2bb8597ee8ea6c4c0d0f93cb6781edda3c5f3"),
                )
            conn.close()
            # Correct hash for the inserted legacy payload.
            import hashlib
            with sqlite3.connect(root / "warehouse.db") as conn:
                conn.execute("UPDATE artifact_blobs SET sha256=? WHERE artifact_id=?", (hashlib.sha256(b"a,b\n1,2\n").hexdigest(), legacy_id))
            conn.close()
            result = store.externalize_legacy_records()
            self.assertEqual(result["migrated"], 1)
            self.assertEqual(store.get_bytes("unit", "csv", "legacy")[0], b"a,b\n1,2\n")
            gc.collect()

    def test_unique_detail_indexes_and_logical_fk_audit(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
            for table in ("backtest_daily_ledger", "agent_signal_records", "backtest_trades"):
                conn.execute(f"CREATE TABLE {table} (run_id TEXT, stock TEXT, date TEXT, source_file TEXT, row_number INTEGER)")
                conn.execute(f"INSERT INTO {table} VALUES ('run-1', '000001.SZ', '2026-01-01', 'source.csv', 0)")
            conn.execute("INSERT INTO runs VALUES ('run-1')")
            result = ensure_run_detail_unique_indexes(conn)
            self.assertTrue(all(item["status"] == "created_or_present" for item in result.values()))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO backtest_daily_ledger VALUES ('run-1', '000001.SZ', '2026-01-01', 'source.csv', 0)")
            self.assertEqual(audit_warehouse_integrity(conn)["status"], "ok")

    def test_inference_reads_do_not_expose_forward_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = ProcessedRepository(Path(temp) / "warehouse.db")
            frame = pd.DataFrame(
                [{"stock": "000001.SZ", "code": "000001.SZ", "date": "2026-01-02", "return_1d": 0.01,
                  "ma5_ratio": 1.0, "future_return_5d": 0.05, "future_up_5d": 1.0}]
            )
            repo.replace_model_feature_snapshots(
                frame, feature_version="unit", factor_version="factor-unit", source_start="2026-01-02", source_end="2026-01-02"
            )
            inference = repo.read_model_feature_snapshots(["000001.SZ"], "2026-01-02", "2026-01-02", "unit")
            training = repo.read_model_feature_snapshots(
                ["000001.SZ"], "2026-01-02", "2026-01-02", "unit", include_labels=True
            )
            self.assertNotIn("future_return_5d", inference.columns)
            self.assertEqual(float(training.loc[0, "future_return_5d"]), 0.05)
            gc.collect()

    def test_detail_archive_plans_run_and_year_partitions_without_deleting_source(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            for table in ("backtest_daily_ledger", "agent_signal_records", "backtest_trades"):
                conn.execute(f"CREATE TABLE {table} (run_id TEXT, date TEXT, stock TEXT)")
                conn.execute(f"INSERT INTO {table} VALUES ('run-1', '2025-12-31', '000001.SZ')")
            planned = archive_run(conn, Path("archive"), "run-1", execute=False)
        self.assertEqual(len(planned), 3)
        self.assertTrue(all(item["year"] == "2025" for item in planned))

    def test_result_images_stay_out_of_both_sqlite_catalogs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "figures").mkdir()
            (root / "reports").mkdir()
            (root / "figures" / "000001_SZ_multi_agent_snapshot.png").write_bytes(b"png")
            (root / "reports" / "summary.json").write_text("{}", encoding="utf-8")
            catalog = _scan_output_artifacts(root, "run-1", "now")
            self.assertNotIn("000001_SZ_multi_agent_snapshot.png", catalog["file_name"].tolist())
            result = import_screen_outputs(root, root / "screen.db", run_id="run-1", include_details=False)
            self.assertEqual(result["counts"]["chart_assets"], 0)
            with sqlite3.connect(root / "screen.db") as conn:
                self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chart_assets'").fetchone())
            conn.close()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
