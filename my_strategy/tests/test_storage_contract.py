from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from my_strategy.storage.storage_contract import load_storage_contract, validate_storage_contract
from my_strategy.scripts.prepare_parallel_storage_migration import build_plan, initialize_workspace, verify_workspace
from my_strategy.scripts.export_artifact_blob_pilot import export_partition, verify_export
from my_strategy.storage.storage_v2 import StorageV2Migrator, StorageV2ReadAdapter
from my_strategy.scripts.validate_raw_recovery_backup import compare_table, connect_read_only


class StorageContractTests(unittest.TestCase):
    def _write_contract(self, root: Path) -> Path:
        config = root / "storage_contract.yaml"
        config.write_text(
            """
version: 2
name: test-contract
runtime: {default_mode: db, parallel_migration_enabled: false}
locations:
  raw_market: {path: raw.db, role: canonical_raw_source, format: sqlite, retention: protected}
  processed_warehouse: {path: warehouse.db, role: canonical_processed_index, format: sqlite, retention: protected}
  screen_cache: {path: screen.db, role: serving_cache, format: sqlite, retention: bounded_cache}
  artifact_runs: {path: artifacts/runs, role: immutable_run_artifacts, format: files, retention: archive_by_run}
  cold_archive: {path: archive, role: archive, format: parquet, retention: append_only}
parallel_migration:
  stage_root: stage
  workspace_root: artifacts/runs
  targets:
    warehouse_stage: warehouse/khquant_v2.db
    screen_cache_stage: screen/ai_agent_screen_v2.db
    detail_archive_stage: archive_partitions
    artifact_stage: artifact_files
table_contracts: {}
cutover_guards: [partition_hashes_match]
""".strip(),
            encoding="utf-8",
        )
        return config

    def test_prepare_and_verify_empty_parallel_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("raw.db", "warehouse.db", "screen.db"):
                (root / name).write_bytes(name.encode("utf-8"))
            contract = load_storage_contract(self._write_contract(root), project_root=root)
            self.assertTrue(validate_storage_contract(contract, require_existing_sources=True)["valid"])
            plan = build_plan(contract, "unit-storage-v2")
            self.assertFalse(plan["data_copied"])
            result = initialize_workspace(contract, plan)
            manifest = Path(result["manifest"])
            self.assertTrue(manifest.exists())
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["migration_id"], "unit-storage-v2")
            verified = verify_workspace(contract, "unit-storage-v2")
            self.assertTrue(verified["valid"])
            self.assertTrue(all(verified["source_signatures_unchanged"].values()))

    def test_rejects_stage_target_outside_stage_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_contract(root)
            content = config.read_text(encoding="utf-8").replace("artifact_stage: artifact_files", "artifact_stage: ../../escape")
            config.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_storage_contract(config, project_root=root)

    def test_runtime_defaults_match_committed_contract(self) -> None:
        from my_strategy.storage.db_paths import DEFAULT_PROCESSED_DB, DEFAULT_RAW_DB

        contract = load_storage_contract()
        self.assertEqual(DEFAULT_RAW_DB, contract.raw_db)
        self.assertEqual(DEFAULT_PROCESSED_DB, contract.processed_db)

    def test_copy_only_artifact_export_keeps_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "warehouse.db"
            import hashlib
            import sqlite3

            payload = b"pilot-artifact"
            digest = hashlib.sha256(payload).hexdigest()
            conn = sqlite3.connect(source)
            try:
                conn.execute(
                    """CREATE TABLE artifact_blobs (
                    artifact_id TEXT PRIMARY KEY, namespace TEXT, kind TEXT, name TEXT,
                    payload BLOB, size_bytes INTEGER, sha256 TEXT, metadata_json TEXT,
                    created_at TEXT, updated_at TEXT)"""
                )
                conn.execute(
                    "INSERT INTO artifact_blobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("artifact-1", "pilot", "json", "item", payload, len(payload), digest, "{}", "now", "now"),
                )
                conn.commit()
            finally:
                conn.close()
            for name in ("raw.db", "screen.db"):
                (root / name).write_bytes(name.encode("utf-8"))
            contract = load_storage_contract(self._write_contract(root), project_root=root)
            plan = build_plan(contract, "artifact-pilot")
            initialize_workspace(contract, plan)
            before = source.read_bytes()
            result = export_partition(
                contract,
                migration_id="artifact-pilot",
                namespace="pilot",
                kind="json",
                max_records=None,
                max_bytes=None,
                reserve_bytes=0,
                execute=True,
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(source.read_bytes(), before)
            verified = verify_export(Path(result["manifest_path"]))
            self.assertTrue(verified["valid"])

    def test_storage_v2_copies_one_run_and_adapter_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "warehouse.db"
            import os
            import sqlite3

            conn = sqlite3.connect(source)
            try:
                conn.execute("CREATE TABLE backtest_reports (run_id TEXT, stock TEXT, report_json TEXT)")
                conn.execute("CREATE TABLE backtest_daily_ledger (run_id TEXT, stock TEXT, date TEXT, row_number INTEGER)")
                conn.execute("CREATE TABLE backtest_trades (run_id TEXT, stock TEXT, date TEXT, row_number INTEGER)")
                conn.execute("CREATE TABLE agent_signal_records (run_id TEXT, stock TEXT, date TEXT, row_number INTEGER)")
                run_id = "run-1"
                conn.execute(
                    "INSERT INTO backtest_reports VALUES (?, ?, ?)",
                    (run_id, "000001.SZ", '{"total_return": 0.1, "max_drawdown": -0.02, "sharpe_ratio": 1.2, "n_trades": 1}'),
                )
                for table in ("backtest_daily_ledger", "backtest_trades", "agent_signal_records"):
                    conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", (run_id, "000001.SZ", "2026-07-18", 0))
                conn.commit()
            finally:
                conn.close()
            for name in ("raw.db", "screen.db"):
                (root / name).write_bytes(name.encode("utf-8"))
            contract = load_storage_contract(self._write_contract(root), project_root=root)
            initialize_workspace(contract, build_plan(contract, "v2-run-pilot"))
            source_before = source.read_bytes()
            result = StorageV2Migrator(contract).copy_run("v2-run-pilot", "run-1")
            self.assertTrue(result["verification"]["valid"])
            self.assertEqual(source.read_bytes(), source_before)
            adapter = StorageV2ReadAdapter(contract)
            rows, source_name = adapter.read_run_rows("backtest_reports", "run-1")
            self.assertEqual(source_name, "legacy")
            self.assertEqual(len(rows), 1)
            previous = os.environ.get("KHQUANT_STORAGE_V2_READ_MODE")
            os.environ["KHQUANT_STORAGE_V2_READ_MODE"] = "warehouse_first"
            try:
                rows, source_name = adapter.read_run_rows("backtest_reports", "run-1")
                self.assertEqual(source_name, "storage_v2")
                self.assertEqual(len(rows), 1)
            finally:
                if previous is None:
                    os.environ.pop("KHQUANT_STORAGE_V2_READ_MODE", None)
                else:
                    os.environ["KHQUANT_STORAGE_V2_READ_MODE"] = previous
            self.assertEqual(adapter.shadow_write_run("v2-run-pilot", "run-1")["status"], "disabled")

    def test_raw_recovery_comparison_accepts_identical_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current_path, recovery_path = root / "current.db", root / "recovery.db"
            for path, rows in [(current_path, [("000001.SZ", "2026-01-01", 1.0), ("000001.SZ", "2026-01-02", 2.0)]), (recovery_path, [("000001.SZ", "2026-01-01", 1.0)])]:
                conn = sqlite3.connect(path)
                try:
                    conn.execute("CREATE TABLE stock_daily_normalized (stock TEXT, date TEXT, close REAL, imported_at TEXT, PRIMARY KEY(stock, date))")
                    conn.executemany("INSERT INTO stock_daily_normalized VALUES (?, ?, ?, 'now')", rows)
                    conn.commit()
                finally:
                    conn.close()
            current, recovery = connect_read_only(current_path), connect_read_only(recovery_path)
            try:
                result = compare_table(current, recovery, "stock_daily_normalized")
            finally:
                current.close()
                recovery.close()
            self.assertTrue(result["recovery_is_value_identical_subset"])
            self.assertEqual(result["current_only_keys"], 1)


if __name__ == "__main__":
    unittest.main()
