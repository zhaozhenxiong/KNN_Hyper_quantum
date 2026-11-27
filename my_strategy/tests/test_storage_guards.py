from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from my_strategy.storage.persistence_guard import assert_unique_keys, require_run_id, upsert_output_reference
from my_strategy.scripts.monitor_storage_growth import inspect_database


class StorageGuardTests(unittest.TestCase):
    def test_run_id_and_daily_key_guards(self) -> None:
        with self.assertRaises(ValueError):
            require_run_id("", operation="test")
        frame = pd.DataFrame({"run_id": ["r", "r"], "stock": ["000001.SZ", "000001.SZ"], "date": ["2026-01-02", "2026-01-02"]})
        with self.assertRaises(ValueError):
            assert_unique_keys(frame, ["run_id", "stock", "date"], table="daily")

    def test_reference_is_one_authoritative_row_per_scope(self) -> None:
        with sqlite3.connect(":memory:") as conn:
            upsert_output_reference(conn, run_id="run-1", scope_key="000001.SZ", output_type="backtest_report", artifact_id="first", authority="warehouse")
            upsert_output_reference(conn, run_id="run-1", scope_key="000001.SZ", output_type="backtest_report", artifact_id="second", authority="warehouse")
            row = conn.execute("SELECT artifact_id, COUNT(*) OVER () AS n FROM run_output_references").fetchone()
            self.assertEqual(row[0], "second")
            self.assertEqual(row[1], 1)

    def test_monitor_reads_tables_and_duplicate_keys_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "sample.db"
            conn = sqlite3.connect(database)
            try:
                conn.execute("CREATE TABLE backtest_daily_ledger (run_id TEXT, stock TEXT, date TEXT)")
                conn.executemany("INSERT INTO backtest_daily_ledger VALUES (?, ?, ?)", [("r1", "000001.SZ", "2026-01-01"), ("r1", "000001.SZ", "2026-01-01")])
                conn.commit()
            finally:
                conn.close()
            before = database.read_bytes()
            result = inspect_database("warehouse", database, collect_dbstat=True, retained_runs=5)
            self.assertEqual(database.read_bytes(), before)
            duplicate = result["duplicate_keys"][0]
            self.assertEqual(duplicate["duplicate_rows"], 1)
            self.assertEqual(result["tables"][0]["rows"], 2)
