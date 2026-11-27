"""Regression coverage for legacy AI-Agent screen SQLite schemas."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from my_strategy.rl.experience_builder import build_rl_experiences


class RLExperienceSchemaTests(unittest.TestCase):
    def test_ml_only_legacy_detail_table_is_migrated_for_rl_experiences(self) -> None:
        # SQLite file handles can be released asynchronously on Windows after
        # pandas finishes a read.  Ignore only temporary-directory cleanup
        # contention; all schema and result assertions still run normally.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            path = Path(temp_dir) / "screen.db"
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """
                    create table stock_daily_results (
                        run_id text, stock text, date text, close real, equity real,
                        position_weight real, target_weight real, effective_target_weight real,
                        unrealized_return real, final_score real, final_action text,
                        risk_action text, ml_score_signal real, ml_confidence real
                    )
                    """
                )
                rows = [
                    ("ml-only", "000001.SZ", "2026-01-01", 10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.1, "hold", "pass", 0.2, 0.2),
                    ("ml-only", "000001.SZ", "2026-01-02", 10.2, 100.0, 0.0, 0.0, 0.0, 0.0, 0.2, "hold", "pass", 0.3, 0.3),
                    ("ml-only", "000001.SZ", "2026-01-05", 10.4, 100.0, 0.0, 0.0, 0.0, 0.0, 0.3, "hold", "pass", 0.4, 0.4),
                ]
                conn.executemany("insert into stock_daily_results values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

            result = build_rl_experiences(path, source_run_id="ml-only", action_weights=[0.0, 0.5])

            self.assertEqual(result["rows"], 4)
            with sqlite3.connect(path) as conn:
                columns = {row[1] for row in conn.execute("pragma table_info(stock_daily_results)")}
                self.assertIn("dl_score_signal", columns)
                self.assertIn("dl_confidence", columns)
                self.assertEqual(conn.execute("select count(*) from rl_experiences").fetchone()[0], 4)


if __name__ == "__main__":
    unittest.main()
