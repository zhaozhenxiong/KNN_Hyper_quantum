"""Tests for actual multi-agent WFO command boundaries and ledger aggregation."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from my_strategy.scripts.run_real_multi_agent_feature_wfo import (
    aggregate_trade_ledger,
    build_backtest_command,
    build_training_command,
    coverage_eligible_stocks,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        stocks="000001.SZ,000002.SZ", max_stocks=0, seed=7, dl_model_type="lstm", dl_lookback=20,
        dl_epochs=2, dl_max_samples=1000, min_rows=120, neutral_group="board", workers="1", executor="serial", fusion_policy="", disable_external_sentiment=False,
    )


class RealMultiAgentFeatureWfoTests(unittest.TestCase):
    def test_training_command_keeps_model_fit_before_evaluation(self) -> None:
        command = build_training_command(
            kind="ml", profile="legacy", model_name="candidate", args=_args(), train_start="2024-01-01",
            train_end="2024-06-28", validation_end="2024-09-30", data_end="2024-09-30", child_run_id="run",
        )
        self.assertIn("--skip-registry", command)
        self.assertIn("--no-update", command)
        self.assertEqual(command[command.index("--train-end") + 1], "2024-06-28")
        self.assertEqual(command[command.index("--feature-profile") + 1], "legacy")

    def test_backtest_explicitly_disables_non_fold_safe_rl(self) -> None:
        command = build_backtest_command(
            args=_args(), window={"start": "2024-10-01", "end": "2024-12-31"}, ml_model="ml", dl_model="dl",
            output_root=Path("output"), child_run_id="run",
        )
        self.assertIn("--no-rl", command)
        self.assertIn("--save-daily", command)
        self.assertIn("--factor-cache", command)

    def test_aggregate_uses_actual_equity_and_checks_duplicate_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "daily").mkdir()
            (root / "trades").mkdir()
            for stock in ["000001", "000002"]:
                pd.DataFrame(
                    {"date": ["2024-01-02", "2024-01-03"], "equity": [100.0, 101.0], "position_weight": [0.0, 1.0]}
                ).to_csv(root / "daily" / f"{stock}_daily.csv", index=False)
                pd.DataFrame({"date": ["2024-01-03"], "action": ["BUY"]}).to_csv(root / "trades" / f"{stock}_trades.csv", index=False)
            metrics = aggregate_trade_ledger(root)
            self.assertEqual(metrics["daily_files"], 2)
            self.assertEqual(metrics["trade_count"], 2)
            self.assertEqual(metrics["duplicate_daily_dates"], 0)
            self.assertAlmostEqual(metrics["total_return"], 0.01, places=6)

    def test_coverage_filter_excludes_names_without_warmup_or_end_date(self) -> None:
        eligible, excluded = coverage_eligible_stocks(
            {
                "000001.SZ": {"first_date": "2024-01-01", "last_date": "2024-12-31"},
                "000002.SZ": {"first_date": "2024-02-01", "last_date": "2024-12-31"},
                "000003.SZ": {"first_date": "2024-01-01", "last_date": "2024-11-30"},
            },
            required_start="2024-01-01",
            required_end="2024-12-31",
        )
        self.assertEqual(eligible, ["000001.SZ"])
        self.assertIn("000002.SZ", excluded)
        self.assertIn("000003.SZ", excluded)


if __name__ == "__main__":
    unittest.main()
