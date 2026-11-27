"""Regression tests for leakage-safe fusion walk-forward calibration."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from my_strategy.scripts.optimize_fusion_from_existing_results import (
    DIRECT_WEIGHT_KEYS,
    generate_walk_forward_policies,
    promotion_eligible,
    promotion_quality_checks,
    slice_period_data,
    simulate_policy,
    select_daily_paths,
    walk_forward_folds,
)


class FusionWalkForwardTests(unittest.TestCase):
    def test_folds_keep_holdout_untouched(self) -> None:
        sessions = list(pd.bdate_range("2026-01-05", periods=100))
        folds, holdout = walk_forward_folds(sessions, train_days=40, test_days=10, step_days=10, holdout_days=10)

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold["train_end"], fold["test_start"])
            self.assertLess(fold["test_end"], holdout["start"])

    def test_candidates_are_direct_and_normalized(self) -> None:
        policies = generate_walk_forward_policies({"weights": {"ml": 1.0}}, trials=4, seed=7)

        self.assertEqual(len(policies), 4)
        for policy in policies:
            self.assertEqual(set(policy["weights"]), set(DIRECT_WEIGHT_KEYS))
            self.assertAlmostEqual(sum(policy["weights"].values()), 1.0, places=5)

    def test_promotion_needs_folds_and_both_deltas(self) -> None:
        self.assertTrue(
            promotion_eligible(
                fold_count=2,
                min_folds=2,
                mean_oos_delta=0.01,
                holdout_delta=0.01,
                positive_fold_rate=1.0,
                min_oos_delta=0.0,
                min_holdout_delta=0.0,
                min_positive_fold_rate=0.5,
            )
        )

    def test_quality_gate_rejects_cost_sensitive_candidate(self) -> None:
        args = argparse.Namespace(
            promotion_min_return_delta=0.0,
            promotion_min_drawdown_delta=0.0,
            promotion_min_sharpe_delta=0.0,
            promotion_max_trade_count_delta=0.0,
            promotion_min_cost_stress_return_delta=0.0,
        )
        baseline = {
            "avg_total_return": 0.02,
            "avg_drawdown": -0.05,
            "avg_sharpe": 0.80,
            "avg_trade_count": 10.0,
        }
        candidate = {
            "avg_total_return": 0.03,
            "avg_drawdown": -0.04,
            "avg_sharpe": 0.90,
            "avg_trade_count": 10.0,
        }
        stressed_candidate = {"avg_total_return": -0.02}
        stressed_baseline = {"avg_total_return": -0.01}

        checks = promotion_quality_checks(candidate, baseline, stressed_candidate, stressed_baseline, args)

        self.assertFalse(checks["cost_stress_return"])
        self.assertFalse(
            promotion_eligible(
                fold_count=2,
                min_folds=2,
                mean_oos_delta=0.01,
                holdout_delta=0.01,
                positive_fold_rate=1.0,
                min_oos_delta=0.0,
                min_holdout_delta=0.0,
                min_positive_fold_rate=0.5,
                quality_checks=checks,
            )
        )

    def test_policy_simulation_uses_prior_position_for_next_return(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05", periods=3, freq="B"),
                "close": [10.0, 11.0, 12.1],
                "return_1d": [0.0, 0.1, 0.1],
                "ml_score_signal": [0.5, 0.5, -0.5],
                "dl_score_signal": [0.0, 0.0, 0.0],
                "prob_score_signal": [0.0, 0.0, 0.0],
                "sentiment_score_signal": [0.0, 0.0, 0.0],
                "trend_score": [0.0, 0.0, 0.0],
                "reversal_score": [0.0, 0.0, 0.0],
                "volume_score": [0.0, 0.0, 0.0],
            }
        )
        policy = {
            "weights": {"ml": 1.0},
            "activation": "linear",
            "temperature": 1.0,
            "buy_probability": 0.60,
            "hold_probability": 0.54,
            "sell_probability": 0.60,
            "high_probability": 0.72,
            "max_probability": 0.86,
            "min_ml_score": -1.0,
            "min_prob_score": -1.0,
            "use_confidence": False,
        }
        args = argparse.Namespace(initial_cash=100.0, rebalance_days=1, min_weight_change=0.08, commission=0.001, slippage=0.0)

        result = simulate_policy(frame, policy, args)

        self.assertEqual(result["trade_flag"].sum(), 2)
        self.assertEqual(result["fusion_action"].tolist(), ["buy", "buy", "sell"])
        self.assertAlmostEqual(float(result.iloc[-1]["equity"]), 107.0162, places=4)

    def test_slice_preserves_short_out_of_sample_window(self) -> None:
        frame = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=30), "close": range(30)})
        sliced = slice_period_data({"000001.SZ": frame}, "2026-02-02", "2026-02-13")

        self.assertIn("000001.SZ", sliced)
        self.assertEqual(len(sliced["000001.SZ"]), 10)

    def test_industry_stratified_sampling_records_group_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daily = root / "daily"
            daily.mkdir()
            for tag in ["000001_SZ", "000002_SZ", "600000_SH", "300001_SZ"]:
                (daily / f"{tag}_daily.csv").touch()
            metadata = root / "metadata.csv"
            metadata.write_text(
                "stock,industry\n000001.SZ,finance\n000002.SZ,finance\n600000.SH,utilities\n",
                encoding="utf-8",
            )

            selected, summary = select_daily_paths(root, max_stocks=3, seed=42, stratify="industry", metadata_path=metadata)

            self.assertEqual(len(selected), 3)
            self.assertEqual(summary["method"], "industry")
            self.assertEqual(summary["group_counts"]["finance"]["selected"], 1)
            self.assertEqual(summary["group_counts"]["utilities"]["selected"], 1)
            self.assertEqual(summary["group_counts"]["unclassified"]["selected"], 1)
        self.assertFalse(
            promotion_eligible(
                fold_count=1,
                min_folds=2,
                mean_oos_delta=0.10,
                holdout_delta=0.10,
                positive_fold_rate=1.0,
                min_oos_delta=0.0,
                min_holdout_delta=0.0,
                min_positive_fold_rate=0.5,
            )
        )
        self.assertFalse(
            promotion_eligible(
                fold_count=2,
                min_folds=2,
                mean_oos_delta=0.10,
                holdout_delta=-0.01,
                positive_fold_rate=1.0,
                min_oos_delta=0.0,
                min_holdout_delta=0.0,
                min_positive_fold_rate=0.5,
            )
        )
        self.assertFalse(
            promotion_eligible(
                fold_count=2,
                min_folds=2,
                mean_oos_delta=0.10,
                holdout_delta=0.10,
                positive_fold_rate=0.5,
                min_oos_delta=0.0,
                min_holdout_delta=0.0,
                min_positive_fold_rate=0.55,
            )
        )


if __name__ == "__main__":
    unittest.main()
