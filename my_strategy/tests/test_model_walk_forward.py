"""Tests for leakage-safe model walk-forward orchestration."""

from __future__ import annotations

import argparse
import unittest

import pandas as pd

from my_strategy.scripts.run_model_walk_forward import build_training_command, build_walk_forward_folds


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        stocks="000001.SZ,000002.SZ",
        seed=7,
        min_rows=120,
        max_stocks=0,
        allow_update=False,
        no_factor_cache=False,
        model_type="lightgbm",
        base_label="future_return_5d",
        neutral_group="auto",
        lookback=30,
        epochs=2,
        batch_size=64,
        hidden_size=16,
        max_samples=1000,
    )


class ModelWalkForwardTests(unittest.TestCase):
    def test_folds_keep_validation_test_and_holdout_separate(self) -> None:
        sessions = list(pd.bdate_range("2024-01-01", periods=90))
        folds, holdout = build_walk_forward_folds(
            sessions,
            train_days=30,
            valid_days=10,
            test_days=10,
            step_days=10,
            holdout_days=10,
        )
        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold["train_end"], fold["valid_start"])
            self.assertLess(fold["valid_end"], fold["test_start"])
            self.assertLess(fold["test_end"], holdout["start"])

    def test_fold_command_is_local_and_skips_registry(self) -> None:
        window = {
            "train_start": "2024-01-01",
            "train_end": "2024-02-15",
            "valid_start": "2024-02-16",
            "valid_end": "2024-02-29",
            "test_start": "2024-03-01",
            "test_end": "2024-03-15",
        }
        command = build_training_command(
            args=_args(),
            kind="ml",
            model_name="candidate_wfo_01",
            run_id="wfo-fold-01",
            window=window,
            skip_registry=True,
        )
        self.assertIn("--no-update", command)
        self.assertIn("--skip-registry", command)
        self.assertEqual(command[command.index("--model-name") + 1], "candidate_wfo_01")
        self.assertEqual(command[command.index("--valid-end") + 1], "2024-02-29")

    def test_final_candidate_command_keeps_registry_enabled(self) -> None:
        window = {
            "train_start": "2024-01-01",
            "train_end": "2024-02-15",
            "valid_start": "2024-02-16",
            "valid_end": "2024-02-29",
            "test_start": "2024-03-01",
            "test_end": "2024-03-29",
        }
        command = build_training_command(
            args=_args(),
            kind="ml",
            model_name="candidate_final",
            run_id="wfo-candidate",
            window=window,
            skip_registry=False,
        )
        self.assertNotIn("--skip-registry", command)
        self.assertEqual(command[command.index("--model-name") + 1], "candidate_final")


if __name__ == "__main__":
    unittest.main()
