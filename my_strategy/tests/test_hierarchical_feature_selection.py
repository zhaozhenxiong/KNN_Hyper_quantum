"""Regression tests for layered X selection and nested walk-forward boundaries."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from my_strategy.ml.hierarchical_feature_selection import (
    build_nested_folds,
    candidate_layer_combinations,
    feature_columns_for_combo,
    select_layer_features,
)


class HierarchicalFeatureSelectionTests(unittest.TestCase):
    def test_nested_folds_keep_final_holdout_untouched(self) -> None:
        sessions = pd.bdate_range("2024-01-01", periods=110)
        folds, holdout = build_nested_folds(sessions, train_days=40, valid_days=10, test_days=10, step_days=10, holdout_days=20)

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold["train_end"], fold["valid_start"])
            self.assertLess(fold["valid_end"], fold["test_start"])
            self.assertLess(fold["test_end"], holdout["start"])

    def test_layer_selection_removes_duplicate_and_generates_combinations(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=40)
        rows = []
        for date_index, date in enumerate(dates):
            for stock_index in range(30):
                signal = stock_index + (date_index % 3) * 0.1
                rows.append({"date": date, "stock": f"{stock_index:06d}.SZ", "future_return_5d": signal, "a": signal, "a_copy": signal, "b": -signal})
        frame = pd.DataFrame(rows)
        selected, _ = select_layer_features(
            frame,
            {"one": ["a", "a_copy"], "two": ["b"]},
            label_column="future_return_5d",
            top_k=2,
            correlation_threshold=0.90,
        )

        self.assertEqual(len(selected["one"]), 1)
        self.assertEqual(selected["two"], ["b"])
        combos = candidate_layer_combinations(selected)
        self.assertIn(("one", "two"), combos)
        self.assertEqual(feature_columns_for_combo(selected, ("one", "two")), [selected["one"][0], "b"])


if __name__ == "__main__":
    unittest.main()
