"""Regression tests for candidate-model promotion gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_strategy.model_registry import ModelRegistry
from my_strategy.model_registry.schema import ModelEntry
from my_strategy.model_selector import refresh_best_models
from my_strategy.validation.model_promotion import assess_model_promotion


def _entry(*, name: str, stage: str, score: float, ic: float, stocks: int = 600, days: int = 80) -> ModelEntry:
    return ModelEntry(
        model_id=f"id-{name}",
        name=name,
        kind="ml",
        stage=stage,  # type: ignore[arg-type]
        run_id=f"run-{name}",
        model_path=f"artifact://ml/model/{name}",
        metrics={
            "score": score,
            "test_stocks": stocks,
            "test_metrics": {
                "daily_rank_ic_mean": ic,
                "daily_rank_ic_days": days,
                "top_bottom_spread": 0.03,
                "stock_rank_ic_ir": 0.5,
            },
        },
    )


class ModelPromotionTests(unittest.TestCase):
    def test_gate_requires_quality_coverage_and_stable_signals(self) -> None:
        result = assess_model_promotion(
            candidate=_entry(name="candidate", stage="candidate", score=0.12, ic=0.08),
            champion=_entry(name="champion", stage="champion", score=0.10, ic=0.06),
            signal_stability={"matched": 40, "action_flip_rate": 0.20},
            walk_forward={"fold_count": 3, "positive_fold_rate": 2 / 3, "mean_oos_delta": 0.01, "holdout_delta": 0.01},
        )
        self.assertTrue(result["eligible"], result)
        self.assertTrue(result["checks"]["test_stock_coverage"])

    def test_gate_rejects_signal_instability_and_short_test_window(self) -> None:
        result = assess_model_promotion(
            candidate=_entry(name="candidate", stage="candidate", score=0.12, ic=0.08, days=20),
            champion=_entry(name="champion", stage="champion", score=0.10, ic=0.06),
            signal_stability={"matched": 40, "action_flip_rate": 0.50},
            walk_forward={"fold_count": 1, "positive_fold_rate": 0.0, "mean_oos_delta": -0.01, "holdout_delta": -0.01},
        )
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["test_day_coverage"])
        self.assertFalse(result["checks"]["signal_stability"])
        self.assertFalse(result["checks"]["walk_forward_folds"])

    def test_refresh_keeps_existing_champion_and_does_not_promote_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ModelRegistry(root / "registry.sqlite", snapshot_path=root / "registry.json")
            champion = service.register(
                name="incumbent",
                kind="ml",
                stage="candidate",
                run_id="incumbent-run",
                model_path="incumbent-model",
                metrics={"score": 0.10},
            )
            service.promote(champion.model_id, "champion")
            candidate_row = {
                "kind": "ml",
                "model_name": "challenger",
                "score": 0.99,
                "report_path": "candidate-report",
                "model_path": "candidate-model",
                "run_id": "candidate-run",
                "test_daily_rank_ic": 0.3,
                "test_stocks": 600,
            }
            candidate = service.register(
                name="challenger",
                kind="ml",
                stage="candidate",
                run_id="candidate-run",
                model_path="candidate-model",
                metrics={"score": 0.99},
            )
            with (
                patch("my_strategy.model_selector.default_registry", return_value=service),
                patch("my_strategy.model_selector.select_best_model", side_effect=[candidate_row, None]),
                patch("my_strategy.model_selector.write_db", return_value=False),
                patch("my_strategy.model_selector.write_files", return_value=False),
            ):
                result = refresh_best_models(bootstrap_if_missing=False)

            self.assertEqual(result["ml"]["model_name"], "incumbent")
            self.assertEqual(service.champion("ml").name, "incumbent")
            self.assertEqual(service.find_by_name("ml", "challenger").stage, "candidate")
            self.assertEqual(candidate.model_id, service.find_by_name("ml", "challenger").model_id)

    def test_refresh_does_not_bootstrap_a_newly_registered_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ModelRegistry(root / "registry.sqlite", snapshot_path=root / "registry.json")
            service.register(
                name="first-candidate",
                kind="ml",
                stage="candidate",
                run_id="first-run",
                model_path="first-model",
                metrics={"score": 0.99},
            )
            candidate_row = {"kind": "ml", "model_name": "first-candidate", "score": 0.99}
            with (
                patch("my_strategy.model_selector.default_registry", return_value=service),
                patch("my_strategy.model_selector.select_best_model", side_effect=[candidate_row, None]),
                patch("my_strategy.model_selector.write_db", return_value=False),
                patch("my_strategy.model_selector.write_files", return_value=False),
            ):
                result = refresh_best_models()

            self.assertIsNone(result["ml"])
            self.assertIsNone(service.champion("ml"))


if __name__ == "__main__":
    unittest.main()
