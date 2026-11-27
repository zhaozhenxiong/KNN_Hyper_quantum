"""Regression tests for ML/DL runtime status reporting."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from my_strategy.agents.deep_learning_agent import DeepLearningAgent
from my_strategy.agents.ml_score_agent import MLScoreAgent
from my_strategy.model_runtime import enrich_configured_model_predictions
from my_strategy.run_multi_agent_backtest import simulate_trades


class ModelRuntimeStatusTests(unittest.TestCase):
    def test_disabled_models_are_explicit_instead_of_missing(self) -> None:
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "close": [1.0, 1.1, 1.2]})
        out, runtime = enrich_configured_model_predictions(frame, {"ml": {"enabled": False}, "deep_learning": {"enabled": False}})
        self.assertEqual(set(out["ml_status"]), {"disabled"})
        self.assertEqual(set(out["dl_status"]), {"disabled"})
        self.assertEqual(runtime["ml"]["status"], "disabled")

    def test_loaded_models_mark_ready_and_dl_warmup(self) -> None:
        frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4), "close": [1.0, 1.1, 1.2, 1.3]})

        def fake_ml(data, *_args, **_kwargs):
            out = data.copy()
            out["ml_score"] = [0.0, 0.2, -0.3, 0.4]
            out["ml_confidence"] = [0.0, 0.2, 0.3, 0.4]
            out["pred_rank_future_return_5d"] = out["ml_score"]
            out["ml_model_name"] = "ml-final"
            out.attrs["ml_metadata"] = {"prediction_column": "pred_rank_future_return_5d", "prediction_kind": "cross_sectional_rank_score"}
            return out

        def fake_dl(data, *_args, **_kwargs):
            out = data.copy()
            out["dl_prediction"] = [np.nan, np.nan, 0.3, -0.2]
            out["dl_score"] = out["dl_prediction"]
            out["dl_confidence"] = out["dl_score"].abs()
            out["dl_model_name"] = "dl-final"
            out.attrs["dl_metadata"] = {"prediction_column": "dl_prediction", "score_column": "dl_score", "confidence_column": "dl_confidence", "lookback": 2}
            return out

        agents = {
            "ml": {"enabled": True, "model_name": "ml-final"},
            "ml_score": {"enabled": False},
            "deep_learning": {"enabled": True, "model_name": "dl-final"},
        }
        with patch("my_strategy.model_runtime.enrich_with_ml_predictions", side_effect=fake_ml), patch(
            "my_strategy.model_runtime.enrich_with_deep_predictions", side_effect=fake_dl
        ):
            out, runtime = enrich_configured_model_predictions(frame, agents)

        self.assertTrue(agents["ml_score"]["enabled"])
        self.assertEqual(list(out["ml_status"]), ["ready"] * 4)
        self.assertEqual(list(out["dl_status"]), ["warmup", "prediction_missing", "ready", "ready"])
        self.assertEqual(runtime["deep_learning"]["status"], "ready")

    def test_agents_distinguish_neutral_from_unavailable(self) -> None:
        ml = MLScoreAgent(enabled=True, neutral_zone=0.10)
        dl = DeepLearningAgent(enabled=True, neutral_zone=0.10)
        self.assertEqual(ml.run({"ml_score": 0.05, "ml_status": "ready"}).extra["ml_status"], "neutral")
        self.assertEqual(dl.run({"dl_score": 0.40, "dl_status": "ready"}).extra["dl_status"], "active")
        self.assertEqual(dl.run({"dl_score": 0.0, "dl_status": "warmup"}).extra["dl_status"], "warmup")

    def test_daily_ledger_prefers_agent_status_over_factor_status(self) -> None:
        date = pd.Timestamp("2026-01-05")
        factors = pd.DataFrame(
            [{"date": date, "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "ml_status": "ready", "ml_reason": "prediction available", "dl_status": "ready", "dl_reason": "prediction available"}]
        )
        decisions = pd.DataFrame(
            [{"date": date, "final_score": 0.0, "final_action": "hold", "target_weight": 0.0, "risk_action": "pass", "reason_json": "{}", "ml_status": "neutral", "ml_reason": "inside neutral zone", "dl_status": "neutral", "dl_reason": "inside neutral zone"}]
        )
        daily, _ = simulate_trades(factors, decisions, {"stock": "000725.SZ", "backtest": {"initial_cash": 100000, "lot_size": 100}})
        self.assertEqual(daily.iloc[0]["ml_status"], "neutral")
        self.assertEqual(daily.iloc[0]["dl_status"], "neutral")


if __name__ == "__main__":
    unittest.main()
