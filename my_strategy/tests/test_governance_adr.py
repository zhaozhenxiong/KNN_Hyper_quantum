#!/usr/bin/env python
"""Smoke tests for governance ADR primitives."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import shutil

import pandas as pd

from my_strategy.agents.contracts import AgentResult
from my_strategy.core.run_context import create_run_context
from my_strategy.execution import BrokerSimulator, CostModel, Order
from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS
from my_strategy.model_registry import ModelRegistry
from my_strategy.validation import (
    validate_feature_columns_no_lookahead,
    validate_feature_label_timing,
    validate_ohlcv,
    validate_signal_execution_timing,
    validate_time_split,
)


class GovernanceADRTests(unittest.TestCase):
    def test_run_context_creates_metadata(self) -> None:
        run = create_run_context(task="unit-test", as_of_date="2026-07-12", config={"x": 1}, run_id="unit-test-run")
        try:
            self.assertTrue((run.run_dir() / "metadata.json").exists())
            self.assertTrue(run.subdir("reports").exists())
        finally:
            shutil.rmtree(run.run_dir(), ignore_errors=True)

    def test_agent_result_contract(self) -> None:
        result = AgentResult(agent="factor-agent", task_id="t1", run_id="r1", status="success", signal="buy", score=0.2)
        self.assertEqual(result.to_dict()["agent"], "factor-agent")

    def test_model_registry_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(Path(tmp) / "registry.json")
            entry = registry.register(name="m1", kind="ml", stage="candidate", run_id="r1", model_path="model")
            promoted = registry.promote(entry.model_id, "champion")
            self.assertEqual(promoted.stage, "champion")
            self.assertEqual(registry.champion("ml").name, "m1")
            self.assertTrue((Path(tmp) / "registry.sqlite").exists())

    def test_broker_ledger(self) -> None:
        broker = BrokerSimulator(initial_cash=10000, cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, min_commission=0), run_id="r1")
        broker.submit(Order(date="2026-01-01", symbol="000001.SZ", side="BUY", shares=100, price=10.0), mark_price=10.0)
        broker.submit(Order(date="2026-01-02", symbol="000001.SZ", side="SELL", shares=100, price=11.0), mark_price=11.0)
        ledger = broker.ledger.to_frame()
        self.assertEqual(len(ledger), 2)
        self.assertAlmostEqual(float(ledger.iloc[-1]["cash_after"]), 10100.0)

    def test_validation_checks(self) -> None:
        ohlcv = pd.DataFrame(
            {
                "date": ["2026-01-01"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
                "volume": [1000],
            }
        )
        self.assertTrue(validate_ohlcv(ohlcv)["ok"])
        self.assertTrue(validate_time_split(train_end="2024-12-31", valid_end="2025-12-31", data_end="2026-07-12")["ok"])
        timing = pd.DataFrame({"signal_time": ["2026-01-01 15:00"], "execution_time": ["2026-01-02 09:30"]})
        self.assertTrue(validate_signal_execution_timing(timing)["ok"])

    def test_realistic_factor_timing_samples(self) -> None:
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        close = pd.Series([10, 10.2, 10.1, 10.5, 10.8, 10.7, 11.0, 11.2])
        frame = pd.DataFrame(
            {
                "date": dates,
                "ma_3": close.rolling(3).mean(),
                "future_return_1d": close.shift(-1) / close - 1,
                "feature_available_at": dates,
                "label_end_at": dates + pd.Timedelta(days=1),
            }
        ).dropna().reset_index(drop=True)
        self.assertTrue(
            validate_feature_label_timing(
                frame,
                feature_available_col="feature_available_at",
                label_end_col="label_end_at",
            )["ok"]
        )
        leaked = frame.copy()
        leaked["feature_available_at"] = leaked["date"] + pd.Timedelta(days=1)
        self.assertFalse(
            validate_feature_label_timing(
                leaked,
                feature_available_col="feature_available_at",
                label_end_col="label_end_at",
            )["ok"]
        )

    def test_future_named_columns_cannot_be_features(self) -> None:
        ok = validate_feature_columns_no_lookahead(["ma_5", "rsi_14"], label_columns=["future_return_5d"])
        bad = validate_feature_columns_no_lookahead(["ma_5", "future_return_5d"], label_columns=["label"])
        self.assertTrue(ok["ok"])
        self.assertFalse(bad["ok"])

    def test_project_default_factors_are_allowed_features(self) -> None:
        realistic_factors = list(DEFAULT_FEATURE_COLUMNS) + [
            "cs_rank_momentum_20d",
            "industry_neutral_momentum_rank",
            "board_neutral_volume_rank",
        ]
        result = validate_feature_columns_no_lookahead(realistic_factors, label_columns=["future_return_5d"])
        self.assertTrue(result["ok"], result["issues"])

    def test_future_leakage_feature_names_are_blocked(self) -> None:
        leaked_features = [
            "next_close",
            "lead_return_1d",
            "forward_rank_5d",
            "future_cross_sectional_rank",
            "tomorrow_limit_up_flag",
        ]
        result = validate_feature_columns_no_lookahead(leaked_features, label_columns=["future_return_5d"])
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["issues"]), len(leaked_features))

    def test_cross_sectional_factor_availability_timing(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
                "stock": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "industry": ["bank", "bank", "bank", "bank"],
                "cs_rank_momentum_20d": [0.2, 0.8, 0.4, 0.6],
                "industry_neutral_momentum_rank": [0.3, 0.7, 0.45, 0.55],
                "feature_available_at": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
                "label_end_at": ["2026-01-10", "2026-01-10", "2026-01-11", "2026-01-11"],
            }
        )
        self.assertTrue(
            validate_feature_label_timing(
                frame,
                feature_available_col="feature_available_at",
                label_end_col="label_end_at",
            )["ok"]
        )
        leaked = frame.copy()
        leaked.loc[leaked["date"] == "2026-01-05", "feature_available_at"] = "2026-01-06"
        self.assertFalse(
            validate_feature_label_timing(
                leaked,
                feature_available_col="feature_available_at",
                label_end_col="label_end_at",
            )["ok"]
        )


if __name__ == "__main__":
    unittest.main()
