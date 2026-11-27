"""Regression coverage for locally calculated technical indicators."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from my_strategy.agents.factor_agent import FactorAgent
from my_strategy.indicators import arbr, cci, expma, macd, obv, sar
from my_strategy.ml.feature_builder import (
    DEFAULT_FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
    build_ml_features,
    clean_feature_matrix,
    feature_columns_for_profile,
)
from my_strategy.rule_strategies import RULE_STRATEGIES
from my_strategy.validation.strategy_acceptance import run_leakage_acceptance


def sample_ohlcv(rows: int = 90) -> pd.DataFrame:
    index = pd.RangeIndex(rows)
    close = 20 + index * 0.08 + np.sin(index / 3)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "open": close - 0.15,
            "high": close + 0.55,
            "low": close - 0.65,
            "close": close,
            "volume": 1_000_000 + index * 10_000,
            "amount": (1_000_000 + index * 10_000) * close,
        }
    )


class TechnicalIndicatorTests(unittest.TestCase):
    def test_indicator_functions_return_finite_series_after_warmup(self) -> None:
        frame = sample_ohlcv()
        line, signal, histogram = macd(frame["close"], 5, 10, 6)
        ar, br = arbr(frame)
        fast, slow = expma(frame["close"], 5, 10)
        values = [line, signal, histogram, ar, br, cci(frame), obv(frame["close"], frame["volume"]), sar(frame), fast, slow]
        for value in values:
            self.assertEqual(len(value), len(frame))
            self.assertTrue(np.isfinite(value.iloc[30:]).all())

    def test_factor_and_feature_contract_contains_new_indicators(self) -> None:
        factors = FactorAgent.add_factors(sample_ohlcv())
        expected = set(TECHNICAL_FEATURE_COLUMNS) | {"boll_upper", "boll_mid", "boll_lower", "obv", "sar"}
        self.assertTrue(expected.issubset(factors.columns))
        features = build_ml_features(factors)
        self.assertTrue(set(DEFAULT_FEATURE_COLUMNS).issubset(features.columns))
        self.assertTrue(np.isfinite(features[DEFAULT_FEATURE_COLUMNS].iloc[60:].to_numpy(dtype=float)).all())

    def test_new_rule_strategies_follow_standard_output_contract(self) -> None:
        frame = sample_ohlcv()
        config = {"backtest": {"max_single_position": 0.30}}
        expected_names = {
            "macd_5_10_6",
            "macd_kdj_resonance",
            "arbr_reversal",
            "cci_reversal",
            "obv_trend",
            "sar_trend",
            "expma_cross",
        }
        self.assertTrue(expected_names.issubset(RULE_STRATEGIES))
        for name in expected_names:
            result = RULE_STRATEGIES[name]().generate(frame, config)
            self.assertEqual(len(result), len(frame))
            self.assertTrue({"score", "target_weight", "final_action"}.issubset(result.columns))
            self.assertTrue(result["score"].between(-1, 1).all())
            self.assertTrue(result["target_weight"].between(0, 1).all())

    def test_extended_feature_names_pass_static_leakage_check(self) -> None:
        report = run_leakage_acceptance(
            run_id="technical-indicator-unit",
            feature_columns=DEFAULT_FEATURE_COLUMNS,
            train_end="2025-12-31",
            valid_end="2026-06-30",
            data_end="2026-07-15",
        )
        self.assertEqual(report.status, "success", report.to_dict())

    def test_feature_profiles_keep_baseline_and_extended_contracts_distinct(self) -> None:
        self.assertEqual(feature_columns_for_profile("legacy"), LEGACY_FEATURE_COLUMNS)
        self.assertEqual(feature_columns_for_profile("extended"), DEFAULT_FEATURE_COLUMNS)
        self.assertTrue(set(TECHNICAL_FEATURE_COLUMNS).isdisjoint(LEGACY_FEATURE_COLUMNS))

    def test_clean_feature_matrix_coerces_sqlite_text_features(self) -> None:
        frame = pd.DataFrame({column: ["1.25"] for column in DEFAULT_FEATURE_COLUMNS})
        matrix = clean_feature_matrix(frame)
        self.assertTrue(np.issubdtype(matrix.dtypes.iloc[0], np.number))
        self.assertTrue(np.isfinite(matrix.to_numpy(dtype=float)).all())

    def test_indicator_values_do_not_change_when_future_rows_are_added(self) -> None:
        short = sample_ohlcv(60)
        full = sample_ohlcv(90)
        short_factors = FactorAgent.add_factors(short)
        full_factors = FactorAgent.add_factors(full).iloc[: len(short)]
        columns = ["macd_5_10_6", "boll_pct_b", "ar_26", "br_26", "cci_14", "obv", "sar", "expma_5_10_ratio"]
        for column in columns:
            np.testing.assert_allclose(
                short_factors[column].to_numpy(dtype=float),
                full_factors[column].to_numpy(dtype=float),
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
