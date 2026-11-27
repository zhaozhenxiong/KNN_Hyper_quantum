"""Regression tests for point-in-time external market sentiment."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from my_strategy.agents.base_agent import AgentSignal
from my_strategy.agents.fusion_decision import FusionDecisionPolicy
from my_strategy.agents.market_sentiment_agent import MarketSentimentAgent
from my_strategy.agents.multi_agent_engine import MultiAgentEngine
from my_strategy.data_manager.external_market import (
    ExternalMarketCache,
    download_global_index_history,
    enrich_with_external_sentiment,
)
from my_strategy.ml.external_sentiment import build_training_frame, feature_columns, train_classifier


class ExternalMarketSentimentTests(unittest.TestCase):
    def test_sp500_falls_back_to_fred_when_primary_sources_fail(self) -> None:
        response = Mock()
        response.text = "observation_date,SP500\n2026-07-15,6200.00\n2026-07-16,6210.00\n"
        response.raise_for_status.return_value = None
        with (
            patch("my_strategy.data_manager.external_market.download_eastmoney_history", side_effect=RuntimeError("tls")),
            patch("akshare.index.index_global_em.index_global_em_symbol_map", {"sp": {"market": "100", "code": "SPX"}}),
            patch("akshare.index.index_global_em.index_global_hist_em", side_effect=RuntimeError("blocked")),
            patch("my_strategy.data_manager.external_market.requests.get", return_value=response),
        ):
            frame, source = download_global_index_history("sp", instrument_id="sp500")
        self.assertEqual(source, "fred_sp500")
        self.assertEqual(frame["date"].astype(str).tolist(), ["2026-07-15", "2026-07-16"])
        self.assertTrue((frame[["open", "high", "low"]].eq(frame["close"], axis=0)).all().all())

    def test_us_close_is_not_visible_before_next_a_share_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ExternalMarketCache({"data_root": directory})
            instrument = {
                "id": "sp500",
                "symbol": "SP500",
                "market": "US",
                "group": "index",
                "weight": 1.0,
                "timezone": "America/New_York",
                "close_time": "16:00",
            }
            dates = pd.bdate_range("2026-01-01", "2026-01-12")
            close = pd.Series(range(100, 100 + len(dates)), dtype=float)
            cache.save(
                instrument,
                pd.DataFrame({"date": dates, "open": close - 0.5, "high": close + 0.5, "low": close - 1.0, "close": close}),
                source="test",
            )
            factors = pd.DataFrame({"date": pd.to_datetime(["2026-01-09", "2026-01-12"]), "close": [10.0, 10.2]})
            config = {
                "external_sentiment": {
                    "enabled": True,
                    "max_stale_days": 5,
                    "instruments": [instrument],
                }
            }
            enriched = enrich_with_external_sentiment(factors, config, cache)
            monday = enriched.iloc[1]
            self.assertEqual(monday["external_sentiment_asof"], "2026-01-09")
            self.assertEqual(monday["external_sentiment_status"], "ok")
            self.assertGreater(monday["external_sentiment_score"], 0.0)

    def test_unavailable_sentiment_is_excluded_from_fusion_weight(self) -> None:
        agent = MarketSentimentAgent(enabled=True, min_confidence=0.35)
        signal = agent.run({"external_sentiment_score": -0.9, "external_sentiment_confidence": 0.0, "external_sentiment_status": "unavailable"})
        policy = FusionDecisionPolicy({"weights": {"sentiment": 0.10}})
        score, confidence, breakdown = policy._weighted_score([signal])
        self.assertEqual(score, 0.0)
        self.assertEqual(confidence, 0.0)
        self.assertTrue(breakdown["sentiment"]["skipped"])

    def test_negative_confirmed_sentiment_blocks_new_entry(self) -> None:
        policy = FusionDecisionPolicy({"sentiment_entry_block_score": -0.55, "sentiment_min_confidence": 0.60})
        risk = AgentSignal("RiskAgent", 0.0, 1.0, "pass", "ok")
        filters = policy._entry_filters(
            {"external_sentiment_score": -0.70, "external_sentiment_confidence": 0.80},
            risk,
        )
        self.assertFalse(filters["allow_entry"])
        self.assertIn("external_sentiment", filters["blocked_reason"])

    def test_registered_policy_keeps_default_sentiment_weight(self) -> None:
        engine = MultiAgentEngine(
            {
                "agents": {"market_sentiment": {"enabled": True, "weight": 0.10}},
                "decision": {"fusion": {"weights": {"trend": 0.50}}},
            }
        )
        self.assertEqual(engine.fusion_policy.weights["sentiment"], 0.10)

    def test_training_frame_uses_local_benchmark_and_time_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ExternalMarketCache({"data_root": directory})
            dates = pd.bdate_range("2023-01-02", "2026-07-16")
            move = np.where(np.arange(len(dates)) % 2 == 0, 0.006, -0.006)
            external_close = 100.0 * np.cumprod(1.0 + move)
            instrument = {
                "id": "sp500",
                "symbol": "SP500",
                "market": "US",
                "group": "index",
                "weight": 1.0,
                "timezone": "America/New_York",
                "close_time": "16:00",
            }
            cache.save(
                instrument,
                pd.DataFrame({"date": dates, "open": external_close, "high": external_close * 1.01, "low": external_close * 0.99, "close": external_close}),
                source="test",
            )
            benchmark_open = 4000.0 + np.arange(len(dates))
            benchmark_close = benchmark_open * (1.0 + np.where(move > 0, 0.003, -0.003))
            cache.save(
                {"id": "csi300", "symbol": "000300.SH", "timezone": "Asia/Shanghai", "close_time": "15:00"},
                pd.DataFrame({"date": dates, "open": benchmark_open, "high": benchmark_open * 1.01, "low": benchmark_open * 0.99, "close": benchmark_close}),
                source="test",
            )
            config = {
                "external_sentiment": {
                    "enabled": True,
                    "use_trained_model": False,
                    "instruments": [instrument],
                    "training_benchmark": {"id": "csi300"},
                }
            }
            frame = build_training_frame(config, cache)
            columns = feature_columns(config)
            self.assertIn("target_up", frame.columns)
            self.assertGreater(len(frame), 500)
            _, metrics = train_classifier(
                frame,
                columns,
                train_start="2023-01-01",
                train_end="2024-12-31",
                valid_end="2025-12-31",
                test_end="2026-07-16",
            )
            self.assertGreater(metrics["test"]["rows"], 50)


if __name__ == "__main__":
    unittest.main()
