"""Regression checks for isolated candidate fusion-policy inputs."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from my_strategy.fusion_registry import apply_best_fusion_policy
from my_strategy.run_multi_agent_backtest import load_config
from my_strategy.train_rl_agent import parse_args as parse_rl_args
from my_strategy.rl.features import JOINT_Q_FEATURE_COLUMNS, clean_feature_frame, state_from_engine_row


class FusionCandidateConfigTests(unittest.TestCase):
    def test_candidate_policy_adds_sentiment_without_mutating_registry(self) -> None:
        config = load_config(Path("my_strategy/configs/multi_agent_config.yaml"))
        candidate = Path("my_strategy/configs/fusion_candidates/extended_ml_dl_sentiment_v1.json")

        self.assertTrue(apply_best_fusion_policy(config, path=candidate))
        weights = config["decision"]["fusion"]["weights"]

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertIn("sentiment", weights)
        self.assertAlmostEqual(weights["sentiment"], 0.10)

    def test_rl_no_promote_flag_preserves_candidate_workflow(self) -> None:
        with patch("sys.argv", ["train_rl_agent.py", "--no-promote"]):
            args = parse_rl_args()

        self.assertTrue(args.no_promote)

    def test_joint_state_uses_whitelist_and_excludes_reward(self) -> None:
        state = state_from_engine_row({"close": 10.0, "boll_pct_b": 0.7, "cci_14": -20.0}, {}, 0.2, 0.3)
        frame = clean_feature_frame(pd.DataFrame([{**state, "action_weight": 0.3, "turnover": 0.0, "reward": 99.0}]), JOINT_Q_FEATURE_COLUMNS)

        self.assertIn("boll_pct_b", frame.columns)
        self.assertNotIn("reward", frame.columns)


if __name__ == "__main__":
    unittest.main()
