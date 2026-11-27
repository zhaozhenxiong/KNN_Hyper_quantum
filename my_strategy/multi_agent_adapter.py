#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter that exposes KHQuant-MultiAgent-V1 as a lab strategy."""

from __future__ import annotations

from pathlib import Path

import yaml

from my_strategy.agents.factor_agent import FactorAgent
from my_strategy.agents.multi_agent_engine import MultiAgentEngine

from .strategy_base import BaseLabStrategy, StrategySpec


class MultiAgentV1Strategy(BaseLabStrategy):
    spec = StrategySpec("multi_agent_v1", "multi_agent", "my_strategy agents", "Trend/Reversal/Volume/Risk multi-agent")

    def generate(self, df, config):
        cfg_path = Path(__file__).resolve().parent / "configs" / "multi_agent_config.yaml"
        ma_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        ma_config["backtest"]["max_single_position"] = config.get("backtest", {}).get("max_single_position", 0.30)
        factors = FactorAgent.add_factors(df, volume_window=ma_config.get("agents", {}).get("volume", {}).get("volume_window", 20))
        decisions = MultiAgentEngine(ma_config).generate_decisions(factors)
        out = decisions.rename(columns={"final_score": "score"})
        out["strategy_name"] = self.spec.name
        out["reason"] = out["reason_json"]
        return out[["date", "strategy_name", "close", "score", "target_weight", "final_action", "reason"]]
