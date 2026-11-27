#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Composite strategies built from Qbot-inspired rule strategies."""

from __future__ import annotations

import pandas as pd

from .rule_strategies import RULE_STRATEGIES
from .strategy_base import BaseLabStrategy, StrategySpec, make_output


class ComboStrategy(BaseLabStrategy):
    spec = StrategySpec("combo_base", "combo", "qbot_combo", "base combo")
    members: list[tuple[str, float]] = []
    buy_score = 0.25
    sell_score = -0.10

    def generate(self, df, config):
        max_position = float(config.get("backtest", {}).get("max_single_position", 0.30))
        total_weight = sum(weight for _, weight in self.members)
        score = pd.Series(0.0, index=df.index)
        member_reasons = []
        for name, weight in self.members:
            child = RULE_STRATEGIES[name]().generate(df.copy(), config)
            child_score = pd.Series(child["score"].values, index=df.index)
            score += child_score * weight / total_weight
            member_reasons.append(name)

        raw_pos = pd.Series(index=df.index, dtype=float)
        raw_pos[score >= self.buy_score] = max_position
        raw_pos[score <= self.sell_score] = 0.0
        target_weight = raw_pos.ffill().fillna(0.0)
        return make_output(df, self.spec.name, score, target_weight, "组合策略: " + ", ".join(member_reasons))


class TrendComboStrategy(ComboStrategy):
    spec = StrategySpec("trend_combo", "combo_trend", "Qbot strategy mix", "MA + MACD + Aroon + VolumeBreakout")
    members = [("ma_cross", 0.30), ("macd", 0.25), ("aroon", 0.25), ("volume_breakout", 0.20)]
    buy_score = 0.30
    sell_score = -0.05


class ReversalComboStrategy(ComboStrategy):
    spec = StrategySpec("reversal_combo", "combo_reversal", "Qbot strategy mix", "RSI + BOLL + StochRSI")
    members = [("rsi_reversal", 0.40), ("boll_reversion", 0.30), ("stoch_rsi", 0.30)]
    buy_score = 0.25
    sell_score = -0.15


class BalancedComboStrategy(ComboStrategy):
    spec = StrategySpec("balanced_combo", "combo_balanced", "Qbot strategy mix", "Trend + reversal + breakout balanced vote")
    members = [
        ("ma_cross", 0.18),
        ("macd", 0.16),
        ("aroon", 0.14),
        ("volume_breakout", 0.12),
        ("rsi_reversal", 0.16),
        ("stoch_rsi", 0.12),
        ("boll_reversion", 0.12),
    ]
    buy_score = 0.22
    sell_score = -0.08


COMBO_STRATEGIES = {
    cls.spec.name: cls
    for cls in [TrendComboStrategy, ReversalComboStrategy, BalancedComboStrategy]
}
