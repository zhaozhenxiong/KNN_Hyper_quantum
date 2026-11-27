#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule-based strategy agents."""

from __future__ import annotations

import pandas as pd

from .base_agent import AgentSignal, BaseAgent, clamp


def safe_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


class TrendAgent(BaseAgent):
    def __init__(self, weight=0.45, short_ma="ma20", long_ma="ma60", enabled=True):
        super().__init__("TrendAgent", weight, enabled)
        self.short_ma = short_ma
        self.long_ma = long_ma

    def run(self, row, context=None):
        close = safe_number(row["close"])
        short_ma = safe_number(row[self.short_ma], close)
        long_ma = safe_number(row[self.long_ma], close)
        macd = safe_number(row.get("macd", 0))
        macd_signal = safe_number(row.get("macd_signal", 0))

        score = 0.0
        reasons = []
        if close > short_ma:
            score += 0.3
            reasons.append("收盘价高于短期均线")
        else:
            score -= 0.3
            reasons.append("收盘价低于短期均线")

        if short_ma > long_ma:
            score += 0.4
            reasons.append("短期均线高于长期均线")
        else:
            score -= 0.4
            reasons.append("短期均线低于长期均线")

        if macd > macd_signal:
            score += 0.3
            reasons.append("MACD处于多头状态")
        else:
            score -= 0.2
            reasons.append("MACD处于弱势状态")

        score = clamp(score)
        action = "buy" if score >= 0.5 else "sell" if score <= -0.3 else "hold"
        return AgentSignal(self.name, score, abs(score), action, "；".join(reasons))


class ReversalAgent(BaseAgent):
    def __init__(self, weight=0.30, rsi_low=35, rsi_high=75, enabled=True):
        super().__init__("ReversalAgent", weight, enabled)
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    def run(self, row, context=None):
        rsi = safe_number(row["rsi14"], 50)
        if rsi < 30:
            score, action, reason = 0.8, "buy", f"RSI={rsi:.2f}，明显超卖，存在反弹机会"
        elif rsi < self.rsi_low:
            score, action, reason = 0.5, "buy", f"RSI={rsi:.2f}，偏低，存在低吸机会"
        elif rsi > 85:
            score, action, reason = -0.8, "sell", f"RSI={rsi:.2f}，严重超买，回调风险较高"
        elif rsi > self.rsi_high:
            score, action, reason = -0.5, "sell", f"RSI={rsi:.2f}，偏高，短期过热"
        else:
            score, action, reason = 0.1, "hold", f"RSI={rsi:.2f}，处于中性区间"
        return AgentSignal(self.name, score, abs(score), action, reason)


class VolumeAgent(BaseAgent):
    def __init__(self, weight=0.25, volume_ratio_buy=1.3, enabled=True):
        super().__init__("VolumeAgent", weight, enabled)
        self.volume_ratio_buy = volume_ratio_buy

    def run(self, row, context=None):
        volume_ratio = safe_number(row.get("volume_ratio", 1), 1)
        if volume_ratio >= 2.0:
            score, action, reason = 0.7, "buy", f"成交量显著放大，量比={volume_ratio:.2f}"
        elif volume_ratio >= self.volume_ratio_buy:
            score, action, reason = 0.4, "buy", f"成交量温和放大，量比={volume_ratio:.2f}"
        elif volume_ratio < 0.6:
            score, action, reason = -0.2, "hold", f"成交量明显萎缩，量比={volume_ratio:.2f}"
        else:
            score, action, reason = 0.0, "hold", f"成交量正常，量比={volume_ratio:.2f}"
        return AgentSignal(self.name, score, abs(score), action, reason)


class MLAgent(BaseAgent):
    """Placeholder for future Qlib predictions."""

    def __init__(self, weight=0.20, prediction_column="pred_return_5d", enabled=False):
        super().__init__("MLAgent", weight, enabled)
        self.prediction_column = prediction_column

    def run(self, row, context=None):
        pred_return = safe_number(row.get(self.prediction_column, 0), 0)
        if pred_return > 0.03:
            score, action = 0.8, "buy"
        elif pred_return < -0.02:
            score, action = -0.7, "sell"
        else:
            score, action = 0.0, "hold"
        reason = f"Qlib预测列 {self.prediction_column}={pred_return:.2%}"
        return AgentSignal(self.name, score, abs(score), action, reason)

