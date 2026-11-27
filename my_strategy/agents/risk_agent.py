#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Risk control agent."""

from __future__ import annotations

from .base_agent import AgentSignal, BaseAgent


class RiskAgent(BaseAgent):
    def __init__(
        self,
        stop_loss=0.08,
        take_profit=0.20,
        max_drawdown=0.15,
        max_volatility=0.08,
        enabled=True,
    ):
        super().__init__("RiskAgent", weight=1.0, enabled=enabled)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.max_drawdown = float(max_drawdown)
        self.max_volatility = float(max_volatility)

    def run(self, row, context=None):
        context = context or {}
        entry_price = context.get("entry_price")
        current_position = context.get("current_position", 0)
        current_drawdown = context.get("current_drawdown", 0)
        close = float(row["close"])
        volatility = float(row.get("volatility_20d", 0) or 0)

        if current_position > 0 and entry_price:
            pnl = close / entry_price - 1
            if pnl <= -self.stop_loss:
                return AgentSignal(self.name, -1, 1, "force_sell", f"触发止损，当前亏损={pnl:.2%}")
            if pnl >= self.take_profit:
                return AgentSignal(self.name, -0.5, 0.8, "take_profit", f"达到止盈区间，当前收益={pnl:.2%}")

        if current_position > 0 and current_drawdown <= -self.max_drawdown:
            return AgentSignal(self.name, -1, 1, "force_sell", f"组合回撤超过限制，当前回撤={current_drawdown:.2%}")

        if volatility > self.max_volatility:
            return AgentSignal(self.name, -0.6, 0.8, "reduce", f"波动率过高，20日波动率={volatility:.2%}")

        return AgentSignal(self.name, 0, 0.5, "pass", "风险检查通过")
