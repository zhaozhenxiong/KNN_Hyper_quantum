#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RL policy agent for position sizing suggestions."""

from __future__ import annotations

import json

from .base_agent import AgentSignal, BaseAgent
from my_strategy.rl.features import state_from_engine_row
from my_strategy.rl.predictor import predict_action_values


class RLPolicyAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "",
        model_root: str | None = None,
        weight: float = 0.10,
        blend: float = 0.15,
        min_confidence: float = 0.05,
        allow_entry: bool = False,
        enabled: bool = False,
    ):
        super().__init__("RLPolicyAgent", weight=weight, enabled=enabled and bool(model_name))
        self.model_name = model_name
        self.model_root = model_root
        self.blend = max(0.0, min(1.0, float(blend)))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.allow_entry = bool(allow_entry)

    def run(self, row, context=None) -> AgentSignal:
        context = context or {}
        base_score = float(context.get("base_final_score", 0.0) or 0.0)
        base_weight = float(context.get("base_target_weight", 0.0) or 0.0)
        if not self.allow_entry and base_weight <= 0:
            reason = "RL仓位Agent: 基础策略未给仓位，且 allow_entry=false，跳过独立开仓建议"
            return AgentSignal(
                self.name,
                0.0,
                0.0,
                "hold",
                reason,
                {
                    "rl_model_name": self.model_name,
                    "rl_target_weight": base_weight,
                    "rl_confidence": 0.0,
                    "rl_action": "hold",
                    "rl_q_values": "{}",
                    "rl_reason": reason,
                },
            )
        try:
            state = state_from_engine_row(row, context, base_score, base_weight)
            pred = predict_action_values(state, self.model_name, self.model_root)
            target_weight = float(pred["best_weight"])
            confidence = float(pred["confidence"])
            q_values = pred["q_values"]
            if target_weight <= 0:
                action = "watch"
            elif target_weight > base_weight + 0.05:
                action = "increase"
            elif target_weight < base_weight - 0.05:
                action = "reduce"
            else:
                action = "hold"
            reason = (
                f"RL仓位Agent: 模型={self.model_name}, 建议仓位={target_weight:.0%}, "
                f"基准仓位={base_weight:.0%}, 置信度={confidence:.3f}"
            )
            score = (target_weight - base_weight) / 0.9 if 0.9 else 0.0
            return AgentSignal(
                self.name,
                score,
                confidence,
                action,
                reason,
                {
                    "rl_model_name": self.model_name,
                    "rl_target_weight": target_weight,
                    "rl_confidence": confidence,
                    "rl_action": action,
                    "rl_q_values": json.dumps(q_values, ensure_ascii=False),
                    "rl_reason": reason,
                },
            )
        except Exception as exc:
            reason = f"RL仓位Agent跳过: {exc}"
            return AgentSignal(
                self.name,
                0.0,
                0.0,
                "hold",
                reason,
                {
                    "rl_model_name": self.model_name,
                    "rl_target_weight": base_weight,
                    "rl_confidence": 0.0,
                    "rl_action": "hold",
                    "rl_q_values": "{}",
                    "rl_reason": reason,
                },
            )
