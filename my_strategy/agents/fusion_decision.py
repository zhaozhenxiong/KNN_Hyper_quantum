#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nonlinear fusion policy for Multi-Agent buy/sell decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .base_agent import AgentSignal, clamp


AGENT_KEY_MAP = {
    "TrendAgent": "trend",
    "ReversalAgent": "reversal",
    "VolumeAgent": "volume",
    "MLScoreAgent": "ml",
    "DeepLearningAgent": "dl",
    "ProbabilisticAgent": "prob",
}


@dataclass
class FusionResult:
    final_score: float
    final_action: str
    target_weight: float
    base_target_weight: float
    raw_score: float
    confidence_score: float
    entry_probability: float
    exit_probability: float
    activation: str
    reason: str
    breakdown: dict[str, Any]


class FusionDecisionPolicy:
    def __init__(self, config: dict[str, Any] | None = None, portfolio_cfg: dict[str, Any] | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.activation = str(cfg.get("activation", "sigmoid")).lower()
        self.temperature = float(cfg.get("temperature", 2.2))
        self.buy_probability = float(cfg.get("buy_probability", 0.62))
        self.hold_probability = float(cfg.get("hold_probability", 0.54))
        self.sell_probability = float(cfg.get("sell_probability", 0.65))
        self.min_ml_score = float(cfg.get("min_ml_score", -0.25))
        self.min_prob_score = float(cfg.get("min_prob_score", -0.15))
        self.blocked_markov_states = set(cfg.get("blocked_markov_states", ["bear", "bear_high_vol"]))
        self.weights = dict(cfg.get("weights") or {})
        self.use_confidence = bool(cfg.get("use_confidence", True))

        p_cfg = portfolio_cfg or {}
        self.min_entry_weight = float(p_cfg.get("min_entry_weight", 0.30))
        self.mid_weight = float(p_cfg.get("mid_weight", 0.50))
        self.high_weight = float(p_cfg.get("high_weight", 0.70))
        self.max_weight = float(p_cfg.get("max_weight", 0.90))
        self.reduce_weight = float(p_cfg.get("reduce_weight", 0.30))
        self.high_probability = float(cfg.get("high_probability", 0.72))
        self.max_probability = float(cfg.get("max_probability", 0.86))

    def decide(
        self,
        agent_results: list[AgentSignal],
        row,
        risk_signal: AgentSignal,
        fallback_score: float,
        fallback_action: str,
        fallback_weight: float,
    ) -> FusionResult:
        if not self.enabled:
            return FusionResult(
                final_score=float(fallback_score),
                final_action=str(fallback_action),
                target_weight=float(fallback_weight),
                base_target_weight=float(fallback_weight),
                raw_score=float(fallback_score),
                confidence_score=abs(float(fallback_score)),
                entry_probability=_sigmoid(self.temperature * float(fallback_score)),
                exit_probability=_sigmoid(-self.temperature * float(fallback_score)),
                activation="linear_fallback",
                reason="Fusion disabled; using weighted Agent score.",
                breakdown={},
            )

        raw_score, confidence_score, breakdown = self._weighted_score(agent_results)
        adjusted = raw_score * confidence_score if self.use_confidence else raw_score
        final_score = self._activate(adjusted)
        entry_probability = _sigmoid(self.temperature * adjusted)
        exit_probability = _sigmoid(-self.temperature * adjusted)
        filters = self._entry_filters(row, risk_signal)

        if risk_signal.action in {"force_sell", "take_profit"}:
            action = "sell"
        elif risk_signal.action == "reduce":
            action = "reduce"
        elif exit_probability >= self.sell_probability:
            action = "sell"
        elif entry_probability >= self.buy_probability and filters["allow_entry"]:
            action = "buy"
        elif entry_probability >= self.hold_probability and filters["allow_entry"]:
            action = "hold"
        else:
            action = "watch"

        target_weight = self._target_weight(action, entry_probability, risk_signal.action)
        reason = (
            f"Fusion: raw={raw_score:.3f}, conf={confidence_score:.3f}, score={final_score:.3f}, "
            f"entry={entry_probability:.3f}, exit={exit_probability:.3f}, action={action}"
        )
        if not filters["allow_entry"]:
            reason += f", entry_filter={filters['blocked_reason']}"

        return FusionResult(
            final_score=float(final_score),
            final_action=action,
            target_weight=float(target_weight),
            base_target_weight=float(target_weight),
            raw_score=float(raw_score),
            confidence_score=float(confidence_score),
            entry_probability=float(entry_probability),
            exit_probability=float(exit_probability),
            activation=self.activation,
            reason=reason,
            breakdown=breakdown,
        )

    def _weighted_score(self, agent_results: list[AgentSignal]) -> tuple[float, float, dict[str, Any]]:
        weighted_score = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        breakdown: dict[str, Any] = {}
        for signal in agent_results:
            key = AGENT_KEY_MAP.get(signal.agent_name)
            if not key:
                continue
            weight = float(self.weights.get(key, signal.weight if hasattr(signal, "weight") else 1.0))
            if weight <= 0:
                continue
            score = clamp(signal.score)
            confidence = max(0.0, min(1.0, float(signal.confidence)))
            weighted_score += score * weight
            weighted_confidence += confidence * weight
            total_weight += weight
            breakdown[key] = {
                "agent": signal.agent_name,
                "weight": round(weight, 6),
                "score": round(score, 6),
                "confidence": round(confidence, 6),
                "weighted_score": round(score * weight, 6),
                "action": signal.action,
            }
        if total_weight <= 0:
            return 0.0, 0.0, breakdown
        return weighted_score / total_weight, weighted_confidence / total_weight, breakdown

    def _activate(self, value: float) -> float:
        if self.activation == "tanh":
            return float(math.tanh(self.temperature * value))
        if self.activation == "softsign":
            return float(value / (1.0 + abs(value)))
        if self.activation == "linear":
            return float(clamp(value))
        return float(_sigmoid(self.temperature * value) * 2.0 - 1.0)

    def _entry_filters(self, row, risk_signal: AgentSignal) -> dict[str, Any]:
        reasons: list[str] = []
        if risk_signal.action not in {"pass", "hold"}:
            reasons.append(f"risk={risk_signal.action}")
        if "ml_score_signal" in row.index:
            ml_score = _safe_number(row.get("ml_score_signal"), 0.0)
            if ml_score < self.min_ml_score:
                reasons.append(f"ml_score_signal={ml_score:.3f}<{self.min_ml_score:.3f}")
        if "prob_score_signal" in row.index:
            prob_score = _safe_number(row.get("prob_score_signal"), row.get("prob_score", 0.0))
            if prob_score < self.min_prob_score:
                reasons.append(f"prob_score_signal={prob_score:.3f}<{self.min_prob_score:.3f}")
        markov_state = str(row.get("markov_state", "") or "")
        if markov_state in self.blocked_markov_states:
            reasons.append(f"markov_state={markov_state}")
        return {"allow_entry": not reasons, "blocked_reason": "; ".join(reasons)}

    def _target_weight(self, action: str, entry_probability: float, risk_action: str) -> float:
        if action == "sell" or risk_action in {"force_sell", "take_profit"}:
            return 0.0
        if action == "reduce" or risk_action == "reduce":
            return min(self.reduce_weight, self.max_weight)
        if action not in {"buy", "hold"}:
            return 0.0
        if entry_probability >= self.max_probability:
            return self.max_weight
        if entry_probability >= self.high_probability:
            span = max(1e-9, self.max_probability - self.high_probability)
            ratio = (entry_probability - self.high_probability) / span
            return self.high_weight + min(max(ratio, 0.0), 1.0) * (self.max_weight - self.high_weight)
        span = max(1e-9, self.high_probability - self.buy_probability)
        ratio = (entry_probability - self.buy_probability) / span
        if action == "hold" and entry_probability < self.buy_probability:
            hold_span = max(1e-9, self.buy_probability - self.hold_probability)
            hold_ratio = (entry_probability - self.hold_probability) / hold_span
            return self.min_entry_weight * min(max(hold_ratio, 0.0), 1.0)
        return self.min_entry_weight + min(max(ratio, 0.0), 1.0) * (self.mid_weight - self.min_entry_weight)


def _sigmoid(value: float) -> float:
    value = max(min(float(value), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-value))


def _safe_number(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default
