#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probabilistic Agent based on Bayesian probability and Markov regimes."""

from __future__ import annotations

import pandas as pd

from .base_agent import AgentSignal, BaseAgent, clamp


class ProbabilisticAgent(BaseAgent):
    def __init__(
        self,
        weight=0.15,
        score_column="prob_score",
        confidence_column="prob_confidence",
        buy_threshold=0.30,
        sell_threshold=-0.25,
        neutral_zone=0.08,
        max_score=1.0,
        enabled=False,
    ):
        super().__init__("ProbabilisticAgent", weight, enabled)
        self.score_column = score_column
        self.confidence_column = confidence_column
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)
        self.neutral_zone = abs(float(neutral_zone))
        self.max_score = abs(float(max_score)) or 1.0

    def run(self, row, context=None):
        raw_score = clamp(_safe_number(row.get(self.score_column, 0), 0), -self.max_score, self.max_score)
        contribution_score = 0.0 if abs(raw_score) < self.neutral_zone else clamp(raw_score)
        confidence = _safe_number(row.get(self.confidence_column, abs(raw_score)), abs(raw_score))
        confidence = max(0.0, min(1.0, confidence))

        if contribution_score >= self.buy_threshold:
            action = "buy"
        elif contribution_score <= self.sell_threshold:
            action = "sell"
        else:
            action = "hold"

        bayes_prob_up = _safe_number(row.get("bayes_prob_up", 0.5), 0.5)
        bayes_score = _safe_number(row.get("bayes_score", 0), 0)
        markov_state = str(row.get("markov_state", "") or "")
        markov_next_state = str(row.get("markov_next_state", "") or "")
        markov_score = _safe_number(row.get("markov_score", 0), 0)
        markov_state_prob = _safe_number(row.get("markov_state_prob", 0), 0)
        model_name = str(row.get("prob_model_name", "") or "")
        reason = (
            f"概率Agent: 模型={model_name or '-'}, Bayes上涨概率={bayes_prob_up:.3f}, "
            f"Bayes分={bayes_score:.3f}, Markov={markov_state}->{markov_next_state}"
            f"({markov_state_prob:.2f}), Markov分={markov_score:.3f}, 有效分={contribution_score:.3f}"
        )
        if abs(raw_score) < self.neutral_zone:
            reason += f"，位于中性区({self.neutral_zone:.2f})"

        return AgentSignal(
            self.name,
            contribution_score,
            confidence,
            action,
            reason,
            {
                "prob_score": raw_score,
                "prob_score_signal": contribution_score,
                "prob_confidence": confidence,
                "prob_action": action,
                "prob_reason": reason,
                "prob_model_name": model_name,
                "bayes_prob_up": bayes_prob_up,
                "bayes_score": bayes_score,
                "markov_state": markov_state,
                "markov_next_state": markov_next_state,
                "markov_state_prob": markov_state_prob,
                "markov_score": markov_score,
            },
        )


def _safe_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default
