#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Machine-learning prediction agent."""

from __future__ import annotations

import pandas as pd

from .base_agent import AgentSignal, BaseAgent, clamp


class MLPredictionAgent(BaseAgent):
    def __init__(
        self,
        weight=0.25,
        score_column="ml_score",
        prediction_column="pred_return_5d",
        confidence_column="ml_confidence",
        enabled=False,
    ):
        super().__init__("MLPredictionAgent", weight, enabled)
        self.score_column = score_column
        self.prediction_column = prediction_column
        self.confidence_column = confidence_column

    def run(self, row, context=None):
        pred_default = row.get("ml_pred_rank_score", row.get("ml_pred_return", 0))
        score = _safe_number(row.get(self.score_column, 0), 0)
        pred = _safe_number(row.get(self.prediction_column, pred_default), 0)
        confidence = _safe_number(row.get(self.confidence_column, abs(score)), abs(score))
        score = clamp(score)
        confidence = max(0.0, min(1.0, confidence))
        if score >= 0.35:
            action = "buy"
        elif score <= -0.25:
            action = "sell"
        else:
            action = "hold"
        model_name = str(row.get("ml_model_name", "") or "")
        if self.prediction_column.startswith("pred_prob"):
            reason = f"ML模型{model_name} 预测上涨概率={pred:.2%}, score={score:.3f}"
        elif "rank" in self.prediction_column or "ml_pred_rank_score" in row:
            reason = f"ML模型{model_name} 横截面排序分数={pred:.3f}, score={score:.3f}"
        else:
            reason = f"ML模型{model_name} 预测收益={pred:.2%}, score={score:.3f}"
        return AgentSignal(
            self.name,
            score,
            confidence,
            action,
            reason,
            {
                "ml_score": score,
                "ml_confidence": confidence,
                "ml_prediction": pred,
                "ml_model_name": model_name,
            },
        )


def _safe_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default
