#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ML score agent that turns model output into a trade decision score."""

from __future__ import annotations

import pandas as pd

from .base_agent import AgentSignal, BaseAgent, clamp


class MLScoreAgent(BaseAgent):
    def __init__(
        self,
        weight=0.25,
        score_column="ml_score",
        prediction_column="pred_rank_future_return_5d",
        confidence_column="ml_confidence",
        buy_threshold=0.35,
        sell_threshold=-0.25,
        neutral_zone=0.10,
        max_score=1.0,
        enabled=False,
    ):
        super().__init__("MLScoreAgent", weight, enabled)
        self.score_column = score_column
        self.prediction_column = prediction_column
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

        pred_default = row.get("ml_pred_rank_score", row.get("ml_pred_return", row.get("ml_pred_prob", 0)))
        pred = _safe_number(row.get(self.prediction_column, pred_default), 0)
        model_name = str(row.get("ml_model_name", "") or "")
        prediction_kind = _prediction_kind(row, self.prediction_column)
        reason = (
            f"ML评分Agent: 模型={model_name or '-'}, 类型={prediction_kind}, "
            f"预测={pred:.4f}, 原始分={raw_score:.3f}, 有效分={contribution_score:.3f}, "
            f"置信度={confidence:.3f}"
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
                "ml_score": raw_score,
                "ml_score_signal": contribution_score,
                "ml_confidence": confidence,
                "ml_prediction": pred,
                "ml_model_name": model_name,
                "ml_prediction_kind": prediction_kind,
                "ml_score_action": action,
                "ml_score_reason": reason,
            },
        )


def _prediction_kind(row, prediction_column: str) -> str:
    if prediction_column.startswith("pred_prob") or "ml_pred_prob" in row:
        return "上涨概率"
    if "rank" in prediction_column or "ml_pred_rank_score" in row:
        return "横截面排序"
    return "收益预测"


def _safe_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default
