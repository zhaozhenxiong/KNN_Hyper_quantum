#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning strategy score agent."""

from __future__ import annotations

import pandas as pd

from .base_agent import AgentSignal, BaseAgent, clamp


class DeepLearningAgent(BaseAgent):
    def __init__(
        self,
        weight=0.20,
        score_column="dl_score",
        prediction_column="dl_prediction",
        confidence_column="dl_confidence",
        buy_threshold=0.35,
        sell_threshold=-0.25,
        neutral_zone=0.10,
        max_score=1.0,
        enabled=False,
    ):
        super().__init__("DeepLearningAgent", weight, enabled)
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

        pred = _safe_number(row.get(self.prediction_column, row.get("dl_prediction", 0)), 0)
        model_name = str(row.get("dl_model_name", "") or "")
        model_type = str(row.get("dl_model_type", "") or "")
        kind = str(row.get("dl_prediction_kind", "横截面排序") or "横截面排序")
        reason = (
            f"深度学习Agent: 模型={model_name or '-'}, 类型={model_type or '-'}, "
            f"预测类型={kind}, 预测={pred:.4f}, 原始分={raw_score:.3f}, "
            f"有效分={contribution_score:.3f}, 置信度={confidence:.3f}"
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
                "dl_score": raw_score,
                "dl_score_signal": contribution_score,
                "dl_confidence": confidence,
                "dl_prediction": pred,
                "dl_model_name": model_name,
                "dl_model_type": model_type,
                "dl_prediction_kind": kind,
                "dl_action": action,
                "dl_reason": reason,
            },
        )


def _safe_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default
