#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Portfolio sizing agent."""

from __future__ import annotations


class PortfolioAgent:
    def __init__(
        self,
        min_entry_weight=0.30,
        mid_weight=0.50,
        high_weight=0.70,
        max_weight=0.90,
        reduce_weight=0.30,
        hold_threshold=0.15,
        buy_threshold=0.55,
        high_threshold=0.75,
    ):
        self.min_entry_weight = float(min_entry_weight)
        self.mid_weight = float(mid_weight)
        self.high_weight = float(high_weight)
        self.max_weight = float(max_weight)
        self.reduce_weight = float(reduce_weight)
        self.hold_threshold = float(hold_threshold)
        self.buy_threshold = float(buy_threshold)
        self.high_threshold = float(high_threshold)

    def calc_target_weight(self, final_score: float, risk_action: str) -> float:
        if risk_action == "force_sell":
            return 0.0
        if risk_action == "reduce":
            return min(self.reduce_weight, self.max_weight)
        if final_score <= self.hold_threshold:
            return 0.0
        if final_score >= self.high_threshold:
            span = max(1e-9, 1.0 - self.high_threshold)
            ratio = min(1.0, max(0.0, (final_score - self.high_threshold) / span))
            return self.high_weight + ratio * (self.max_weight - self.high_weight)
        if final_score >= self.buy_threshold:
            span = max(1e-9, self.high_threshold - self.buy_threshold)
            ratio = min(1.0, max(0.0, (final_score - self.buy_threshold) / span))
            return self.mid_weight + ratio * (self.high_weight - self.mid_weight)

        span = max(1e-9, self.buy_threshold - self.hold_threshold)
        ratio = min(1.0, max(0.0, (final_score - self.hold_threshold) / span))
        return self.min_entry_weight + ratio * (self.mid_weight - self.min_entry_weight)
