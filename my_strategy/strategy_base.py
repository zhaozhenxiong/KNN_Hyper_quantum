#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base classes for screening strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategySpec:
    name: str
    family: str
    source: str
    description: str


class BaseLabStrategy:
    spec = StrategySpec("base", "base", "local", "base strategy")

    def generate(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        raise NotImplementedError


def make_output(df: pd.DataFrame, name: str, score: pd.Series, target_weight: pd.Series, reason: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": df["date"],
            "strategy_name": name,
            "close": df["close"],
            "score": score.fillna(0).clip(-1, 1),
            "target_weight": target_weight.fillna(0).clip(0, 1),
            "reason": reason,
        }
    )
    out["final_action"] = "watch"
    out.loc[out["target_weight"] > 0, "final_action"] = "hold"
    out.loc[(out["target_weight"] > 0) & (out["target_weight"].shift(1).fillna(0) == 0), "final_action"] = "buy"
    out.loc[(out["target_weight"] == 0) & (out["target_weight"].shift(1).fillna(0) > 0), "final_action"] = "sell"
    return out

