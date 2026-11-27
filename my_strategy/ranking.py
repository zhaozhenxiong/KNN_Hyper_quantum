#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ranking helpers for strategy screening."""

from __future__ import annotations

import pandas as pd


def add_rank_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    weights = config.get("screening", {}).get("rank_weights", {})
    ar_w = float(weights.get("annual_return", 0.40))
    sh_w = float(weights.get("sharpe_ratio", 0.30))
    dd_w = float(weights.get("max_drawdown", 0.20))
    tr_w = float(weights.get("trade_penalty", 0.10))
    min_trades = int(config.get("screening", {}).get("min_trades", 4))

    out["trade_penalty"] = (out["n_trades"] / out["n_trades"].replace(0, 1).max()).fillna(0)
    out["rank_score"] = (
        ar_w * out["annual_return"]
        + sh_w * out["sharpe_ratio"].clip(-2, 2) / 2
        + dd_w * (1 + out["max_drawdown"])
        - tr_w * out["trade_penalty"]
    )
    out.loc[out["n_trades"] < min_trades, "rank_score"] -= 0.25
    out.sort_values(["rank_score", "annual_return"], ascending=False, inplace=True)
    out["rank"] = range(1, len(out) + 1)
    return out

