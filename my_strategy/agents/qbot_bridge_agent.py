#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export multi-agent decisions to a simple Qbot-friendly table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class QbotBridgeAgent:
    def export_to_qbot_format(self, signals: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qbot_df = signals.copy()
        qbot_df = qbot_df.rename(columns={"final_action": "signal", "target_weight": "weight"})
        keep_cols = ["date", "stock", "signal", "weight", "final_score", "risk_action", "reason_json"]
        keep_cols = [col for col in keep_cols if col in qbot_df.columns]
        qbot_df = qbot_df[keep_cols]
        qbot_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return qbot_df

