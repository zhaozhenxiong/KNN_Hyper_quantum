#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading agent backed by KHQuant Local Data Center."""

from __future__ import annotations

import pandas as pd

from my_strategy.data_manager import DataCenter


class DataAgent:
    """Load daily bars through the local data center only.

    The data center handles local parquet reads, incremental updates, fallback
    sources, and failed-update protection.
    """

    def __init__(self, source: str = "local_data_center", auto_update: bool = True):
        self.source = source
        self.auto_update = auto_update
        self.data_center = DataCenter()

    def load(self, stock_code: str, start: str, end: str) -> pd.DataFrame:
        df = self.data_center.get_stock_daily(
            code=stock_code,
            start_date=start,
            end_date=end,
            auto_update=self.auto_update,
        )
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{stock_code} 数据缺少必要列: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        df = _dedupe_daily_rows(df)
        df = df.sort_values("date").reset_index(drop=True)
        return df


def _dedupe_daily_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated trade dates, preferring rows that carry real trade prices."""
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    trade_close = pd.to_numeric(out.get("trade_close"), errors="coerce") if "trade_close" in out.columns else pd.Series(False, index=out.index)
    out["_date_key"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["_has_trade_price"] = (trade_close > 0).fillna(False).astype(int)
    out.sort_values(["_date_key", "_has_trade_price"], ascending=[True, False], inplace=True)
    out = out.drop_duplicates(subset=["_date_key"], keep="first")
    out.drop(columns=["_date_key", "_has_trade_price"], inplace=True, errors="ignore")
    return out
