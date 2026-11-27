#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build local stock metadata with optional industry information.

The ML trainer reads ``my_strategy/data/metadata/stock_metadata.csv`` when it
exists. If AkShare industry endpoints are reachable, this script enriches local
stocks with Eastmoney industry names; otherwise it still writes a usable board
fallback so cross-sectional training can run locally.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.train_ml_model import list_local_stocks, market_board, normalize_stock_code  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "my_strategy" / "data" / "metadata" / "stock_metadata.csv"
UNIVERSE_PATH = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "reports" / "ranking.csv"


def base_metadata() -> pd.DataFrame:
    stocks = list_local_stocks()
    out = pd.DataFrame({"stock": stocks})
    out["stock"] = out["stock"].map(normalize_stock_code)
    out["board"] = out["stock"].map(market_board)
    if UNIVERSE_PATH.exists():
        try:
            universe = pd.read_csv(UNIVERSE_PATH)
            if "stock" in universe.columns:
                universe = universe.copy()
                universe["stock"] = universe["stock"].map(normalize_stock_code)
                extra_cols = [col for col in ["stock_name", "total_market_cap", "turnover"] if col in universe.columns]
                out = out.merge(universe[["stock", *extra_cols]].drop_duplicates("stock"), on="stock", how="left")
        except Exception as exc:
            print(f"warning: failed to merge ai_agent_screen ranking.csv: {exc}")
    return out


def fetch_akshare_industry() -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as exc:
        print(f"warning: akshare unavailable: {exc}")
        return pd.DataFrame(columns=["stock", "industry"])

    try:
        boards = ak.stock_board_industry_name_em()
    except Exception as exc:
        print(f"warning: failed to fetch industry board list: {exc}")
        return pd.DataFrame(columns=["stock", "industry"])

    if boards.empty:
        return pd.DataFrame(columns=["stock", "industry"])

    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    rows = []
    for idx, industry in enumerate(boards[name_col].dropna().astype(str).unique(), start=1):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception as exc:
            print(f"warning: failed industry {industry}: {exc}")
            continue
        code_col = "代码" if "代码" in cons.columns else ("code" if "code" in cons.columns else None)
        name_col2 = "名称" if "名称" in cons.columns else ("stock_name" if "stock_name" in cons.columns else None)
        if not code_col:
            continue
        part = pd.DataFrame({"stock": cons[code_col].astype(str).map(normalize_stock_code), "industry": industry})
        if name_col2:
            part["stock_name_industry_source"] = cons[name_col2].astype(str)
        rows.append(part)
        if idx % 20 == 0:
            print(f"industry progress: {idx}/{len(boards)}")
    if not rows:
        return pd.DataFrame(columns=["stock", "industry"])
    industry_df = pd.concat(rows, ignore_index=True).drop_duplicates("stock", keep="first")
    return industry_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    meta = base_metadata()
    industry = fetch_akshare_industry()
    if not industry.empty:
        meta = meta.merge(industry, on="stock", how="left")
        meta["neutral_group"] = meta["industry"].fillna(meta["board"])
    else:
        meta["industry"] = ""
        meta["neutral_group"] = meta["board"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT_PATH}")
    print(f"stocks: {len(meta)}, industry filled: {int(meta['industry'].astype(str).str.len().gt(0).sum())}")


if __name__ == "__main__":
    main()
