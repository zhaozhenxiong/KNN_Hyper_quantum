#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily incremental update entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.local_cache import LocalCache
from my_strategy.data_manager.stock_pool import StockPoolManager
from my_strategy.data_manager.updater import DailyUpdater


def _quote_symbol(code: str) -> str:
    raw = str(code).split(".")[0]
    return ("sh" if str(code).upper().endswith(".SH") else "sz") + raw


def _default_for_column(df: pd.DataFrame, column: str):
    dtype = df[column].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return 0
    if pd.api.types.is_float_dtype(dtype):
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.NaT
    return ""


def _cast_like(df: pd.DataFrame, column: str, value):
    dtype = df[column].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return int(float(value)) if value not in ("", None) else 0
    if pd.api.types.is_float_dtype(dtype):
        return float(value) if value not in ("", None) else 0.0
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(value)
    return "" if value is None else str(value)


def _write_realtime_row(cache: LocalCache, code: str, values: dict) -> bool:
    path = cache.stock_daily_path(code)
    if not path.exists():
        return False
    old = pd.read_parquet(path)
    if old.empty or "date" not in old.columns:
        return False
    old["date"] = pd.to_datetime(old["date"])
    row = {col: _default_for_column(old, col) for col in old.columns}
    for column, value in values.items():
        if column in row:
            row[column] = _cast_like(old, column, value)
    new_row = pd.DataFrame([row], columns=old.columns)
    out = pd.concat([old, new_row], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    tmp = path.with_name(path.stem + ".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return True


def update_today_realtime(config: dict, stocks: list[str] | None, end_date: str | None, batch_size: int = 30) -> pd.DataFrame:
    """Append today's valid realtime OHLC row to local parquet cache.

    The updater refuses quotes with zero OHLC or volume, so suspended/no-trade
    stocks keep their previous cache unchanged.
    """
    target_date = end_date or datetime.now().strftime("%Y-%m-%d")
    target_stamp = target_date.replace("-", "")
    cache = LocalCache(config)
    if stocks:
        pool = StockPoolManager(config).build_pool(stocks=stocks)
    else:
        paths = sorted(cache.stock_daily_dir.glob("*.parquet"))
        pool = pd.DataFrame({"stock": [path.stem for path in paths]})

    rows = []
    codes = pool["stock"].astype(str).drop_duplicates().tolist()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for start in range(0, len(codes), batch_size):
        batch = codes[start : start + batch_size]
        url = "https://qt.gtimg.cn/q=" + ",".join(_quote_symbol(code) for code in batch)
        try:
            result = subprocess.run(
                ["curl.exe", "-s", "--noproxy", "*", "--max-time", "20", url],
                capture_output=True,
                timeout=25,
            )
            text = result.stdout.decode("gbk", errors="replace")
        except Exception as exc:
            for code in batch:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": str(exc)})
            continue

        seen: set[str] = set()
        for line in text.splitlines():
            if "=" not in line or "~" not in line:
                continue
            parts = line.split("~")
            if len(parts) < 35:
                continue
            raw = parts[2].strip()
            if not raw.isdigit():
                continue
            code = raw + (".SH" if raw.startswith("6") else ".SZ")
            if code not in batch:
                continue
            seen.add(code)
            stamp = parts[30].strip() if len(parts) > 30 else ""
            if not stamp.startswith(target_stamp):
                rows.append({"code": code, "last_date": "", "status": "skipped", "error_msg": f"quote date {stamp[:8]} != {target_stamp}"})
                continue
            try:
                close = float(parts[3]) if parts[3] else 0.0
                open_price = float(parts[5]) if parts[5] else 0.0
                high = float(parts[33]) if parts[33] else 0.0
                low = float(parts[34]) if parts[34] else 0.0
                volume = float(parts[6].replace(",", "")) if parts[6] else 0.0
                pct_chg = float(parts[32]) if len(parts) > 32 and parts[32] else 0.0
            except Exception as exc:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": f"bad quote: {exc}"})
                continue
            if min(open_price, high, low, close) <= 0 or high < low or volume <= 0:
                rows.append({"code": code, "last_date": "", "status": "skipped", "error_msg": "zero OHLC or volume"})
                continue

            values = {
                "date": pd.to_datetime(target_date),
                "code": code,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "trade_open": open_price,
                "trade_high": high,
                "trade_low": low,
                "trade_close": close,
                "volume": volume * 100,
                "amount": 0.0,
                "pct_chg": pct_chg,
                "turnover": 0.0,
                "source": "tencent_realtime",
                "updated_at": now_text,
            }
            ok = _write_realtime_row(cache, code, values)
            rows.append({"code": code, "last_date": target_date if ok else "", "status": "success" if ok else "failed", "error_msg": "" if ok else "cache missing/invalid"})

        for code in batch:
            if code not in seen:
                rows.append({"code": code, "last_date": "", "status": "failed", "error_msg": "quote missing"})
        if start + batch_size < len(codes):
            time.sleep(0.2)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update KHQuant local daily data cache.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "my_strategy" / "configs" / "data_config.yaml"))
    parser.add_argument("--mode", default=None, choices=[None, "custom", "all_a"], nargs="?")
    parser.add_argument("--stocks", default="", help="Comma-separated stock codes.")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--today-realtime", action="store_true", help="Append valid realtime OHLC for --end-date/today into local parquet cache.")
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    config = load_data_config(args.config)
    stocks = [item.strip() for item in args.stocks.split(",") if item.strip()] or None
    if args.today_realtime:
        report = update_today_realtime(config, stocks=stocks, end_date=args.end_date, batch_size=args.batch_size)
    else:
        updater = DailyUpdater(config)
        report = updater.update_pool(mode=args.mode, stocks=stocks, end_date=args.end_date)
    if report.empty:
        print("No stocks to update.")
    else:
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
