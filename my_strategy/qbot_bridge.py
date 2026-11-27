#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export latest backtest action into a simple Qbot-friendly signal file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def export_signal(daily_csv: Path, stock: str | None, output_dir: Path) -> Path:
    df = pd.read_csv(daily_csv)
    if df.empty:
        raise ValueError(f"No rows in {daily_csv}")

    row = df.iloc[-1]
    action_value = int(row.get("trade_action", row.get("signal", 0)))
    action = "BUY" if action_value == 1 else "SELL" if action_value == -1 else "HOLD"
    payload = {
        "stock": stock or daily_csv.name.split("_")[0],
        "date": str(row["date"])[:10],
        "action": action,
        "signal": action_value,
        "price": float(row["price"]),
        "nav": float(row["nav"]),
        "position": int(row.get("position", 0)),
        "source_file": str(daily_csv.resolve()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{payload['stock'].replace('.', '_')}_{payload['date']}_signal.json"
    csv_path = output_dir / f"{payload['stock'].replace('.', '_')}_{payload['date']}_signal.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([payload]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Qbot signal JSON: {json_path}")
    print(f"Qbot signal CSV:  {csv_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Export latest daily backtest row as a Qbot signal.")
    parser.add_argument("--daily", required=True, help="Path to a daily CSV from results/daily")
    parser.add_argument("--stock", help="Stock code, e.g. 002436.SZ")
    parser.add_argument("--output-dir", default="results/qbot_signals", help="Signal output folder")
    args = parser.parse_args()
    export_signal(Path(args.daily), args.stock, Path(args.output_dir))


if __name__ == "__main__":
    main()

