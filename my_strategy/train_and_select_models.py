#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train candidate models, compare with history, and refresh best-model pointers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.model_selector import refresh_best_models  # noqa: E402
from my_strategy.train_ml_model import resolve_stocks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML/DL models and select the best historical model.")
    parser.add_argument("--stocks", default="local")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--valid-end", default="2025-12-31")
    parser.add_argument("--model-prefix", default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--dl-lookback", type=int, default=20)
    parser.add_argument("--dl-epochs", type=int, default=12)
    parser.add_argument("--dl-batch-size", type=int, default=4096)
    parser.add_argument("--dl-hidden-size", type=int, default=64)
    parser.add_argument("--dl-max-samples", type=int, default=160000)
    parser.add_argument("--no-update", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    args = parse_args()
    stocks, stock_source = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    if not stocks:
        raise RuntimeError("No stocks found for training")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.model_prefix or ("100b" if stock_source == "local_parquet_all" else "custom")
    ml_model = f"lightgbm_cs_rank_5d_{prefix}_{stamp}"
    dl_model = f"dl_lstm_cs_rank_5d_{prefix}_{stamp}"
    common = [
        "--stocks",
        args.stocks,
        "--start",
        args.start,
        "--end",
        args.end,
        "--train-end",
        args.train_end,
        "--valid-end",
        args.valid_end,
    ]
    if args.max_stocks:
        common += ["--max-stocks", str(args.max_stocks)]
    if args.no_update:
        common += ["--no-update"]

    trained: dict[str, str] = {}
    if not args.skip_ml:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "my_strategy" / "train_ml_model.py"),
                *common,
                "--model-type",
                "lightgbm",
                "--model-name",
                ml_model,
                "--label-mode",
                "cross_sectional_rank",
                "--base-label",
                "future_return_5d",
                "--neutral-group",
                "auto",
            ]
        )
        trained["ml"] = ml_model

    if not args.skip_dl:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "my_strategy" / "train_deep_model.py"),
                *common,
                "--model-type",
                "lstm",
                "--lookback",
                str(args.dl_lookback),
                "--epochs",
                str(args.dl_epochs),
                "--batch-size",
                str(args.dl_batch_size),
                "--hidden-size",
                str(args.dl_hidden_size),
                "--max-samples",
                str(args.dl_max_samples),
                "--model-name",
                dl_model,
            ]
        )
        trained["deep_learning"] = dl_model

    registry = refresh_best_models()
    result = {
        "trained": trained,
        "best_models": {
            "ml": (registry.get("ml") or {}).get("model_name", ""),
            "deep_learning": (registry.get("deep_learning") or {}).get("model_name", ""),
        },
        "registry": str(PROJECT_ROOT / "my_strategy" / "data" / "processed" / "model_registry" / "best_models.json"),
    }
    print("\nBest model selection:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
