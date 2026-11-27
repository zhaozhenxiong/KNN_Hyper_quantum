#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimize nonlinear Multi-Agent fusion policy on the local stock universe."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.agents.data_agent import DataAgent  # noqa: E402
from my_strategy.agents.factor_agent import FactorAgent  # noqa: E402
from my_strategy.agents.multi_agent_engine import MultiAgentEngine  # noqa: E402
from my_strategy.deep_learning.predictor import enrich_with_deep_predictions  # noqa: E402
from my_strategy.fusion_registry import save_best_fusion_policy  # noqa: E402
from my_strategy.ml.predictor import enrich_with_ml_predictions  # noqa: E402
from my_strategy.ml.probabilistic import enrich_with_probabilistic_predictions  # noqa: E402
from my_strategy.model_selector import best_model_name, refresh_best_models  # noqa: E402
from my_strategy.rl.registry import best_rl_model_name  # noqa: E402
from my_strategy.run_multi_agent_backtest import calc_metrics, load_config, simulate_trades  # noqa: E402
from my_strategy.train_ml_model import resolve_stocks  # noqa: E402


DEFAULT_WEIGHTS = {
    "trend": 0.32,
    "reversal": 0.18,
    "volume": 0.15,
    "ml": 0.16,
    "dl": 0.10,
    "prob": 0.09,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Agent fusion policy on local A-share data.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "my_strategy" / "configs" / "multi_agent_config.yaml"))
    parser.add_argument("--stocks", default="local")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--tune-max-stocks", type=int, default=300, help="Representative stock count for parameter search; 0 means all.")
    parser.add_argument("--full-test-max-stocks", type=int, default=0, help="Final test stock cap; 0 means all available stocks.")
    parser.add_argument("--skip-full-test", action="store_true", help="Only tune/save the best policy; skip final full test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-model", default="")
    parser.add_argument("--dl-model", default="")
    parser.add_argument("--rl-model", default="")
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--no-dl", action="store_true")
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config))
    config.setdefault("data", {})["start"] = args.start
    config.setdefault("data", {})["end"] = args.end
    config.setdefault("data", {})["auto_update"] = False
    config.setdefault("outputs", {})["root_dir"] = "my_strategy/data/processed/ai_agent_screen"

    agents = config.setdefault("agents", {})
    refresh_best_models()
    ml_model = "" if args.no_ml else (args.ml_model or best_model_name("ml"))
    dl_model = "" if args.no_dl else (args.dl_model or best_model_name("deep_learning"))
    rl_model = "" if args.no_rl else (args.rl_model or best_rl_model_name())

    ml = agents.setdefault("ml", {})
    ml["enabled"] = bool(ml_model)
    ml["model_name"] = ml_model
    agents.setdefault("ml_score", {})["enabled"] = bool(ml_model)

    dl = agents.setdefault("deep_learning", {})
    dl["enabled"] = bool(dl_model)
    dl["model_name"] = dl_model

    rl = agents.setdefault("reinforcement_learning", {})
    rl["enabled"] = bool(rl_model)
    rl["model_name"] = rl_model
    return config


def preload_factors(stocks: list[str], config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    data_agent = DataAgent(config.get("data", {}).get("source", "local_data_center"), auto_update=False)
    load_start = (pd.to_datetime(args.start) - pd.Timedelta(days=int(config.get("data", {}).get("warmup_days", 180)))).strftime("%Y-%m-%d")
    volume_window = config.get("agents", {}).get("volume", {}).get("volume_window", 20)
    factors_by_stock: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    start_ts = time.time()

    for idx, stock in enumerate(stocks, start=1):
        try:
            raw = data_agent.load(stock, load_start, args.end)
            factors = FactorAgent.add_factors(raw, volume_window=volume_window)
            factors = enrich_optional_predictions(factors, config)
            if not factors.empty:
                factors_by_stock[stock] = factors
        except Exception as exc:
            failures.append({"stock": stock, "error": str(exc)[:180]})
        if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(stocks):
            elapsed = max(time.time() - start_ts, 1)
            print(f"preload [{idx}/{len(stocks)}] ok={len(factors_by_stock)} fail={len(failures)} rate={idx/elapsed:.2f}/s", flush=True)
    return factors_by_stock, failures


def enrich_optional_predictions(factors: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    agents = config.setdefault("agents", {})
    ml_cfg = agents.setdefault("ml", {})
    ml_score_cfg = agents.setdefault("ml_score", {})
    if ml_cfg.get("enabled") and ml_cfg.get("model_name"):
        factors = enrich_with_ml_predictions(factors, ml_cfg["model_name"], ml_cfg.get("model_root"))
        meta = factors.attrs.get("ml_metadata", {})
        if meta.get("prediction_column"):
            ml_cfg["prediction_column"] = meta["prediction_column"]
            ml_score_cfg["prediction_column"] = meta["prediction_column"]
        ml_cfg["score_column"] = "ml_score"
        ml_cfg["confidence_column"] = "ml_confidence"
        ml_score_cfg["score_column"] = "ml_score"
        ml_score_cfg["confidence_column"] = "ml_confidence"
        ml_score_cfg["enabled"] = True

    dl_cfg = agents.setdefault("deep_learning", {})
    if dl_cfg.get("enabled") and dl_cfg.get("model_name"):
        factors = enrich_with_deep_predictions(factors, dl_cfg["model_name"], dl_cfg.get("model_root"))
        meta = factors.attrs.get("dl_metadata", {})
        dl_cfg["prediction_column"] = meta.get("prediction_column", "dl_prediction")
        dl_cfg["score_column"] = meta.get("score_column", "dl_score")
        dl_cfg["confidence_column"] = meta.get("confidence_column", "dl_confidence")
        dl_cfg["model_type"] = meta.get("model_type", "")

    prob_cfg = agents.setdefault("probabilistic", {})
    if prob_cfg.get("enabled", True):
        factors = enrich_with_probabilistic_predictions(factors, prob_cfg)
        meta = factors.attrs.get("probabilistic_metadata", {})
        prob_cfg["model_name"] = meta.get("model_name", prob_cfg.get("model_name", ""))
        prob_cfg["model_type"] = meta.get("model_type", "rolling_naive_bayes_markov")
        prob_cfg["prediction_column"] = meta.get("prediction_column", "bayes_prob_up")
        prob_cfg["score_column"] = meta.get("score_column", "prob_score")
        prob_cfg["confidence_column"] = meta.get("confidence_column", "prob_confidence")
    return factors


def candidate_policies(config: dict[str, Any], trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = copy.deepcopy(config.get("decision", {}).get("fusion", {}))
    base.setdefault("enabled", True)
    base.setdefault("weights", DEFAULT_WEIGHTS)
    policies = [base]

    keys = ["trend", "reversal", "volume", "ml", "dl", "prob"]
    prior = np.array([3.2, 1.8, 1.5, 1.6, 1.0, 1.2], dtype=float)
    np_rng = np.random.default_rng(seed)
    for _ in range(max(0, trials - 1)):
        weights = np_rng.dirichlet(prior)
        policy = {
            "enabled": True,
            "activation": rng.choice(["sigmoid", "tanh", "softsign"]),
            "temperature": round(rng.uniform(1.4, 3.2), 3),
            "buy_probability": round(rng.uniform(0.58, 0.68), 3),
            "hold_probability": round(rng.uniform(0.52, 0.57), 3),
            "sell_probability": round(rng.uniform(0.60, 0.72), 3),
            "high_probability": round(rng.uniform(0.70, 0.78), 3),
            "max_probability": round(rng.uniform(0.84, 0.92), 3),
            "min_ml_score": round(rng.uniform(-0.35, -0.10), 3),
            "min_prob_score": round(rng.uniform(-0.25, -0.05), 3),
            "use_confidence": rng.choice([True, True, False]),
            "blocked_markov_states": ["bear", "bear_high_vol"],
            "weights": {key: round(float(value), 4) for key, value in zip(keys, weights)},
        }
        policies.append(policy)
    return policies


def _select_representative_stocks(stocks: list[str], max_stocks: int, seed: int) -> list[str]:
    if max_stocks <= 0 or len(stocks) <= max_stocks:
        return list(stocks)
    rng = random.Random(seed)
    ordered = sorted(stocks)
    buckets: dict[str, list[str]] = {"SH": [], "SZ": [], "BJ": [], "OTHER": []}
    for stock in ordered:
        suffix = stock.split(".")[-1].upper() if "." in stock else "OTHER"
        buckets.setdefault(suffix, []).append(stock)
    selected: list[str] = []
    remaining = max_stocks
    non_empty = {key: value for key, value in buckets.items() if value}
    for key, values in non_empty.items():
        quota = max(1, round(max_stocks * len(values) / len(ordered)))
        quota = min(quota, len(values), remaining)
        selected.extend(rng.sample(values, quota))
        remaining -= quota
    if remaining > 0:
        pool = [stock for stock in ordered if stock not in selected]
        selected.extend(rng.sample(pool, min(remaining, len(pool))))
    return sorted(selected[:max_stocks])


def evaluate_policy(
    factors_by_stock: dict[str, pd.DataFrame],
    base_config: dict[str, Any],
    policy: dict[str, Any],
    start: str,
    end: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config = copy.deepcopy(base_config)
    config.setdefault("decision", {})["fusion"] = copy.deepcopy(policy)
    config.setdefault("data", {})["start"] = start
    config.setdefault("data", {})["end"] = end
    initial_cash = float(config.get("backtest", {}).get("initial_cash", 500000))
    start_ts = time.time()

    for stock, factors in factors_by_stock.items():
        try:
            decisions = MultiAgentEngine(copy.deepcopy(config)).generate_decisions(factors)
            period = (pd.to_datetime(factors["date"]) >= pd.to_datetime(start)) & (pd.to_datetime(factors["date"]) <= pd.to_datetime(end))
            decision_period = (pd.to_datetime(decisions["date"]) >= pd.to_datetime(start)) & (pd.to_datetime(decisions["date"]) <= pd.to_datetime(end))
            period_factors = factors.loc[period].reset_index(drop=True)
            period_decisions = decisions.loc[decision_period].reset_index(drop=True)
            if period_factors.empty or period_decisions.empty:
                continue
            daily, trades = simulate_trades(period_factors, period_decisions, config)
            metrics = calc_metrics(daily.set_index("date")["equity"], initial_cash, trades)
            latest = period_decisions.iloc[-1]
            rows.append(
                {
                    "stock": stock,
                    **metrics,
                    "latest_action": latest.get("final_action", ""),
                    "entry_probability": _safe_float(latest.get("entry_probability")),
                    "exit_probability": _safe_float(latest.get("exit_probability")),
                    "latest_score": _safe_float(latest.get("final_score")),
                    "trade_count": metrics.get("n_trades", 0),
                }
            )
        except Exception:
            continue

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"stocks": 0, "score": -999.0, "elapsed_seconds": round(time.time() - start_ts, 2)}
    for col in ["annual_return", "total_return", "sharpe_ratio", "max_drawdown", "win_rate", "trade_count"]:
        frame[col] = pd.to_numeric(frame.get(col, 0), errors="coerce").fillna(0.0)

    avg_annual = float(frame["annual_return"].mean())
    median_annual = float(frame["annual_return"].median())
    avg_sharpe = float(frame["sharpe_ratio"].replace([np.inf, -np.inf], 0).mean())
    worst_drawdown = float(frame["max_drawdown"].min())
    avg_drawdown = float(frame["max_drawdown"].mean())
    win_rate = float(frame["win_rate"].mean())
    positive_rate = float((frame["total_return"] > 0).mean())
    turnover_penalty = float((frame["trade_count"].mean() / 30.0))
    score = (
        0.30 * avg_annual
        + 0.18 * median_annual
        + 0.22 * avg_sharpe
        + 0.16 * avg_drawdown
        + 0.10 * positive_rate
        + 0.08 * win_rate
        - 0.06 * turnover_penalty
        + 0.10 * worst_drawdown
    )
    return {
        "stocks": int(len(frame)),
        "score": float(score),
        "avg_annual_return": avg_annual,
        "median_annual_return": median_annual,
        "avg_total_return": float(frame["total_return"].mean()),
        "avg_sharpe": avg_sharpe,
        "avg_drawdown": avg_drawdown,
        "worst_drawdown": worst_drawdown,
        "avg_win_rate": win_rate,
        "positive_rate": positive_rate,
        "avg_trade_count": float(frame["trade_count"].mean()),
        "elapsed_seconds": round(time.time() - start_ts, 2),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    stocks, source = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    tune_stocks = _select_representative_stocks(stocks, args.tune_max_stocks, args.seed)
    full_test_stocks = _select_representative_stocks(stocks, args.full_test_max_stocks, args.seed + 17)
    config = prepare_config(args)
    print("=" * 72)
    print("KHQuant fusion policy optimization")
    print(f"stocks={len(stocks)} source={source} tune_stocks={len(tune_stocks)} full_test_stocks={len(full_test_stocks)} trials={args.trials}")
    print(f"preload={args.start}~{args.end} valid={args.valid_start}~{pd.to_datetime(args.test_start) - pd.Timedelta(days=1):%Y-%m-%d} test={args.test_start}~{args.end}")
    print("=" * 72)

    factors_by_stock, failures = preload_factors(tune_stocks, config, args)
    policies = candidate_policies(config, args.trials, args.seed)
    valid_end = (pd.to_datetime(args.test_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []

    for idx, policy in enumerate(policies, start=1):
        metrics = evaluate_policy(factors_by_stock, config, policy, args.valid_start, valid_end)
        row = {"trial": idx, **metrics, "policy": json.dumps(policy, ensure_ascii=False)}
        rows.append(row)
        print(
            f"trial {idx}/{len(policies)} score={metrics['score']:.4f} stocks={metrics['stocks']} "
            f"annual={metrics.get('avg_annual_return', 0):+.2%} dd={metrics.get('worst_drawdown', 0):+.2%} "
            f"sharpe={metrics.get('avg_sharpe', 0):.2f}",
            flush=True,
        )

    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best_policy = json.loads(ranking.iloc[0]["policy"])
    preliminary_metrics = {"valid": ranking.iloc[0].drop(labels=["policy"]).to_dict(), "test": {"status": "pending"}}
    best_path = save_best_fusion_policy(best_policy, preliminary_metrics)
    print(f"Preliminary best fusion policy saved: {best_path}", flush=True)
    if args.skip_full_test:
        test_metrics = {}
        full_failures: list[dict[str, str]] = []
    else:
        full_factors_by_stock, full_failures = preload_factors(full_test_stocks, config, args)
        test_metrics = evaluate_policy(full_factors_by_stock, config, best_policy, args.test_start, args.end)
    best_metrics = {"valid": ranking.iloc[0].drop(labels=["policy"]).to_dict(), "test": test_metrics}
    best_path = save_best_fusion_policy(best_policy, best_metrics)

    report_root = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_path = report_root / f"fusion_optimization_{stamp}.csv"
    report_path = report_root / f"fusion_optimization_{stamp}.json"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stocks_requested": len(stocks),
        "tune_stocks": len(tune_stocks),
        "tune_stocks_preloaded": len(factors_by_stock),
        "full_test_stocks": len(full_test_stocks),
        "failures": failures[:200],
        "full_test_failures": full_failures[:200],
        "trials": args.trials,
        "valid_start": args.valid_start,
        "valid_end": valid_end,
        "test_start": args.test_start,
        "test_end": args.end,
        "best_policy_path": str(best_path),
        "ranking_path": str(ranking_path),
        "best_policy": best_policy,
        "best_metrics": best_metrics,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBest fusion policy saved:")
    print(best_path)
    print(json.dumps({"best_policy": best_policy, "best_metrics": best_metrics, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
