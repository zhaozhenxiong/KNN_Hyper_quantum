#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimize fusion policy from existing AI-Agent daily result CSVs only."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.fusion_registry import save_best_fusion_policy  # noqa: E402

DEFAULT_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen"


AGENT_COLUMNS = {
    "rule": "rule_score",
    "trend": "trend_score",
    "reversal": "reversal_score",
    "volume": "volume_score",
    "ml": "ml_score_signal",
    "dl": "dl_score_signal",
    "prob": "prob_score_signal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize fusion policy from existing daily CSV outputs.")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--valid-end", default="2026-05-31")
    parser.add_argument("--test-start", default="2026-06-01")
    parser.add_argument("--initial-cash", type=float, default=500000)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--min-weight-change", type=float, default=0.08)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--parse-agent-json", action="store_true", help="Parse reason_json into Trend/Reversal/Volume scores; slower.")
    return parser.parse_args()


def load_existing_daily(root: Path, max_stocks: int, seed: int, parse_agent_json: bool = False) -> dict[str, pd.DataFrame]:
    daily_dir = root / "daily"
    paths = sorted(daily_dir.glob("*_daily.csv"))
    if max_stocks > 0 and len(paths) > max_stocks:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, max_stocks))
    out: dict[str, pd.DataFrame] = {}
    for path in paths:
        try:
            if parse_agent_json:
                df = pd.read_csv(path)
            else:
                usecols = [
                    "date",
                    "close",
                    "final_score",
                    "ml_score_signal",
                    "dl_score_signal",
                    "prob_score_signal",
                    "prob_score",
                ]
                df = pd.read_csv(path, usecols=lambda col: col in usecols)
        except Exception:
            continue
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        stock = _stock_from_daily_name(path.name)
        df = _prepare_daily(df, parse_agent_json=parse_agent_json)
        if len(df) >= 30:
            out[stock] = df
    return out


def _stock_from_daily_name(name: str) -> str:
    stem = name.replace("_daily.csv", "")
    parts = stem.split("_")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else stem


def _prepare_daily(df: pd.DataFrame, parse_agent_json: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])
    out["return_1d"] = out["close"].pct_change().fillna(0.0)
    for col in ["ml_score_signal", "dl_score_signal", "prob_score_signal", "prob_score"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "prob_score_signal" not in out.columns or out["prob_score_signal"].abs().sum() == 0:
        out["prob_score_signal"] = pd.to_numeric(out.get("prob_score", 0.0), errors="coerce").fillna(0.0)
    out["rule_score"] = pd.to_numeric(out.get("final_score", 0.0), errors="coerce").fillna(0.0).clip(-1, 1)
    if parse_agent_json and "reason_json" in out.columns:
        parsed = out["reason_json"].apply(_parse_reason_scores)
        for key in ["trend", "reversal", "volume"]:
            out[f"{key}_score"] = parsed.apply(lambda item, k=key: float(item.get(k, 0.0)))
            out[f"{key}_confidence"] = parsed.apply(lambda item, k=key: float(item.get(f"{k}_confidence", abs(item.get(k, 0.0)))))
    else:
        out["trend_score"] = out["rule_score"]
        out["reversal_score"] = 0.0
        out["volume_score"] = 0.0
    return out


def _parse_reason_scores(text: Any) -> dict[str, float]:
    try:
        data = json.loads(text) if isinstance(text, str) and text.strip() else {}
    except Exception:
        return {}
    mapping = {
        "TrendAgent": "trend",
        "ReversalAgent": "reversal",
        "VolumeAgent": "volume",
    }
    out: dict[str, float] = {}
    for agent, key in mapping.items():
        item = data.get(agent) or {}
        try:
            out[key] = float(item.get("score", 0.0))
            out[f"{key}_confidence"] = float(item.get("confidence", abs(out[key])))
        except Exception:
            out[key] = 0.0
            out[f"{key}_confidence"] = 0.0
    return out


def generate_policies(trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    policies = [
        {
            "enabled": True,
            "activation": "sigmoid",
            "temperature": 2.2,
            "buy_probability": 0.62,
            "hold_probability": 0.54,
            "sell_probability": 0.65,
            "high_probability": 0.72,
            "max_probability": 0.86,
            "min_ml_score": -0.25,
            "min_prob_score": -0.15,
            "blocked_markov_states": ["bear", "bear_high_vol"],
            "use_confidence": True,
            "weights": {"rule": 0.65, "ml": 0.16, "dl": 0.10, "prob": 0.09},
        }
    ]
    keys = ["rule", "ml", "dl", "prob"]
    prior = np.array([5.0, 1.8, 1.2, 0.8], dtype=float)
    for _ in range(max(0, trials - 1)):
        weights = np_rng.dirichlet(prior)
        buy = rng.uniform(0.54, 0.66)
        policy = {
            "enabled": True,
            "activation": rng.choice(["sigmoid", "tanh", "softsign"]),
            "temperature": round(rng.uniform(1.2, 3.4), 3),
            "buy_probability": round(buy, 3),
            "hold_probability": round(max(0.50, buy - rng.uniform(0.04, 0.10)), 3),
            "sell_probability": round(rng.uniform(0.57, 0.72), 3),
            "high_probability": round(rng.uniform(0.68, 0.78), 3),
            "max_probability": round(rng.uniform(0.82, 0.92), 3),
            "min_ml_score": round(rng.uniform(-0.38, -0.05), 3),
            "min_prob_score": round(rng.uniform(-0.30, 0.02), 3),
            "blocked_markov_states": ["bear", "bear_high_vol"],
            "use_confidence": rng.choice([True, True, False]),
            "weights": {key: round(float(value), 4) for key, value in zip(keys, weights)},
        }
        policies.append(policy)
    return policies


def evaluate_policy(data: dict[str, pd.DataFrame], policy: dict[str, Any], start: str | None, end: str | None, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for stock, df in data.items():
        part = df
        if start:
            part = part[part["date"] >= pd.to_datetime(start)]
        if end:
            part = part[part["date"] <= pd.to_datetime(end)]
        if len(part) < 20:
            continue
        daily = simulate_policy(part.reset_index(drop=True), policy, args)
        metrics = calc_metrics(daily["equity"], args.initial_cash, int(daily["trade_flag"].sum()))
        latest = daily.iloc[-1]
        rows.append(
            {
                "stock": stock,
                **metrics,
                "latest_action": latest.get("fusion_action", ""),
                "entry_probability": latest.get("entry_probability", 0.0),
                "target_weight": latest.get("target_weight", 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"stocks": 0, "score": -999.0}
    for col in ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "trade_count"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    score = (
        0.30 * frame["annual_return"].mean()
        + 0.22 * frame["sharpe_ratio"].replace([np.inf, -np.inf], 0).mean()
        + 0.18 * frame["max_drawdown"].mean()
        + 0.12 * (frame["total_return"] > 0).mean()
        + 0.08 * frame["total_return"].median()
        - 0.06 * (frame["trade_count"].mean() / 30.0)
        + 0.10 * frame["max_drawdown"].min()
    )
    avg_trade_count = float(frame["trade_count"].mean())
    active_rate = float((frame["trade_count"] > 0).mean())
    if active_rate < 0.10:
        score -= 0.08
    if avg_trade_count < 0.25:
        score -= 0.04
    return {
        "stocks": int(len(frame)),
        "score": float(score),
        "avg_total_return": float(frame["total_return"].mean()),
        "median_total_return": float(frame["total_return"].median()),
        "avg_annual_return": float(frame["annual_return"].mean()),
        "avg_sharpe": float(frame["sharpe_ratio"].replace([np.inf, -np.inf], 0).mean()),
        "avg_drawdown": float(frame["max_drawdown"].mean()),
        "worst_drawdown": float(frame["max_drawdown"].min()),
        "positive_rate": float((frame["total_return"] > 0).mean()),
        "avg_trade_count": avg_trade_count,
        "active_rate": active_rate,
        "buy_or_hold_rate": float(frame["latest_action"].isin(["buy", "hold"]).mean()),
    }


def simulate_policy(df: pd.DataFrame, policy: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    out = df[["date", "close", "return_1d"]].copy()
    weights = policy.get("weights") or {}
    raw = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for key, col in AGENT_COLUMNS.items():
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        raw += pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).clip(-1, 1) * weight
        total_weight += weight
    raw = raw / total_weight if total_weight else raw
    confidence = raw.abs().clip(0, 1)
    adjusted = raw * confidence if policy.get("use_confidence", True) else raw
    temp = float(policy.get("temperature", 2.2))
    entry = adjusted.apply(lambda x: _sigmoid(temp * x))
    exitp = adjusted.apply(lambda x: _sigmoid(-temp * x))
    score = _activate(adjusted, str(policy.get("activation", "sigmoid")), temp)

    buy_prob = float(policy.get("buy_probability", 0.62))
    hold_prob = float(policy.get("hold_probability", 0.54))
    sell_prob = float(policy.get("sell_probability", 0.65))
    min_ml = float(policy.get("min_ml_score", -0.25))
    min_prob = float(policy.get("min_prob_score", -0.15))

    position = 0.0
    cash = args.initial_cash
    shares_value = 0.0
    equity = args.initial_cash
    last_trade_idx = -10_000
    equities = []
    positions = []
    actions = []
    trade_flags = []
    target_weights = []

    for idx, row in df.iterrows():
        if idx > 0:
            shares_value *= 1.0 + float(row.get("return_1d", 0.0))
        equity = cash + shares_value
        can_trade = idx - last_trade_idx >= args.rebalance_days
        allow_entry = pd.to_numeric(row.get("ml_score_signal", 0.0), errors="coerce") >= min_ml
        allow_entry = bool(allow_entry and pd.to_numeric(row.get("prob_score_signal", 0.0), errors="coerce") >= min_prob)
        action = "watch"
        target = position
        if exitp.iloc[idx] >= sell_prob:
            action = "sell"
            target = 0.0
        elif entry.iloc[idx] >= buy_prob and allow_entry:
            action = "buy"
            target = _target_weight(entry.iloc[idx], policy)
        elif entry.iloc[idx] >= hold_prob and allow_entry and position > 0:
            action = "hold"
            target = max(position, min(_target_weight(entry.iloc[idx], policy), position + 0.15))
        elif position > 0:
            action = "hold"
            target = position

        trade_flag = 0
        if can_trade and abs(target - position) >= args.min_weight_change:
            trade_value = equity * (target - position)
            cost = abs(trade_value) * (args.commission + args.slippage)
            if target > position:
                cash -= trade_value + cost
                shares_value += trade_value
            else:
                cash += -trade_value - cost
                shares_value += trade_value
            position = max(0.0, min(0.95, target))
            last_trade_idx = idx
            trade_flag = 1
            equity = cash + shares_value
        equities.append(equity)
        positions.append(position)
        actions.append(action)
        trade_flags.append(trade_flag)
        target_weights.append(target)

    out["agent_weighted_score"] = raw
    out["fusion_score"] = score
    out["entry_probability"] = entry
    out["exit_probability"] = exitp
    out["fusion_action"] = actions
    out["target_weight"] = target_weights
    out["position_weight"] = positions
    out["trade_flag"] = trade_flags
    out["equity"] = equities
    return out


def _activate(series: pd.Series, activation: str, temp: float) -> pd.Series:
    if activation == "tanh":
        return np.tanh(temp * series)
    if activation == "softsign":
        return series / (1.0 + series.abs())
    if activation == "linear":
        return series.clip(-1, 1)
    return series.apply(lambda x: _sigmoid(temp * x) * 2.0 - 1.0)


def _target_weight(entry_probability: float, policy: dict[str, Any]) -> float:
    buy = float(policy.get("buy_probability", 0.62))
    high = float(policy.get("high_probability", 0.72))
    maxp = float(policy.get("max_probability", 0.86))
    if entry_probability >= maxp:
        return 0.90
    if entry_probability >= high:
        return 0.70 + min(1.0, (entry_probability - high) / max(maxp - high, 1e-9)) * 0.20
    return 0.30 + min(1.0, max(0.0, (entry_probability - buy) / max(high - buy, 1e-9))) * 0.20


def _sigmoid(value: float) -> float:
    value = max(min(float(value), 60), -60)
    return 1.0 / (1.0 + math.exp(-value))


def calc_metrics(equity: pd.Series, initial_cash: float, trade_count: int) -> dict[str, float]:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "trade_count": 0}
    total_return = float(eq.iloc[-1] / initial_cash - 1.0)
    days = max(len(eq), 1)
    annual_return = float((1 + total_return) ** (252 / days) - 1) if total_return > -0.99 else -1.0
    returns = eq.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 2 and returns.std() > 1e-12 else 0.0
    drawdown = eq / eq.cummax() - 1.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
        "trade_count": int(trade_count),
    }


def to_engine_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    weights = dict(out.get("weights") or {})
    if "rule" in weights:
        rule_weight = float(weights.pop("rule") or 0.0)
        weights["trend"] = weights.get("trend", 0.0) + rule_weight * 0.45
        weights["reversal"] = weights.get("reversal", 0.0) + rule_weight * 0.30
        weights["volume"] = weights.get("volume", 0.0) + rule_weight * 0.25
    total = sum(float(v) for v in weights.values() if float(v) > 0)
    if total > 0:
        weights = {key: round(float(value) / total, 6) for key, value in weights.items() if float(value) > 0}
    out["weights"] = weights
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args()
    root = Path(args.output_root)
    data = load_existing_daily(root, args.max_stocks, args.seed, parse_agent_json=args.parse_agent_json)
    print(f"Loaded existing daily results: {len(data)} stocks")
    policies = generate_policies(args.trials, args.seed)
    rows = []
    for idx, policy in enumerate(policies, start=1):
        metrics = evaluate_policy(data, policy, None, args.valid_end, args)
        rows.append({"trial": idx, **metrics, "policy": json.dumps(policy, ensure_ascii=False)})
        print(
            f"trial {idx}/{len(policies)} score={metrics['score']:.4f} "
            f"stocks={metrics['stocks']} annual={metrics.get('avg_annual_return', 0):+.2%} "
            f"dd={metrics.get('worst_drawdown', 0):+.2%}",
            flush=True,
        )
    ranking = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    test_rows = []
    for _, row in ranking.iterrows():
        policy = json.loads(row["policy"])
        test_metrics = evaluate_policy(data, policy, args.test_start, None, args)
        combined = 0.60 * float(row["score"]) + 0.40 * float(test_metrics.get("score", -999.0))
        test_rows.append({"trial": int(row["trial"]), "test_score": test_metrics.get("score", -999.0), "combined_score": combined, "test_metrics": test_metrics})
    test_frame = pd.DataFrame(test_rows)
    ranking = ranking.merge(test_frame[["trial", "test_score", "combined_score"]], on="trial", how="left")
    ranking = ranking.sort_values("combined_score", ascending=False).reset_index(drop=True)
    best_policy = json.loads(ranking.iloc[0]["policy"])
    test_metrics = next(item["test_metrics"] for item in test_rows if item["trial"] == int(ranking.iloc[0]["trial"]))
    best_metrics = {"valid": ranking.iloc[0].drop(labels=["policy"]).to_dict(), "test": test_metrics}
    engine_policy = to_engine_policy(best_policy)
    best_path = save_best_fusion_policy(engine_policy, best_metrics)

    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_path = report_root / f"fusion_existing_optimization_{stamp}.csv"
    report_path = report_root / f"fusion_existing_optimization_{stamp}.json"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "existing_daily_results",
        "stocks": len(data),
        "valid_end": args.valid_end,
        "test_start": args.test_start,
        "best_policy_path": str(best_path),
        "ranking_path": str(ranking_path),
        "best_policy": engine_policy,
        "offline_policy": best_policy,
        "best_metrics": best_metrics,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
