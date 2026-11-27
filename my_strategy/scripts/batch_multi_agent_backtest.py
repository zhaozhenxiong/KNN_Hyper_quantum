#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch backtest the current Multi-Agent strategy on the local A-share universe."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.agents.data_agent import DataAgent  # noqa: E402
from my_strategy.agents.factor_agent import FactorAgent  # noqa: E402
from my_strategy.agents.multi_agent_engine import MultiAgentEngine  # noqa: E402
from my_strategy.deep_learning.predictor import enrich_with_deep_predictions  # noqa: E402
from my_strategy.ml.predictor import enrich_with_ml_predictions  # noqa: E402
from my_strategy.ml.probabilistic import enrich_with_probabilistic_predictions  # noqa: E402
from my_strategy.fusion_registry import apply_best_fusion_policy  # noqa: E402
from my_strategy.rl.experience_builder import build_rl_experiences  # noqa: E402
from my_strategy.run_multi_agent_backtest import (  # noqa: E402
    RL_OUTPUT_COLUMNS,
    calc_metrics,
    load_config,
    plot_multi_agent_chart,
    resolve_stock_name,
    simulate_trades,
)
from my_strategy.model_selector import best_model_name, refresh_best_models  # noqa: E402
from my_strategy.rl.registry import best_rl_model_name  # noqa: E402
from my_strategy.storage.screen_sqlite import import_screen_outputs  # noqa: E402
from my_strategy.train_ml_model import resolve_stocks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Multi-Agent backtest on local A-share parquet universe.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "my_strategy" / "configs" / "multi_agent_config.yaml"))
    parser.add_argument("--stocks", default="local", help="local/all or comma-separated stock codes.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--ml-model", default="", help="Empty means use best registered ML model.")
    parser.add_argument("--dl-model", default="", help="Empty means use best registered DL model.")
    parser.add_argument("--rl-model", default="", help="Empty means use best registered RL model when available.")
    parser.add_argument("--output-root", default="my_strategy/data/processed/ai_agent_screen")
    parser.add_argument("--top-chart-count", type=int, default=10)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--include-st", action="store_true", help="Keep ST/delisting names in candidate charts.")
    parser.add_argument("--no-save-daily", action="store_true", help="Only save reports/ranking and top charts.")
    parser.add_argument("--no-sql", action="store_true", help="Do not import outputs into SQLite after the batch run.")
    parser.add_argument("--sql-db", default="", help="Optional SQLite database path; defaults to <output-root>/ai_agent_screen.db.")
    parser.add_argument("--no-rl-experience", action="store_true", help="Do not refresh rl_experiences after SQLite import.")
    return parser.parse_args()


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config))
    apply_best_fusion_policy(config)
    config.setdefault("data", {})["start"] = args.start
    config.setdefault("data", {})["end"] = args.end
    config.setdefault("data", {})["auto_update"] = False
    config.setdefault("outputs", {})["root_dir"] = args.output_root

    agents = config.setdefault("agents", {})
    if not args.ml_model or not args.dl_model:
        refresh_best_models()
    if not args.ml_model:
        args.ml_model = best_model_name("ml")
    if not args.dl_model:
        args.dl_model = best_model_name("deep_learning")

    ml = agents.setdefault("ml", {})
    ml["enabled"] = bool(args.ml_model)
    ml["model_name"] = args.ml_model
    agents.setdefault("ml_score", {})["enabled"] = bool(args.ml_model)

    dl = agents.setdefault("deep_learning", {})
    dl["enabled"] = bool(args.dl_model)
    dl["model_name"] = args.dl_model

    if not args.rl_model:
        args.rl_model = best_rl_model_name()
    rl = agents.setdefault("reinforcement_learning", {})
    rl["enabled"] = bool(args.rl_model)
    rl["model_name"] = args.rl_model
    return config


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "signals": root / "signals",
        "daily": root / "daily",
        "trades": root / "trades",
        "reports": root / "reports",
        "top_charts": root / "top_charts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def latest_trade_dates(trades: pd.DataFrame) -> tuple[str, str]:
    if trades.empty or "date" not in trades.columns:
        return "", ""
    data = trades.copy()
    data["date"] = pd.to_datetime(data["date"])
    opened = data[(data["action"] == "BUY") & (data["position_before"] == 0) & (data["position_after"] > 0)]
    closed = data[(data["action"].isin(["SELL", "REDUCE"])) & (data["position_before"] > 0) & (data["position_after"] == 0)]
    last_open = opened["date"].max()
    last_close = closed["date"].max()
    return (
        "" if pd.isna(last_open) else last_open.strftime("%Y-%m-%d"),
        "" if pd.isna(last_close) else last_close.strftime("%Y-%m-%d"),
    )


def numeric_latest(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, default)]), errors="coerce").fillna(default).iloc[0]
    return float(value)


def run_one(stock: str, base_config: dict[str, Any], dirs: dict[str, Path], save_daily: bool) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    data_cfg = config["data"]
    start = pd.to_datetime(data_cfg["start"])
    warmup_days = int(data_cfg.get("warmup_days", 180))
    load_start = (start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    data_agent = DataAgent(data_cfg.get("source", "local_data_center"), auto_update=False)
    raw = data_agent.load(stock, load_start, data_cfg["end"])
    factors = FactorAgent.add_factors(raw, volume_window=config.get("agents", {}).get("volume", {}).get("volume_window", 20))

    agents = config.setdefault("agents", {})
    ml_cfg = agents.setdefault("ml", {})
    ml_score_cfg = agents.setdefault("ml_score", {})
    if ml_cfg.get("enabled") and ml_cfg.get("model_name"):
        factors = enrich_with_ml_predictions(factors, ml_cfg["model_name"], ml_cfg.get("model_root"))
        meta = factors.attrs.get("ml_metadata", {})
        if meta.get("prediction_column"):
            ml_cfg["prediction_column"] = meta["prediction_column"]
            ml_score_cfg["prediction_column"] = meta["prediction_column"]
        if meta.get("prediction_kind") == "cross_sectional_rank_score":
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
    if prob_cfg.get("enabled", False):
        factors = enrich_with_probabilistic_predictions(factors, prob_cfg)
        meta = factors.attrs.get("probabilistic_metadata", {})
        prob_cfg["model_name"] = meta.get("model_name", prob_cfg.get("model_name", ""))
        prob_cfg["model_type"] = meta.get("model_type", "rolling_naive_bayes_markov")
        prob_cfg["prediction_column"] = meta.get("prediction_column", "bayes_prob_up")
        prob_cfg["score_column"] = meta.get("score_column", "prob_score")
        prob_cfg["confidence_column"] = meta.get("confidence_column", "prob_confidence")

    engine = MultiAgentEngine(config)
    decisions = engine.generate_decisions(factors)
    period = (pd.to_datetime(factors["date"]) >= pd.to_datetime(data_cfg["start"])) & (
        pd.to_datetime(factors["date"]) <= pd.to_datetime(data_cfg["end"])
    )
    factors = factors.loc[period].reset_index(drop=True)
    decisions = decisions.loc[
        (pd.to_datetime(decisions["date"]) >= pd.to_datetime(data_cfg["start"]))
        & (pd.to_datetime(decisions["date"]) <= pd.to_datetime(data_cfg["end"]))
    ].reset_index(drop=True)
    if factors.empty or decisions.empty:
        raise RuntimeError("no factor/decision rows in period")

    daily, trades = simulate_trades(factors, decisions, config)
    equity = daily.set_index("date")["equity"]
    metrics = calc_metrics(equity, float(config["backtest"]["initial_cash"]), trades)
    latest = decisions.iloc[-1]
    latest_daily = daily.iloc[-1]
    stock_name = resolve_stock_name(stock)
    last_open, last_close = latest_trade_dates(trades)

    tag = stock.replace(".", "_")
    signals_path = dirs["signals"] / f"{tag}_signals.csv"
    daily_path = dirs["daily"] / f"{tag}_daily.csv"
    trades_path = dirs["trades"] / f"{tag}_trades.csv"
    report_path = dirs["reports"] / f"{tag}_report.json"

    if save_daily:
        decisions.to_csv(signals_path, index=False, encoding="utf-8-sig")
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    row = {
        "stock": stock,
        "stock_name": stock_name,
        "status": "ok",
        "start": str(daily["date"].iloc[0])[:10],
        "end": str(daily["date"].iloc[-1])[:10],
        **metrics,
        "latest_date": str(latest.get("date", ""))[:10],
        "latest_action": latest.get("final_action", ""),
        "latest_score": numeric_latest(latest, "final_score"),
        "latest_target_weight": numeric_latest(latest, "target_weight"),
        "latest_position_weight": numeric_latest(latest_daily, "position_weight"),
        "risk_action": latest.get("risk_action", ""),
        "fallback_score": numeric_latest(latest, "fallback_score"),
        "fallback_action": latest.get("fallback_action", ""),
        "agent_weighted_score": numeric_latest(latest, "agent_weighted_score"),
        "agent_confidence_score": numeric_latest(latest, "agent_confidence_score"),
        "fusion_score": numeric_latest(latest, "fusion_score"),
        "fusion_action": latest.get("fusion_action", ""),
        "fusion_target_weight": numeric_latest(latest, "fusion_target_weight"),
        "entry_probability": numeric_latest(latest, "entry_probability"),
        "exit_probability": numeric_latest(latest, "exit_probability"),
        "fusion_activation": latest.get("fusion_activation", ""),
        "agent_score_breakdown": latest.get("agent_score_breakdown", ""),
        "ml_score": numeric_latest(latest, "ml_score"),
        "ml_score_signal": numeric_latest(latest, "ml_score_signal"),
        "dl_score": numeric_latest(latest, "dl_score"),
        "dl_score_signal": numeric_latest(latest, "dl_score_signal"),
        "prob_score": numeric_latest(latest, "prob_score"),
        "prob_score_signal": numeric_latest(latest, "prob_score_signal"),
        "prob_confidence": numeric_latest(latest, "prob_confidence"),
        "prob_action": latest.get("prob_action", ""),
        "bayes_prob_up": numeric_latest(latest, "bayes_prob_up", 0.5),
        "markov_state": latest.get("markov_state", ""),
        "markov_next_state": latest.get("markov_next_state", ""),
        "markov_state_prob": numeric_latest(latest, "markov_state_prob"),
        "rl_target_weight": numeric_latest(latest, "rl_target_weight"),
        "rl_confidence": numeric_latest(latest, "rl_confidence"),
        "rl_action": latest.get("rl_action", ""),
        "rl_model_name": latest.get("rl_model_name", ""),
        "recent_open_date": last_open,
        "recent_close_date": last_close,
        "signals_file": str(signals_path) if save_daily else "",
        "daily_file": str(daily_path) if save_daily else "",
        "trades_file": str(trades_path) if save_daily else "",
    }
    report_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    row["report_file"] = str(report_path)
    return row


def add_screen_scores(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    for col in ["annual_return", "total_return", "sharpe_ratio", "max_drawdown", "latest_score", "latest_target_weight", "ml_score_signal", "dl_score_signal", "prob_score_signal", "entry_probability", "exit_probability"]:
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0.0)

    sharpe_score = ((out["sharpe_ratio"].clip(-1, 2) + 1) / 3).clip(0, 1)
    return_score = ((out["annual_return"].clip(-0.5, 1.0) + 0.5) / 1.5).clip(0, 1)
    drawdown_score = (1 + out["max_drawdown"].clip(-0.8, 0)).clip(0, 1)
    latest_score = ((out["latest_score"].clip(-1, 1) + 1) / 2).clip(0, 1)
    entry_score = out["entry_probability"].clip(0, 1)
    model_score = ((0.45 * out["ml_score_signal"] + 0.35 * out["dl_score_signal"] + 0.20 * out["prob_score_signal"]).clip(-1, 1) + 1) / 2
    action_bonus = out["latest_action"].map({"buy": 0.16, "hold": 0.08, "watch": 0.03, "reduce": -0.03, "sell": -0.08}).fillna(0.0)
    holding_bonus = out["latest_position_weight"].clip(0, 1) * 0.08

    out["screen_score"] = (
        0.18 * latest_score
        + 0.06 * entry_score
        + 0.22 * model_score
        + 0.18 * sharpe_score
        + 0.16 * return_score
        + 0.12 * drawdown_score
        + 0.08 * out["latest_target_weight"].clip(0, 1)
        + action_bonus
        + holding_bonus
    )
    out.sort_values(["screen_score", "latest_score", "annual_return"], ascending=False, inplace=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def filter_candidate_universe(ranking: pd.DataFrame, include_st: bool = False) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    candidates = ranking[ranking["latest_action"].isin(["buy", "hold", "watch"])].copy()
    if include_st or candidates.empty or "stock_name" not in candidates.columns:
        return candidates
    names = candidates["stock_name"].fillna("").astype(object).astype(str).str.upper()
    delist_token = "\u9000"
    keep = ~(names.str.contains("ST", regex=False, na=False) | names.str.contains(delist_token, regex=False, na=False))
    keep = ~(names.str.contains("ST", regex=False, na=False) | names.str.contains("退", regex=False, na=False))
    keep = ~(names.str.contains("ST", regex=False, na=False) | names.str.contains(delist_token, regex=False, na=False))
    candidates = candidates.loc[keep].copy().reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)
    return candidates


def write_top_charts(ranking: pd.DataFrame, base_config: dict[str, Any], dirs: dict[str, Path], count: int) -> None:
    for old in dirs["top_charts"].glob("*.png"):
        old.unlink()
    if ranking.empty or count <= 0:
        return
    for _, row in ranking.head(count).iterrows():
        daily_path = Path(str(row.get("daily_file", "")))
        trades_path = Path(str(row.get("trades_file", "")))
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
        rank = int(row["rank"])
        stock = row["stock"]
        stock_name = str(row.get("stock_name", "") or "")
        safe_name = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in stock_name).strip() or "unknown"
        out_path = dirs["top_charts"] / f"rank_{rank:02d}_{stock.replace('.', '_')}_{safe_name}_score_{row['screen_score']:.3f}.png"
        plot_multi_agent_chart(
            stock,
            stock_name,
            daily,
            trades,
            out_path,
            bool(base_config.get("backtest", {}).get("mark_full_cycles_only", True)),
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    args = parse_args()
    stocks, stock_source = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    config = prepare_config(args)
    dirs = ensure_dirs(PROJECT_ROOT / args.output_root)

    print("=" * 72)
    print("KHQuant Multi-Agent batch backtest")
    print(f"stocks={len(stocks)} source={stock_source}")
    print(f"period={args.start} ~ {args.end}")
    print(f"ml_model={args.ml_model}")
    print(f"dl_model={args.dl_model}")
    print(f"rl_model={args.rl_model}")
    print(f"output={dirs['root']}")
    print("=" * 72)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_ts = time.time()
    save_daily = not args.no_save_daily

    for idx, stock in enumerate(stocks, start=1):
        try:
            row = run_one(stock, config, dirs, save_daily=save_daily)
            rows.append(row)
            status = f"{row['latest_action']} score={row['latest_score']:.3f} ret={row['total_return']:+.2%}"
        except Exception as exc:
            row = {"stock": stock, "status": "error", "error": str(exc)[:180]}
            failures.append(row)
            status = f"ERROR {row['error']}"

        if idx == 1 or idx % 10 == 0 or idx == len(stocks):
            elapsed = max(time.time() - start_ts, 1)
            rate = idx / elapsed
            remain = (len(stocks) - idx) / rate if rate else 0
            print(f"[{idx}/{len(stocks)}] {stock} -> {status} | {rate:.2f}/s remain={remain/60:.1f}m", flush=True)

        if idx % 50 == 0:
            partial = add_screen_scores(pd.DataFrame(rows))
            partial.to_csv(dirs["reports"] / "ranking_partial.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(failures).to_csv(dirs["reports"] / "failures_partial.csv", index=False, encoding="utf-8-sig")

    ranking = add_screen_scores(pd.DataFrame(rows))
    candidates = filter_candidate_universe(ranking, include_st=args.include_st)
    failures_df = pd.DataFrame(failures)

    ranking_path = dirs["reports"] / "ranking.csv"
    candidates_path = dirs["reports"] / "buy_candidates.csv"
    failures_path = dirs["reports"] / "failures.csv"
    summary_path = dirs["reports"] / "summary.json"

    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")
    write_top_charts(candidates, config, dirs, args.top_chart_count)

    summary = {
        "stocks": len(stocks),
        "success": int(len(ranking)),
        "failed": int(len(failures_df)),
        "start": args.start,
        "end": args.end,
        "ml_model": args.ml_model,
        "dl_model": args.dl_model,
        "rl_model": args.rl_model,
        "ranking_file": str(ranking_path),
        "buy_candidates_file": str(candidates_path),
        "failures_file": str(failures_path),
        "top_charts_dir": str(dirs["top_charts"]),
        "elapsed_seconds": round(time.time() - start_ts, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sql:
        try:
            sql_result = import_screen_outputs(
                output_root=dirs["root"],
                db_path=args.sql_db or None,
                include_details=save_daily,
            )
            summary["sqlite_db"] = sql_result["db_path"]
            summary["sqlite_run_id"] = sql_result["run_id"]
            summary["sqlite_counts"] = sql_result["counts"]
            if not args.no_rl_experience and save_daily:
                rl_info = build_rl_experiences(db_path=sql_result["db_path"], source_run_id=sql_result["run_id"])
                summary["rl_experience_rows"] = rl_info["rows"]
                summary["rl_experience_source_run_id"] = rl_info["source_run_id"]
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"SQLite imported: {sql_result['db_path']} run_id={sql_result['run_id']}")
            if "rl_experience_rows" in summary:
                print(f"RL experiences refreshed: rows={summary['rl_experience_rows']}")
        except Exception as exc:
            print(f"SQLite import skipped: {exc}")

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not candidates.empty:
        cols = ["rank", "stock", "stock_name", "latest_action", "latest_score", "entry_probability", "exit_probability", "latest_target_weight", "screen_score", "annual_return", "max_drawdown", "sharpe_ratio", "ml_score_signal", "dl_score_signal", "prob_score_signal"]
        print("\nTop candidates:")
        print(candidates[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
