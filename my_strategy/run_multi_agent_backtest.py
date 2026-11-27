#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run KHQuant-MultiAgent-V1 backtests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.agents.data_agent import DataAgent  # noqa: E402
from my_strategy.agents.factor_agent import FactorAgent  # noqa: E402
from my_strategy.agents.multi_agent_engine import MultiAgentEngine  # noqa: E402
from my_strategy.agents.qbot_bridge_agent import QbotBridgeAgent  # noqa: E402
from my_strategy.data_manager import LocalCache  # noqa: E402
from my_strategy.deep_learning.predictor import enrich_with_deep_predictions  # noqa: E402
from my_strategy.fusion_registry import apply_best_fusion_policy  # noqa: E402
from my_strategy.model_selector import best_model_name, refresh_best_models  # noqa: E402
from my_strategy.ml.predictor import enrich_with_ml_predictions  # noqa: E402
from my_strategy.ml.probabilistic import enrich_with_probabilistic_predictions  # noqa: E402
from my_strategy.rl.experience_builder import build_rl_experiences  # noqa: E402
from my_strategy.rl.registry import best_rl_model_name  # noqa: E402
from my_strategy.maintenance import (  # noqa: E402
    format_maintenance_preview,
    run_maintenance,
    summarize_maintenance_report,
)
from my_strategy.storage.screen_sqlite import import_screen_outputs  # noqa: E402
from my_strategy.train_ml_model import resolve_stocks  # noqa: E402


ML_OUTPUT_COLUMNS = [
    "ml_score",
    "ml_score_signal",
    "ml_confidence",
    "ml_prediction",
    "ml_model_name",
    "ml_prediction_kind",
    "ml_score_action",
    "ml_score_reason",
]

DL_OUTPUT_COLUMNS = [
    "dl_score",
    "dl_score_signal",
    "dl_confidence",
    "dl_prediction",
    "dl_model_name",
    "dl_model_type",
    "dl_prediction_kind",
    "dl_action",
    "dl_reason",
]

PROB_OUTPUT_COLUMNS = [
    "prob_score",
    "prob_score_signal",
    "prob_confidence",
    "prob_action",
    "prob_model_name",
    "prob_reason",
    "bayes_prob_up",
    "bayes_score",
    "markov_state",
    "markov_next_state",
    "markov_state_prob",
    "markov_score",
]

FUSION_OUTPUT_COLUMNS = [
    "fallback_score",
    "fallback_action",
    "agent_weighted_score",
    "agent_confidence_score",
    "fusion_score",
    "fusion_action",
    "fusion_target_weight",
    "entry_probability",
    "exit_probability",
    "fusion_activation",
    "agent_score_breakdown",
]

RL_OUTPUT_COLUMNS = [
    "base_target_weight",
    "rl_model_name",
    "rl_target_weight",
    "rl_confidence",
    "rl_action",
    "rl_q_values",
    "rl_reason",
]

LOCAL_STOCK_TOKENS = {"all", "local", "local_all", "*"}
AI_AGENT_SCREEN_ROOT = "my_strategy/data/processed/ai_agent_screen"


def resolve_stock_name(stock: str) -> str:
    """自动获取股票中文名称，来源优先级：本地CSV > 腾讯财经API > 回退码"""
    # 1. 查本地CSV文件
    lookup_files = [
        PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "reports" / "ranking.csv",
        PROJECT_ROOT / "my_strategy" / "data" / "raw" / "stock_basic" / "all_a_stock_list.parquet",
        PROJECT_ROOT / "my_strategy" / "data" / "raw" / "custom_pool" / "my_pool.csv",
    ]
    for path in lookup_files:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            code_col = "stock" if "stock" in df.columns else "code" if "code" in df.columns else None
            name_col = "stock_name" if "stock_name" in df.columns else "name" if "name" in df.columns else None
            if code_col is None or name_col is None:
                continue
            hit = df[df[code_col].astype(str).eq(stock)]
            if not hit.empty:
                name = str(hit.iloc[0].get(name_col, "") or "").strip()
                if name:
                    return name
        except Exception:
            continue

    # 2. 通过腾讯财经API查询
    try:
        market = "sh" if stock.endswith(".SH") else "sz"
        code = stock.split(".")[0]
        url = f"https://qt.gtimg.cn/q={market}{code}"
        result = subprocess.run(
            ["curl", "-s", "--noproxy", "*", url],
            capture_output=True, timeout=10
        )
        # API返回GBK编码，手动解码
        text = result.stdout.decode("gbk", errors="ignore")
        # 返回格式: v_sh600707="1~600707~彩虹股份~..."
        if "~" in text:
            parts = text.split("~")
            if len(parts) >= 3:
                name = parts[1].strip()
                if name and not name.startswith("v_"):
                    return name
    except Exception:
        pass

    # 3. 回退：直接返回代码
    return stock


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def ensure_output_dirs(root_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": root_dir,
        "signals": root_dir / "signals",
        "figures": root_dir / "figures",
        "reports": root_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_filename_part(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|\\s]+', "_", text)
    return text.strip("_") or "unknown"


def calc_metrics(equity: pd.Series, initial_cash: float, trades: pd.DataFrame) -> dict[str, Any]:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_cash - 1
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    annual_return = (equity.iloc[-1] / initial_cash) ** (365 / days) - 1
    peak = equity.cummax()
    drawdown = equity / peak - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    paired = 0
    wins = 0
    if not trades.empty:
        buys = trades[trades["action"] == "BUY"].reset_index(drop=True)
        sells = trades[trades["action"].isin(["SELL", "REDUCE"])].reset_index(drop=True)
        paired = min(len(buys), len(sells))
        for i in range(paired):
            if sells.loc[i, "cash_flow"] + buys.loc[i, "cash_flow"] > 0:
                wins += 1

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": float(sharpe),
        "n_trades": int(len(trades)),
        "paired_trades": int(paired),
        "win_rate": float(wins / paired) if paired else 0.0,
        "days": int(days),
        "final_equity": float(equity.iloc[-1]),
    }


def plot_multi_agent_chart(stock: str, stock_name: str, daily: pd.DataFrame, trades: pd.DataFrame, output_path: Path, mark_full_cycles_only: bool = True) -> None:
    try:
        from my_strategy.charting import plot_backtest_chart

        plot_backtest_chart(
            f"{stock_name} {stock}",
            daily,
            trades,
            output_path,
            score_col="final_score",
            mark_full_cycles_only=mark_full_cycles_only,
        )
    except Exception as exc:
        print(f"图表生成跳过: {exc}")


def simulate_trades(df: pd.DataFrame, decisions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = config.get("backtest", {})
    initial_cash = float(bt.get("initial_cash", 100000))
    commission = float(bt.get("commission", 0.0003))
    stamp_tax = float(bt.get("stamp_tax", 0.0005))
    slippage = float(bt.get("slippage", 0.0005))
    lot_size = int(bt.get("lot_size", 100))
    rebalance_interval_days = int(bt.get("rebalance_interval_days", 1))
    min_weight_change = float(bt.get("min_weight_change", 0.0))
    reduce_weight = float(config.get("portfolio", {}).get("reduce_weight", 0.30))

    decision_cols = ["date", "final_score", "final_action", "target_weight", "risk_action", "reason_json"]
    decision_cols += [col for col in ML_OUTPUT_COLUMNS if col in decisions.columns]
    decision_cols += [col for col in DL_OUTPUT_COLUMNS if col in decisions.columns]
    decision_cols += [col for col in PROB_OUTPUT_COLUMNS if col in decisions.columns]
    decision_cols += [col for col in FUSION_OUTPUT_COLUMNS if col in decisions.columns]
    decision_cols += [col for col in RL_OUTPUT_COLUMNS if col in decisions.columns]
    merged = df.merge(
        decisions[decision_cols],
        on="date",
        how="inner",
        suffixes=("", "_decision"),
    )

    cash = initial_cash
    shares = 0
    avg_cost = np.nan
    entry_price = None
    last_trade_date = None
    trades = []
    daily_rows = []

    for _, row in merged.iterrows():
        signal_close = float(row["close"])
        trade_close = pd.to_numeric(pd.Series([row.get("trade_close", np.nan)]), errors="coerce").iloc[0]
        close = float(trade_close) if pd.notna(trade_close) and float(trade_close) > 0 else signal_close
        signal_open = pd.to_numeric(pd.Series([row.get("open", np.nan)]), errors="coerce").iloc[0]
        signal_high = pd.to_numeric(pd.Series([row.get("high", np.nan)]), errors="coerce").iloc[0]
        signal_low = pd.to_numeric(pd.Series([row.get("low", np.nan)]), errors="coerce").iloc[0]
        trade_open = pd.to_numeric(pd.Series([row.get("trade_open", np.nan)]), errors="coerce").iloc[0]
        trade_high = pd.to_numeric(pd.Series([row.get("trade_high", np.nan)]), errors="coerce").iloc[0]
        trade_low = pd.to_numeric(pd.Series([row.get("trade_low", np.nan)]), errors="coerce").iloc[0]
        open_price = float(trade_open) if pd.notna(trade_open) and float(trade_open) > 0 else float(signal_open) if pd.notna(signal_open) else close
        high_price = float(trade_high) if pd.notna(trade_high) and float(trade_high) > 0 else float(signal_high) if pd.notna(signal_high) else max(open_price, close)
        low_price = float(trade_low) if pd.notna(trade_low) and float(trade_low) > 0 else float(signal_low) if pd.notna(signal_low) else min(open_price, close)
        high_price = max(high_price, open_price, close)
        low_price = min(low_price, open_price, close)
        if not np.isfinite(close) or close <= 0:
            continue
        date = row["date"]
        equity_before = cash + shares * close
        current_weight = shares * close / equity_before if equity_before else 0.0
        target_weight = float(row["target_weight"])

        if row["final_action"] in ["sell"]:
            target_weight = 0.0
        elif row["final_action"] == "reduce":
            target_weight = min(target_weight, reduce_weight)

        is_force_exit = row["risk_action"] == "force_sell" or row["final_action"] == "sell"
        days_since_trade = None
        if last_trade_date is not None:
            days_since_trade = (pd.to_datetime(date) - pd.to_datetime(last_trade_date)).days
        can_rebalance = (
            is_force_exit
            or last_trade_date is None
            or days_since_trade is None
            or days_since_trade >= rebalance_interval_days
        )
        if abs(target_weight - current_weight) < min_weight_change and not is_force_exit:
            target_weight = current_weight
        if not can_rebalance:
            target_weight = current_weight

        target_value = equity_before * target_weight
        current_value = shares * close
        diff_value = target_value - current_value

        if diff_value > close * lot_size:
            trade_price = close * (1 + slippage)
            buy_shares = int(diff_value / (trade_price * lot_size)) * lot_size
            if buy_shares > 0:
                position_before = shares
                cost = buy_shares * trade_price
                fee = max(cost * commission, 5)
                if cost + fee <= cash:
                    old_cost_basis = 0 if shares == 0 or pd.isna(avg_cost) else avg_cost * shares
                    cash -= cost + fee
                    shares += buy_shares
                    avg_cost = (old_cost_basis + cost + fee) / shares if shares > 0 else np.nan
                    position_after = shares
                    if entry_price is None:
                        entry_price = avg_cost
                    last_trade_date = date
                    trades.append({
                        "date": date,
                        "action": "BUY",
                        "price": trade_price,
                        "shares": buy_shares,
                        "position_before": position_before,
                        "position_after": position_after,
                        "cash_flow": -(cost + fee),
                        "fee": fee,
                        "holding_cost_after": avg_cost,
                        "target_weight": target_weight,
                        "position_weight_before": current_weight,
                        "reason": row["reason_json"],
                    })

        elif diff_value <= -close * lot_size and shares > 0:
            trade_price = close * (1 - slippage)
            sell_shares = min(int(abs(diff_value) / (trade_price * lot_size)) * lot_size, shares)
            if row["final_action"] == "sell":
                sell_shares = shares
            if sell_shares > 0:
                position_before = shares
                proceeds = sell_shares * trade_price
                fee = max(proceeds * commission, 5)
                tax = proceeds * stamp_tax
                net = proceeds - fee - tax
                cash += net
                shares -= sell_shares
                position_after = shares
                if shares == 0:
                    entry_price = None
                    avg_cost = np.nan
                last_trade_date = date
                trades.append({
                    "date": date,
                    "action": "SELL" if row["final_action"] == "sell" else "REDUCE",
                    "price": trade_price,
                    "shares": sell_shares,
                    "position_before": position_before,
                    "position_after": position_after,
                    "cash_flow": net,
                    "fee": fee + tax,
                    "holding_cost_after": avg_cost,
                    "target_weight": target_weight,
                    "position_weight_before": current_weight,
                    "reason": row["reason_json"],
                })

        equity = cash + shares * close
        daily_rows.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close,
            "factor_close": signal_close,
            "cash": cash,
            "shares": shares,
            "equity": equity,
            "position_weight": shares * close / equity if equity else 0,
            "holding_cost": avg_cost,
            "avg_cost": avg_cost,
            "unrealized_return": close / avg_cost - 1 if shares > 0 and not pd.isna(avg_cost) and avg_cost else np.nan,
            "final_score": row["final_score"],
            "final_action": row["final_action"],
            "target_weight": row["target_weight"],
            "effective_target_weight": target_weight,
            "risk_action": row["risk_action"],
            "reason_json": row["reason_json"],
            "entry_price": entry_price,
        })
        for col in ML_OUTPUT_COLUMNS:
            value = _first_row_value(row, [col, f"{col}_decision", f"{col}_y", f"{col}_x"])
            if value is not None:
                daily_rows[-1][col] = value
        for col in DL_OUTPUT_COLUMNS:
            value = _first_row_value(row, [col, f"{col}_decision", f"{col}_y", f"{col}_x"])
            if value is not None:
                daily_rows[-1][col] = value
        for col in PROB_OUTPUT_COLUMNS:
            value = _first_row_value(row, [col, f"{col}_decision", f"{col}_y", f"{col}_x"])
            if value is not None:
                daily_rows[-1][col] = value
        for col in FUSION_OUTPUT_COLUMNS:
            value = _first_row_value(row, [col, f"{col}_decision", f"{col}_y", f"{col}_x"])
            if value is not None:
                daily_rows[-1][col] = value
        for col in RL_OUTPUT_COLUMNS:
            value = _first_row_value(row, [col, f"{col}_decision", f"{col}_y", f"{col}_x"])
            if value is not None:
                daily_rows[-1][col] = value

    daily = pd.DataFrame(daily_rows)
    trade_df = pd.DataFrame(trades)
    return daily, trade_df


def _first_row_value(row: pd.Series, names: list[str]):
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return value
    return None


def save_outputs(stock: str, decisions: pd.DataFrame, daily: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    output_root = PROJECT_ROOT / config.get("outputs", {}).get("root_dir", AI_AGENT_SCREEN_ROOT)
    dirs = ensure_output_dirs(output_root)
    tag = stock.replace(".", "_")
    cache = LocalCache()

    signals_path = dirs["signals"] / f"{tag}_multi_agent_signals.csv"
    daily_path = dirs["signals"] / f"{tag}_multi_agent_daily.csv"
    trades_path = dirs["signals"] / f"{tag}_multi_agent_trades.csv"
    report_path = dirs["reports"] / f"{tag}_multi_agent_report.json"
    qbot_path = dirs["signals"] / f"{tag}_qbot_signal.csv"
    latest_figure_path = dirs["figures"] / f"{tag}_multi_agent_equity.png"

    decisions.to_csv(signals_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    equity = daily.set_index("date")["equity"]
    metrics = calc_metrics(equity, config["backtest"]["initial_cash"], trades)
    stock_name = resolve_stock_name(stock)
    start_tag = pd.to_datetime(daily["date"]).min().strftime("%Y%m%d")
    end_tag = pd.to_datetime(daily["date"]).max().strftime("%Y%m%d")
    name_tag = safe_filename_part(stock_name)
    report_figure_path = dirs["figures"] / f"{tag}_{name_tag}_{start_tag}_{end_tag}_multi_agent.png"
    plot_multi_agent_chart(stock, stock_name, daily, trades, report_figure_path, bool(config.get("backtest", {}).get("mark_full_cycles_only", False)))
    shutil.copy2(report_figure_path, latest_figure_path)
    latest = decisions.iloc[-1].to_dict()
    report = {
        "strategy_name": config.get("strategy_name", "KHQuant-MultiAgent-V1"),
        "stock": stock,
        "stock_name": stock_name,
        "data_center": {
            "daily_cache": str(cache.stock_daily_path(stock)),
            "processed_output_root": str(output_root),
            "auto_update": bool(config.get("data", {}).get("auto_update", True)),
        },
        "metrics": metrics,
        "latest_decision": {
            "date": str(latest.get("date"))[:10],
            "final_action": latest.get("final_action"),
            "final_score": latest.get("final_score"),
            "target_weight": latest.get("target_weight"),
            "risk_action": latest.get("risk_action"),
            "fallback_score": latest.get("fallback_score"),
            "fallback_action": latest.get("fallback_action"),
            "agent_weighted_score": latest.get("agent_weighted_score"),
            "agent_confidence_score": latest.get("agent_confidence_score"),
            "fusion_score": latest.get("fusion_score"),
            "fusion_action": latest.get("fusion_action"),
            "fusion_target_weight": latest.get("fusion_target_weight"),
            "entry_probability": latest.get("entry_probability"),
            "exit_probability": latest.get("exit_probability"),
            "fusion_activation": latest.get("fusion_activation"),
            "agent_score_breakdown": latest.get("agent_score_breakdown"),
            "ml_score": latest.get("ml_score"),
            "ml_score_signal": latest.get("ml_score_signal"),
            "ml_score_action": latest.get("ml_score_action"),
            "ml_confidence": latest.get("ml_confidence"),
            "ml_prediction": latest.get("ml_prediction"),
            "ml_prediction_kind": latest.get("ml_prediction_kind"),
            "dl_score": latest.get("dl_score"),
            "dl_score_signal": latest.get("dl_score_signal"),
            "dl_action": latest.get("dl_action"),
            "dl_confidence": latest.get("dl_confidence"),
            "dl_prediction": latest.get("dl_prediction"),
            "dl_prediction_kind": latest.get("dl_prediction_kind"),
            "prob_score": latest.get("prob_score"),
            "prob_score_signal": latest.get("prob_score_signal"),
            "prob_action": latest.get("prob_action"),
            "prob_confidence": latest.get("prob_confidence"),
            "prob_model_name": latest.get("prob_model_name"),
            "bayes_prob_up": latest.get("bayes_prob_up"),
            "bayes_score": latest.get("bayes_score"),
            "markov_state": latest.get("markov_state"),
            "markov_next_state": latest.get("markov_next_state"),
            "markov_state_prob": latest.get("markov_state_prob"),
            "markov_score": latest.get("markov_score"),
            "rl_model_name": latest.get("rl_model_name"),
            "rl_target_weight": latest.get("rl_target_weight"),
            "rl_confidence": latest.get("rl_confidence"),
            "rl_action": latest.get("rl_action"),
            "reason": latest.get("reason"),
        },
        "ml": {
            "enabled": bool(config.get("agents", {}).get("ml", {}).get("enabled", False)),
            "score_agent_enabled": bool(config.get("agents", {}).get("ml_score", {}).get("enabled", False)),
            "model_name": config.get("agents", {}).get("ml", {}).get("model_name", ""),
            "prediction_column": config.get("agents", {}).get("ml", {}).get("prediction_column", ""),
            "score_column": config.get("agents", {}).get("ml_score", {}).get("score_column", config.get("agents", {}).get("ml", {}).get("score_column", "")),
        },
        "deep_learning": {
            "enabled": bool(config.get("agents", {}).get("deep_learning", {}).get("enabled", False)),
            "model_name": config.get("agents", {}).get("deep_learning", {}).get("model_name", ""),
            "model_type": config.get("agents", {}).get("deep_learning", {}).get("model_type", ""),
            "prediction_column": config.get("agents", {}).get("deep_learning", {}).get("prediction_column", ""),
            "score_column": config.get("agents", {}).get("deep_learning", {}).get("score_column", ""),
        },
        "probabilistic": {
            "enabled": bool(config.get("agents", {}).get("probabilistic", {}).get("enabled", False)),
            "model_name": config.get("agents", {}).get("probabilistic", {}).get("model_name", ""),
            "model_type": config.get("agents", {}).get("probabilistic", {}).get("model_type", "rolling_naive_bayes_markov"),
            "prediction_column": config.get("agents", {}).get("probabilistic", {}).get("prediction_column", "bayes_prob_up"),
            "score_column": config.get("agents", {}).get("probabilistic", {}).get("score_column", "prob_score"),
        },
        "fusion": config.get("decision", {}).get("fusion", {}),
        "reinforcement_learning": {
            "enabled": bool(config.get("agents", {}).get("reinforcement_learning", {}).get("enabled", False)),
            "model_name": config.get("agents", {}).get("reinforcement_learning", {}).get("model_name", ""),
            "blend": config.get("agents", {}).get("reinforcement_learning", {}).get("blend", 0.15),
            "allow_entry": config.get("agents", {}).get("reinforcement_learning", {}).get("allow_entry", False),
        },
        "files": {
            "signals": str(signals_path),
            "daily": str(daily_path),
            "trades": str(trades_path),
            "qbot": str(qbot_path),
            "figure": str(report_figure_path),
            "latest_figure": str(latest_figure_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    qbot_source = decisions.copy()
    qbot_source["stock"] = stock
    QbotBridgeAgent().export_to_qbot_format(qbot_source, qbot_path)

    return {
        "signals": signals_path,
        "daily": daily_path,
        "trades": trades_path,
        "report": report_path,
        "qbot": qbot_path,
        "figure": report_figure_path,
        "latest_figure": latest_figure_path,
    }


def run_stock(stock: str, config: dict[str, Any]) -> dict[str, Path]:
    data_cfg = config["data"]
    auto_update = bool(data_cfg.get("auto_update", True))
    start = pd.to_datetime(data_cfg["start"])
    end = pd.to_datetime(data_cfg["end"])
    warmup_days = int(data_cfg.get("warmup_days", 180))
    load_start = (start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    print(f"📦 Local Data Center: {stock} auto_update={auto_update}")
    data_agent = DataAgent(data_cfg.get("source", "local_data_center"), auto_update=auto_update)
    raw = data_agent.load(stock, load_start, data_cfg["end"])
    volume_window = config.get("agents", {}).get("volume", {}).get("volume_window", 20)
    factors = FactorAgent.add_factors(raw, volume_window=volume_window)
    agents_cfg = config.setdefault("agents", {})
    ml_cfg = agents_cfg.get("ml", {})
    ml_score_cfg = agents_cfg.setdefault("ml_score", {})
    dl_cfg = agents_cfg.setdefault("deep_learning", {})
    prob_cfg = agents_cfg.setdefault("probabilistic", {})
    if ml_cfg.get("enabled", False):
        model_name = ml_cfg.get("model_name", "")
        if model_name:
            try:
                factors = enrich_with_ml_predictions(factors, model_name, ml_cfg.get("model_root"))
                ml_metadata = factors.attrs.get("ml_metadata", {})
                if ml_metadata.get("prediction_column"):
                    ml_cfg["prediction_column"] = ml_metadata["prediction_column"]
                    ml_score_cfg["prediction_column"] = ml_metadata["prediction_column"]
                if ml_metadata.get("prediction_kind") == "cross_sectional_rank_score":
                    ml_cfg["score_column"] = "ml_score"
                    ml_cfg["confidence_column"] = "ml_confidence"
                    ml_score_cfg["score_column"] = "ml_score"
                    ml_score_cfg["confidence_column"] = "ml_confidence"
                ml_score_cfg["enabled"] = bool(ml_score_cfg.get("enabled", True))
            except FileNotFoundError:
                print(f"ML??????????????: {model_name}")
                ml_cfg["enabled"] = False
                ml_score_cfg["enabled"] = False
            except Exception as exc:
                print(f"ML?????????????: {exc}")
                ml_cfg["enabled"] = False
                ml_score_cfg["enabled"] = False
        else:
            print("agents.ml.enabled=true ???? model_name?????????")
            ml_cfg["enabled"] = False
            ml_score_cfg["enabled"] = False
    if dl_cfg.get("enabled", False):
        model_name = dl_cfg.get("model_name", "")
        if model_name:
            try:
                factors = enrich_with_deep_predictions(factors, model_name, dl_cfg.get("model_root"))
                dl_metadata = factors.attrs.get("dl_metadata", {})
                dl_cfg["prediction_column"] = dl_metadata.get("prediction_column", "dl_prediction")
                dl_cfg["score_column"] = dl_metadata.get("score_column", "dl_score")
                dl_cfg["confidence_column"] = dl_metadata.get("confidence_column", "dl_confidence")
                dl_cfg["model_type"] = dl_metadata.get("model_type", "")
            except FileNotFoundError:
                print(f"深度学习模型不存在，跳过DL信号: {model_name}")
                dl_cfg["enabled"] = False
            except Exception as exc:
                print(f"深度学习预测失败，跳过DL信号: {exc}")
                dl_cfg["enabled"] = False
        else:
            print("agents.deep_learning.enabled=true 但未配置 model_name，跳过DL信号")
            dl_cfg["enabled"] = False
    if prob_cfg.get("enabled", False):
        try:
            factors = enrich_with_probabilistic_predictions(factors, prob_cfg)
            prob_metadata = factors.attrs.get("probabilistic_metadata", {})
            prob_cfg["model_name"] = prob_metadata.get("model_name", prob_cfg.get("model_name", ""))
            prob_cfg["model_type"] = prob_metadata.get("model_type", "rolling_naive_bayes_markov")
            prob_cfg["prediction_column"] = prob_metadata.get("prediction_column", "bayes_prob_up")
            prob_cfg["score_column"] = prob_metadata.get("score_column", "prob_score")
            prob_cfg["confidence_column"] = prob_metadata.get("confidence_column", "prob_confidence")
        except Exception as exc:
            print(f"概率模型预测失败，跳过ProbabilisticAgent: {exc}")
            prob_cfg["enabled"] = False
    engine = MultiAgentEngine(config)
    decisions = engine.generate_decisions(factors)
    period_mask = (pd.to_datetime(factors["date"]) >= start) & (pd.to_datetime(factors["date"]) <= end)
    factors = factors.loc[period_mask].reset_index(drop=True)
    decisions = decisions.loc[(pd.to_datetime(decisions["date"]) >= start) & (pd.to_datetime(decisions["date"]) <= end)].reset_index(drop=True)
    if factors.empty or decisions.empty:
        raise RuntimeError(f"{stock} 在 {data_cfg['start']} ~ {data_cfg['end']} 没有足够的因子数据，请增加 warmup_days 或检查本地数据。")
    daily, trades = simulate_trades(factors, decisions, config)
    paths = save_outputs(stock, decisions, daily, trades, config)

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    metrics = report["metrics"]
    latest = report["latest_decision"]
    print("\n" + "=" * 72)
    print(f"{config.get('strategy_name')} / {stock}")
    print(f"总收益: {metrics['total_return']:.2%} | 年化: {metrics['annual_return']:.2%} | 最大回撤: {metrics['max_drawdown']:.2%} | 夏普: {metrics['sharpe_ratio']:.2f}")
    print(f"交易次数: {metrics['n_trades']} | 胜率: {metrics['win_rate']:.1%}")
    print(f"最新决策: {latest['date']} {latest['final_action']} score={latest['final_score']} weight={latest['target_weight']}")
    print(f"信号: {paths['signals']}")
    print(f"报告: {paths['report']}")
    return paths


def _split_stock_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_local_universe(value: str) -> bool:
    return (value or "").strip().lower() in LOCAL_STOCK_TOKENS


def _is_many_stock_request(stocks_arg: str, stocks: list[str], threshold: int = 20) -> bool:
    return _is_local_universe(stocks_arg) or len(stocks) > threshold


def _apply_best_models(
    config: dict[str, Any],
    explicit_ml: str = "",
    explicit_dl: str = "",
    explicit_rl: str = "",
    auto_model: bool = True,
    auto_rl: bool = True,
) -> tuple[str, str]:
    if not auto_model and not explicit_ml and not explicit_dl and not explicit_rl and not auto_rl:
        return "", ""
    if auto_model:
        refresh_best_models()
    ml_model = explicit_ml or (best_model_name("ml") if auto_model else "")
    dl_model = explicit_dl or (best_model_name("deep_learning") if auto_model else "")

    agents_cfg = config.setdefault("agents", {})
    if ml_model:
        ml_cfg = agents_cfg.setdefault("ml", {})
        ml_cfg["enabled"] = True
        ml_cfg["model_name"] = ml_model
        agents_cfg.setdefault("ml_score", {})["enabled"] = True
    if dl_model:
        dl_cfg = agents_cfg.setdefault("deep_learning", {})
        dl_cfg["enabled"] = True
        dl_cfg["model_name"] = dl_model
    rl_model = explicit_rl or (best_rl_model_name() if auto_rl else "")
    if rl_model:
        rl_cfg = agents_cfg.setdefault("reinforcement_learning", {})
        rl_cfg["enabled"] = True
        rl_cfg["model_name"] = rl_model
    return ml_model, dl_model


def _run_subprocess(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _maintenance_namespace(
    args: argparse.Namespace,
    output_root: str | Path,
    db_path: str | Path,
    execute: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        execute=execute,
        vacuum=bool(args.maintenance_vacuum),
        delete_files=bool(args.maintenance_delete_files),
        delete_models=bool(args.maintenance_delete_models),
        include_legacy_backtests=bool(args.maintenance_include_legacy_backtests),
        output_root=str(_resolve_project_path(output_root)),
        processed_root=str(_resolve_project_path(args.maintenance_processed_root)),
        db_path=str(_resolve_project_path(db_path)),
        keep_latest_runs=args.maintenance_keep_latest_runs,
        keep_detail_runs=args.maintenance_keep_detail_runs,
        keep_summary_runs=args.maintenance_keep_summary_runs,
        keep_rl_experience_runs=args.maintenance_keep_rl_experience_runs,
        keep_run_logs=args.maintenance_keep_run_logs,
        keep_figures=args.maintenance_keep_figures,
        keep_models_per_kind=args.maintenance_keep_models_per_kind,
        keep_top_models_per_kind=args.maintenance_keep_top_models_per_kind,
    )


def _maintenance_output_root(args: argparse.Namespace, current_output_root: str | Path) -> Path:
    if args.maintenance_output_root:
        return _resolve_project_path(args.maintenance_output_root)
    return _resolve_project_path(current_output_root)


def _maintenance_db_path(args: argparse.Namespace, output_root: Path) -> Path:
    if args.maintenance_db_path:
        return _resolve_project_path(args.maintenance_db_path)
    if args.sql_db:
        return _resolve_project_path(args.sql_db)
    output_db = output_root / "ai_agent_screen.db"
    if output_root.name.startswith("ai_agent_screen"):
        return output_db
    main_screen_db = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "ai_agent_screen.db"
    return main_screen_db if main_screen_db.exists() else output_db


def _maintenance_log_dir(args: argparse.Namespace) -> Path:
    path = args.maintenance_log_dir or "my_strategy/data/processed/maintenance_logs"
    return _resolve_project_path(path)


def _write_maintenance_log(
    args: argparse.Namespace,
    preview: dict[str, Any],
    result: dict[str, Any] | None,
    decision: str,
    context: dict[str, Any],
) -> Path:
    log_dir = _maintenance_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"maintenance_{stamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "context": context,
        "preview_summary": summarize_maintenance_report(preview),
        "preview": preview,
        "result_summary": summarize_maintenance_report(result) if result else None,
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _has_maintenance_candidates(preview: dict[str, Any]) -> bool:
    summary = summarize_maintenance_report(preview)
    return (
        int(summary["sql_candidate_rows"]) > 0
        or int(summary["file_candidate_count"]) > 0
        or int(summary["model_candidate_count"]) > 0
    )


def _db_above_maintenance_threshold(preview: dict[str, Any], threshold_gb: float) -> bool:
    if threshold_gb <= 0:
        return True
    summary = summarize_maintenance_report(preview)
    return int(summary["db_size_before_bytes"]) >= threshold_gb * 1024 * 1024 * 1024


def _confirm_maintenance(args: argparse.Namespace) -> bool:
    if args.maintenance_no_confirm:
        return True
    if not sys.stdin.isatty():
        print("非交互环境，已跳过自动清理；如需无人值守执行，请加 --maintenance-no-confirm。")
        return False
    try:
        answer = input("是否执行维护清理？输入 y 或 ok 继续: ").strip().lower()
    except EOFError:
        print("未读取到确认输入，已跳过自动清理；如需无人值守执行，请加 --maintenance-no-confirm。")
        return False
    return answer in {"y", "ok"}


def _maybe_run_maintenance(
    args: argparse.Namespace,
    current_output_root: str | Path,
    context: dict[str, Any] | None = None,
) -> None:
    if not args.auto_maintenance:
        return

    context = context or {}
    output_root = _maintenance_output_root(args, current_output_root)
    db_path = _maintenance_db_path(args, output_root)
    preview_args = _maintenance_namespace(args, output_root, db_path, execute=False)

    try:
        preview = run_maintenance(preview_args)
    except Exception as exc:
        print(f"维护检测跳过: {exc}")
        return

    if not _has_maintenance_candidates(preview):
        print("本次无需维护清理。")
        _write_maintenance_log(args, preview, None, "no_candidates", context)
        return

    print(format_maintenance_preview(preview))
    if not _db_above_maintenance_threshold(preview, args.maintenance_threshold_gb):
        print(f"SQLite 未超过维护阈值 {args.maintenance_threshold_gb:.2f} GB，暂不自动清理。")
        _write_maintenance_log(args, preview, None, "below_threshold", context)
        return

    if not _confirm_maintenance(args):
        print("已跳过维护清理。")
        _write_maintenance_log(args, preview, None, "skipped_by_user", context)
        return

    execute_args = _maintenance_namespace(args, output_root, db_path, execute=True)
    result = run_maintenance(execute_args)
    print("维护清理完成。")
    print(format_maintenance_preview(result))
    log_path = _write_maintenance_log(args, preview, result, "executed", context)
    print(f"维护日志: {log_path}")


def _single_run_id(stocks: list[str], config: dict[str, Any]) -> str:
    stock_part = "_".join(safe_filename_part(stock.replace(".", "_")) for stock in stocks[:3]) or "stocks"
    if len(stocks) > 3:
        stock_part += f"_plus{len(stocks) - 3}"
    data_cfg = config.get("data", {})
    start = str(data_cfg.get("start", "")).replace("-", "")
    end = str(data_cfg.get("end", "")).replace("-", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"single_{stock_part}_{start}_{end}_{stamp}"


def _train_and_select_for_many(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "my_strategy" / "train_and_select_models.py"),
        "--stocks",
        args.stocks,
        "--start",
        args.model_train_start,
        "--end",
        args.model_data_end or args.end or "2026-07-07",
        "--train-end",
        args.model_train_end,
        "--valid-end",
        args.model_valid_end,
        "--dl-lookback",
        str(args.dl_lookback),
        "--dl-epochs",
        str(args.dl_epochs),
        "--dl-batch-size",
        str(args.dl_batch_size),
        "--dl-hidden-size",
        str(args.dl_hidden_size),
        "--dl-max-samples",
        str(args.dl_max_samples),
        "--no-update",
    ]
    if args.max_stocks:
        cmd += ["--max-stocks", str(args.max_stocks)]
    if args.model_prefix:
        cmd += ["--model-prefix", args.model_prefix]
    _run_subprocess(cmd)


def _run_batch_backtest(args: argparse.Namespace, ml_model: str, dl_model: str) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "my_strategy" / "scripts" / "batch_multi_agent_backtest.py"),
        "--config",
        args.config,
        "--stocks",
        args.stocks,
        "--start",
        args.start or "2026-01-01",
        "--end",
        args.end or "2026-07-07",
        "--output-root",
        args.output_root or AI_AGENT_SCREEN_ROOT,
        "--top-chart-count",
        str(args.top_chart_count),
    ]
    if ml_model:
        cmd += ["--ml-model", ml_model]
    if dl_model:
        cmd += ["--dl-model", dl_model]
    if args.rl_model:
        cmd += ["--rl-model", args.rl_model]
    elif not args.no_auto_rl:
        rl_model = best_rl_model_name()
        if rl_model:
            cmd += ["--rl-model", rl_model]
    if args.max_stocks:
        cmd += ["--max-stocks", str(args.max_stocks)]
    if args.include_st:
        cmd += ["--include-st"]
    if args.no_save_daily:
        cmd += ["--no-save-daily"]
    if args.no_sql:
        cmd += ["--no-sql"]
    if args.no_rl_experience:
        cmd += ["--no-rl-experience"]
    if args.sql_db:
        cmd += ["--sql-db", args.sql_db]
    _run_subprocess(cmd)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run KHQuant-MultiAgent-V1.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "my_strategy" / "configs" / "multi_agent_config.yaml"))
    parser.add_argument("--stocks", default="", help="Comma-separated stock codes, or local/all for the local A-share universe.")
    parser.add_argument("--start", default="", help="Override start date.")
    parser.add_argument("--end", default="", help="Override end date.")
    parser.add_argument("--no-update", action="store_true", help="Read local cache only; do not incrementally update.")
    parser.add_argument("--ml-model", default="", help="Enable ML agent with model name, e.g. random_forest_future_return_5d.")
    parser.add_argument("--dl-model", default="", help="Enable deep learning agent with model name, e.g. dl_mlp_cs_rank_5d_local_all.")
    parser.add_argument("--rl-model", default="", help="Enable RL position-sizing agent with model name.")
    parser.add_argument("--no-auto-model", action="store_true", help="Do not automatically load the registered best models.")
    parser.add_argument("--no-auto-rl", action="store_true", help="Do not automatically load the registered best RL model.")
    parser.add_argument("--select-best-model", action="store_true", help="Refresh best_models.json before backtesting.")
    parser.add_argument("--force-train", action="store_true", help="Train new models before backtesting, even for a small stock list.")
    parser.add_argument("--no-train-if-many", action="store_true", help="For local/all or large lists, skip automatic training and only use the best registered models.")
    parser.add_argument("--many-threshold", type=int, default=20, help="Explicit stock-count threshold for train+batch mode.")
    parser.add_argument("--max-stocks", type=int, default=0, help="Optional cap for local/all training and batch backtest.")
    parser.add_argument("--output-root", default="", help="Output root for batch or single-stock runs.")
    parser.add_argument("--top-chart-count", type=int, default=10, help="Batch mode top chart count.")
    parser.add_argument("--include-st", action="store_true", help="Batch mode: keep ST/delisting names in candidates.")
    parser.add_argument("--no-save-daily", action="store_true", help="Batch mode: do not save per-stock daily/trade CSV files.")
    parser.add_argument("--no-sql", action="store_true", help="Do not import AI-Agent outputs into SQLite after backtesting.")
    parser.add_argument("--sql-db", default="", help="Optional SQLite database path; defaults to <output-root>/ai_agent_screen.db.")
    parser.add_argument("--no-rl-experience", action="store_true", help="Do not refresh rl_experiences after SQLite import.")
    parser.add_argument("--model-train-start", default="2021-01-01", help="Training dataset start date for automatic many-stock training.")
    parser.add_argument("--model-data-end", default="", help="Training dataset end date; defaults to --end.")
    parser.add_argument("--model-train-end", default="2024-12-31", help="Training split end date.")
    parser.add_argument("--model-valid-end", default="2025-12-31", help="Validation split end date.")
    parser.add_argument("--model-prefix", default="", help="Optional model-name prefix for automatic training.")
    parser.add_argument("--dl-lookback", type=int, default=20)
    parser.add_argument("--dl-epochs", type=int, default=12)
    parser.add_argument("--dl-batch-size", type=int, default=4096)
    parser.add_argument("--dl-hidden-size", type=int, default=64)
    parser.add_argument("--dl-max-samples", type=int, default=160000)
    parser.add_argument("--auto-maintenance", dest="auto_maintenance", action="store_true", default=True, help="Run maintenance preview after backtest.")
    parser.add_argument("--no-auto-maintenance", dest="auto_maintenance", action="store_false", help="Disable post-backtest maintenance preview.")
    parser.add_argument("--maintenance-no-confirm", action="store_true", help="Execute maintenance without y/ok confirmation when needed.")
    parser.add_argument("--maintenance-threshold-gb", type=float, default=1.0, help="Only execute confirmed cleanup when SQLite is at least this large; 0 means always.")
    parser.add_argument("--maintenance-output-root", default="", help="Override output root scanned by maintenance.")
    parser.add_argument("--maintenance-db-path", default="", help="Override SQLite database maintained after backtest.")
    parser.add_argument("--maintenance-processed-root", default="my_strategy/data/processed")
    parser.add_argument("--maintenance-log-dir", default="my_strategy/data/processed/maintenance_logs")
    parser.add_argument("--maintenance-vacuum", dest="maintenance_vacuum", action="store_true", default=True)
    parser.add_argument("--maintenance-no-vacuum", dest="maintenance_vacuum", action="store_false")
    parser.add_argument("--maintenance-delete-files", dest="maintenance_delete_files", action="store_true", default=True)
    parser.add_argument("--maintenance-no-delete-files", dest="maintenance_delete_files", action="store_false")
    parser.add_argument("--maintenance-delete-models", dest="maintenance_delete_models", action="store_true", default=True)
    parser.add_argument("--maintenance-no-delete-models", dest="maintenance_delete_models", action="store_false")
    parser.add_argument("--maintenance-include-legacy-backtests", action="store_true")
    parser.add_argument("--maintenance-keep-latest-runs", type=int, default=3)
    parser.add_argument("--maintenance-keep-detail-runs", type=int, default=1)
    parser.add_argument("--maintenance-keep-summary-runs", type=int, default=10)
    parser.add_argument("--maintenance-keep-rl-experience-runs", type=int, default=1)
    parser.add_argument("--maintenance-keep-run-logs", type=int, default=10)
    parser.add_argument("--maintenance-keep-figures", type=int, default=80)
    parser.add_argument("--maintenance-keep-models-per-kind", type=int, default=2)
    parser.add_argument("--maintenance-keep-top-models-per-kind", type=int, default=3)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    fusion_loaded = apply_best_fusion_policy(config)
    if fusion_loaded:
        print("Auto fusion policy: loaded best_fusion_policy.json")
    auto_model = not args.no_auto_model
    if args.select_best_model or auto_model:
        refresh_best_models()

    if args.stocks and _is_local_universe(args.stocks):
        stocks, _ = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    elif args.stocks:
        stocks = _split_stock_arg(args.stocks)
    else:
        stocks = list(config.get("data", {}).get("stocks", []))

    many_mode = _is_many_stock_request(args.stocks, stocks, threshold=args.many_threshold)
    if many_mode and not args.stocks:
        args.stocks = ",".join(stocks)
    should_train = args.force_train or (many_mode and not args.no_train_if_many)

    if should_train:
        try:
            _train_and_select_for_many(args)
            refresh_best_models()
        except subprocess.CalledProcessError as exc:
            print(f"Auto training failed; fallback to current best registered models. exit_code={exc.returncode}", flush=True)

    ml_model, dl_model = _apply_best_models(
        config,
        explicit_ml=args.ml_model,
        explicit_dl=args.dl_model,
        explicit_rl=args.rl_model,
        auto_model=auto_model,
        auto_rl=not args.no_auto_rl,
    )

    if many_mode:
        if not args.stocks:
            args.stocks = ",".join(stocks)
        batch_output_root = args.output_root or AI_AGENT_SCREEN_ROOT
        _run_batch_backtest(args, ml_model, dl_model)
        _maybe_run_maintenance(
            args,
            current_output_root=batch_output_root,
            context={
                "mode": "many",
                "stocks": args.stocks,
                "start": args.start or "2026-01-01",
                "end": args.end or "2026-07-07",
                "ml_model": ml_model,
                "dl_model": dl_model,
            },
        )
        return

    if args.stocks:
        config.setdefault("data", {})["stocks"] = stocks
    if args.start:
        config.setdefault("data", {})["start"] = args.start
    if args.end:
        config.setdefault("data", {})["end"] = args.end
    if args.no_update:
        config.setdefault("data", {})["auto_update"] = False
    if args.output_root:
        config.setdefault("outputs", {})["root_dir"] = args.output_root

    rl_model = config.get("agents", {}).get("reinforcement_learning", {}).get("model_name", "")
    if ml_model or dl_model or rl_model:
        print(f"Auto models: ML={ml_model or '-'} | DL={dl_model or '-'} | RL={rl_model or '-'}")

    for stock in config["data"].get("stocks", []):
        run_stock(stock, config)

    if not args.no_sql:
        output_root = PROJECT_ROOT / config.get("outputs", {}).get("root_dir", AI_AGENT_SCREEN_ROOT)
        sql_run_id = ""
        try:
            sql_result = import_screen_outputs(
                output_root=output_root,
                db_path=args.sql_db or None,
                run_id=_single_run_id(config["data"].get("stocks", []), config),
                include_details=True,
                stocks=config["data"].get("stocks", []),
            )
            sql_run_id = str(sql_result.get("run_id", ""))
            print(f"SQLite imported: {sql_result['db_path']} run_id={sql_result['run_id']}")
            if not args.no_rl_experience:
                rl_info = build_rl_experiences(db_path=sql_result["db_path"], source_run_id=sql_result["run_id"])
                print(f"RL experiences refreshed: rows={rl_info['rows']} source_run_id={rl_info['source_run_id']}")
        except Exception as exc:
            print(f"SQLite import skipped: {exc}")
        _maybe_run_maintenance(
            args,
            current_output_root=output_root,
            context={
                "mode": "single",
                "stocks": ",".join(config["data"].get("stocks", [])),
                "start": config.get("data", {}).get("start", ""),
                "end": config.get("data", {}).get("end", ""),
                "run_id": sql_run_id,
                "ml_model": ml_model,
                "dl_model": dl_model,
                "rl_model": rl_model,
            },
        )
    else:
        output_root = PROJECT_ROOT / config.get("outputs", {}).get("root_dir", AI_AGENT_SCREEN_ROOT)
        _maybe_run_maintenance(
            args,
            current_output_root=output_root,
            context={
                "mode": "single_no_sql",
                "stocks": ",".join(config["data"].get("stocks", [])),
                "start": config.get("data", {}).get("start", ""),
                "end": config.get("data", {}).get("end", ""),
                "ml_model": ml_model,
                "dl_model": dl_model,
                "rl_model": rl_model,
            },
        )


if __name__ == "__main__":
    main()
