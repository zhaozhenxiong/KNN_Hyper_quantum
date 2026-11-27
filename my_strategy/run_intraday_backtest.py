#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run KHQuant intraday backtests on local minute bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.charting import plot_intraday_backtest_chart  # noqa: E402
from my_strategy.data_manager import MinuteDataCenter  # noqa: E402
from my_strategy.data_manager.minute_data import normalize_period  # noqa: E402
from my_strategy.run_multi_agent_backtest import resolve_stock_name  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


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


def add_intraday_factors(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config.get("intraday_strategy", {})
    fast = int(cfg.get("fast_ma", 6))
    slow = int(cfg.get("slow_ma", 24))
    rsi_window = int(cfg.get("rsi_window", 14))
    volume_window = int(cfg.get("volume_window", 20))

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    out["date"] = out["datetime"]
    out["trade_date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["ma_fast"] = out["close"].rolling(fast, min_periods=fast).mean()
    out["ma_slow"] = out["close"].rolling(slow, min_periods=slow).mean()
    out["rsi"] = calc_rsi(out["close"], rsi_window)
    out["volume_ma"] = out["volume"].rolling(volume_window, min_periods=1).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma"].replace(0, np.nan)
    pv = out["close"] * out["volume"].fillna(0)
    out["vwap"] = pv.groupby(out["trade_date"]).cumsum() / out["volume"].fillna(0).groupby(out["trade_date"]).cumsum().replace(0, np.nan)
    out["vwap"] = out["vwap"].fillna(out["close"])
    return out.dropna(subset=["ma_fast", "ma_slow", "rsi"]).reset_index(drop=True)


def calc_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def generate_intraday_signals(factors: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config.get("intraday_strategy", {})
    buy_threshold = float(cfg.get("buy_threshold", 0.35))
    exit_threshold = float(cfg.get("exit_threshold", 0.05))
    min_entry_weight = float(cfg.get("min_entry_weight", 0.30))
    max_weight = float(cfg.get("max_weight", 0.80))

    rows = []
    for _, row in factors.iterrows():
        score = 0.0
        reasons = []
        close = float(row["close"])
        ma_fast = float(row["ma_fast"])
        ma_slow = float(row["ma_slow"])
        rsi = float(row["rsi"])
        vwap = float(row["vwap"])
        volume_ratio = float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 1.0

        if close > ma_fast > ma_slow:
            score += 0.45
            reasons.append("价格在快慢均线上方")
        elif close > ma_fast:
            score += 0.15
            reasons.append("价格在快线上方")
        elif ma_fast < ma_slow:
            score -= 0.25
            reasons.append("快线低于慢线")

        if close > vwap:
            score += 0.15
            reasons.append("价格强于VWAP")
        else:
            score -= 0.10
            reasons.append("价格弱于VWAP")

        if rsi < 35:
            score += 0.15
            reasons.append(f"RSI={rsi:.1f}偏低")
        elif rsi > 80:
            score -= 0.25
            reasons.append(f"RSI={rsi:.1f}过热")
        elif 45 <= rsi <= 70:
            score += 0.05
            reasons.append(f"RSI={rsi:.1f}中性偏强")

        if volume_ratio > 1.5:
            score += 0.15
            reasons.append(f"量比={volume_ratio:.2f}放大")
        elif volume_ratio < 0.5:
            score -= 0.05
            reasons.append(f"量比={volume_ratio:.2f}萎缩")

        score = float(max(-1.0, min(1.0, score)))
        if score >= buy_threshold:
            ratio = min(1.0, max(0.0, (score - buy_threshold) / max(1e-9, 1 - buy_threshold)))
            target_weight = min_entry_weight + ratio * (max_weight - min_entry_weight)
            action = "buy"
        elif score <= exit_threshold:
            target_weight = 0.0
            action = "sell"
        else:
            target_weight = 0.0
            action = "watch"

        rows.append({
            "date": row["datetime"],
            "close": close,
            "score": score,
            "target_weight": target_weight,
            "final_action": action,
            "reason": "；".join(reasons),
        })
    return pd.DataFrame(rows)


def simulate_next_bar_execution(factors: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = config.get("backtest", {})
    risk = config.get("risk", {})
    initial_cash = float(bt.get("initial_cash", 500000))
    commission = float(bt.get("commission", 0.0003))
    stamp_tax = float(bt.get("stamp_tax", 0.0005))
    slippage = float(bt.get("slippage", 0.0005))
    lot_size = int(bt.get("lot_size", 1))
    min_weight_change = float(bt.get("min_weight_change", 0.05))
    stop_loss = float(risk.get("stop_loss", 0.04))
    take_profit = float(risk.get("take_profit", 0.12))

    merged = factors.merge(signals[["date", "score", "target_weight", "final_action", "reason"]], left_on="datetime", right_on="date", how="inner", suffixes=("", "_signal"))
    merged["exec_target_weight"] = merged["target_weight"].shift(1).fillna(0.0)
    merged["exec_action"] = merged["final_action"].shift(1).fillna("watch")
    merged["exec_reason"] = merged["reason"].shift(1).fillna("")

    cash = initial_cash
    shares = 0
    avg_cost = np.nan
    trades = []
    daily_rows = []

    for _, row in merged.iterrows():
        exec_price = float(row["open"])
        mark_price = float(row["close"])
        equity_before = cash + shares * exec_price
        current_weight = shares * exec_price / equity_before if equity_before else 0.0
        target_weight = float(row["exec_target_weight"])
        exec_action = row["exec_action"]

        if shares > 0 and not pd.isna(avg_cost) and avg_cost > 0:
            pnl = mark_price / avg_cost - 1
            if pnl <= -stop_loss:
                target_weight = 0.0
                exec_action = "stop_loss"
            elif pnl >= take_profit:
                target_weight = min(target_weight, 0.30)
                exec_action = "take_profit"

        if exec_action in {"sell", "stop_loss"}:
            target_weight = 0.0
        if abs(target_weight - current_weight) < min_weight_change and exec_action not in {"sell", "stop_loss"}:
            target_weight = current_weight

        target_value = equity_before * target_weight
        current_value = shares * exec_price
        diff_value = target_value - current_value

        if diff_value > exec_price * lot_size:
            trade_price = exec_price * (1 + slippage)
            buy_shares = int(diff_value / (trade_price * lot_size)) * lot_size
            if buy_shares > 0:
                cost = buy_shares * trade_price
                fee = max(cost * commission, 5)
                if cost + fee <= cash:
                    before = shares
                    old_cost_basis = 0 if shares == 0 or pd.isna(avg_cost) else avg_cost * shares
                    cash -= cost + fee
                    shares += buy_shares
                    avg_cost = (old_cost_basis + cost + fee) / shares
                    trades.append({
                        "date": row["datetime"],
                        "action": "BUY",
                        "price": trade_price,
                        "shares": buy_shares,
                        "position_before": before,
                        "position_after": shares,
                        "cash_flow": -(cost + fee),
                        "fee": fee,
                        "holding_cost_after": avg_cost,
                        "reason": row["exec_reason"],
                    })
        elif diff_value <= -exec_price * lot_size and shares > 0:
            trade_price = exec_price * (1 - slippage)
            sell_shares = min(int(abs(diff_value) / (trade_price * lot_size)) * lot_size, shares)
            if target_weight == 0:
                sell_shares = shares
            if sell_shares > 0:
                before = shares
                proceeds = sell_shares * trade_price
                fee = max(proceeds * commission, 5)
                tax = proceeds * stamp_tax
                cash += proceeds - fee - tax
                shares -= sell_shares
                if shares == 0:
                    avg_cost = np.nan
                trades.append({
                    "date": row["datetime"],
                    "action": "SELL" if shares == 0 else "REDUCE",
                    "price": trade_price,
                    "shares": sell_shares,
                    "position_before": before,
                    "position_after": shares,
                    "cash_flow": proceeds - fee - tax,
                    "fee": fee + tax,
                    "holding_cost_after": avg_cost,
                    "reason": row["exec_reason"],
                })

        equity = cash + shares * mark_price
        daily_rows.append({
            "date": row["datetime"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": mark_price,
            "volume": row["volume"],
            "vwap": row["vwap"],
            "cash": cash,
            "shares": shares,
            "equity": equity,
            "position_weight": shares * mark_price / equity if equity else 0,
            "holding_cost": avg_cost,
            "avg_cost": avg_cost,
            "unrealized_return": mark_price / avg_cost - 1 if shares > 0 and not pd.isna(avg_cost) and avg_cost else np.nan,
            "score": row["score"],
            "target_weight": row["target_weight"],
            "effective_target_weight": target_weight,
            "final_action": row["final_action"],
        })

    return pd.DataFrame(daily_rows), pd.DataFrame(trades)


def calc_metrics(equity: pd.Series, initial_cash: float, trades: pd.DataFrame) -> dict[str, Any]:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_cash - 1
    span_days = max((pd.to_datetime(equity.index[-1]) - pd.to_datetime(equity.index[0])).days, 1)
    annual_return = (equity.iloc[-1] / initial_cash) ** (365 / span_days) - 1
    drawdown = equity / equity.cummax() - 1
    bars_per_year = 252 * max(1, int(len(equity) / max(span_days, 1)))
    sharpe = np.sqrt(bars_per_year) * returns.mean() / returns.std() if returns.std() > 0 else 0.0

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
        "bars": int(len(equity)),
        "days": int(span_days),
        "final_equity": float(equity.iloc[-1]),
    }


def save_outputs(stock: str, period: str, bars: pd.DataFrame, signals: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    output_root = PROJECT_ROOT / config.get("outputs", {}).get("root_dir", "my_strategy/data/processed/intraday_backtest")
    dirs = ensure_output_dirs(output_root)
    tag = f"{stock.replace('.', '_')}_{period}m_intraday"
    signals_path = dirs["signals"] / f"{tag}_signals.csv"
    daily_path = dirs["signals"] / f"{tag}_daily.csv"
    trades_path = dirs["signals"] / f"{tag}_trades.csv"
    report_path = dirs["reports"] / f"{tag}_report.json"
    figure_path = dirs["figures"] / f"{tag}_equity.png"

    signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
    bars.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    equity = bars.set_index("date")["equity"]
    metrics = calc_metrics(equity, float(config.get("backtest", {}).get("initial_cash", 500000)), trades)
    stock_name = resolve_stock_name(stock)
    title_name = f"{stock} {stock_name}".strip()
    plot_intraday_backtest_chart(f"{title_name} - {period}m - {config.get('strategy_name', 'KHQuant-Intraday-V1')}", bars, trades, figure_path)
    latest = signals.iloc[-1].to_dict() if not signals.empty else {}
    report = {
        "strategy_name": config.get("strategy_name", "KHQuant-Intraday-V1"),
        "stock": stock,
        "stock_name": stock_name,
        "period": f"{period}m",
        "metrics": metrics,
        "latest_decision": {
            "date": str(latest.get("date"))[:19],
            "final_action": latest.get("final_action"),
            "score": latest.get("score"),
            "target_weight": latest.get("target_weight"),
            "reason": latest.get("reason"),
        },
        "files": {
            "signals": str(signals_path),
            "daily": str(daily_path),
            "trades": str(trades_path),
            "figure": str(figure_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"signals": signals_path, "daily": daily_path, "trades": trades_path, "report": report_path, "figure": figure_path}


def run_stock(stock: str, config: dict[str, Any]) -> dict[str, Path]:
    data_cfg = config.get("data", {})
    period = normalize_period(data_cfg.get("period", "5"))
    start = data_cfg.get("start")
    end = data_cfg.get("end")
    auto_update = bool(data_cfg.get("auto_update", True))

    print(f"📦 Minute Data Center: {stock} {period}m auto_update={auto_update}")
    raw = MinuteDataCenter().get_stock_minute(stock, period, start, end, auto_update=auto_update)
    factors = add_intraday_factors(raw, config)
    if factors.empty:
        raise RuntimeError(f"{stock} {period}m 在 {start} ~ {end} 没有足够分时因子数据")
    signals = generate_intraday_signals(factors, config)
    bars, trades = simulate_next_bar_execution(factors, signals, config)
    paths = save_outputs(stock, period, bars, signals, trades, config)

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    metrics = report["metrics"]
    latest = report["latest_decision"]
    print("\n" + "=" * 72)
    print(f"{config.get('strategy_name')} / {stock} / {period}m")
    print(f"总收益: {metrics['total_return']:.2%} | 年化: {metrics['annual_return']:.2%} | 最大回撤: {metrics['max_drawdown']:.2%} | 夏普: {metrics['sharpe_ratio']:.2f}")
    print(f"交易次数: {metrics['n_trades']} | 胜率: {metrics['win_rate']:.1%} | bars={metrics['bars']}")
    print(f"最新决策: {latest['date']} {latest['final_action']} score={latest['score']} weight={latest['target_weight']}")
    print(f"图表: {paths['figure']}")
    return paths


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run KHQuant intraday backtest.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "my_strategy" / "configs" / "intraday_config.yaml"))
    parser.add_argument("--stocks", default="", help="Comma-separated stock codes.")
    parser.add_argument("--period", default="", help="1, 5, 15, 30, or 60.")
    parser.add_argument("--start", default="", help="Start date or datetime.")
    parser.add_argument("--end", default="", help="End date or datetime.")
    parser.add_argument("--no-update", action="store_true", help="Read local cache only.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.stocks:
        config.setdefault("data", {})["stocks"] = [item.strip() for item in args.stocks.split(",") if item.strip()]
    if args.period:
        config.setdefault("data", {})["period"] = args.period
    if args.start:
        config.setdefault("data", {})["start"] = args.start
    if args.end:
        config.setdefault("data", {})["end"] = args.end
    if args.no_update:
        config.setdefault("data", {})["auto_update"] = False

    for stock in config.get("data", {}).get("stocks", []):
        run_stock(stock, config)


if __name__ == "__main__":
    main()
