#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified backtest engine for strategy lab candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(df: pd.DataFrame, signals: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = config.get("backtest", {})
    initial_cash = float(bt.get("initial_cash", 100000))
    commission = float(bt.get("commission", 0.0003))
    stamp_tax = float(bt.get("stamp_tax", 0.0005))
    slippage = float(bt.get("slippage", 0.0005))
    lot_size = int(bt.get("lot_size", 100))
    rebalance_interval_days = int(bt.get("rebalance_interval_days", 5))
    min_weight_change = float(bt.get("min_weight_change", 0.08))

    price_cols = ["date", "open", "high", "low", "close", "volume"]
    if "trade_close" in df.columns:
        price_cols.append("trade_close")
    merged = df[price_cols].merge(
        signals[["date", "strategy_name", "score", "target_weight", "final_action", "reason"]],
        on="date",
        how="inner",
    )
    cash = initial_cash
    shares = 0
    avg_cost = np.nan
    last_trade_date = None
    daily_rows = []
    trades = []

    for _, row in merged.iterrows():
        signal_close = float(row["close"])
        close = float(row["trade_close"]) if "trade_close" in row and pd.notna(row["trade_close"]) else signal_close
        date = row["date"]
        equity_before = cash + shares * close
        current_weight = shares * close / equity_before if equity_before else 0.0
        target_weight = float(row["target_weight"])
        if row["final_action"] == "sell":
            target_weight = 0.0

        days_since_trade = None
        if last_trade_date is not None:
            days_since_trade = (pd.to_datetime(date) - pd.to_datetime(last_trade_date)).days
        can_rebalance = last_trade_date is None or days_since_trade is None or days_since_trade >= rebalance_interval_days
        if abs(target_weight - current_weight) < min_weight_change:
            target_weight = current_weight
        if not can_rebalance and target_weight > 0:
            target_weight = current_weight

        target_value = equity_before * target_weight
        current_value = shares * close
        diff_value = target_value - current_value

        if diff_value > close * lot_size:
            price = close * (1 + slippage)
            buy_shares = int(diff_value / (price * lot_size)) * lot_size
            if buy_shares > 0:
                cost = buy_shares * price
                fee = max(cost * commission, 5)
                if cost + fee <= cash:
                    before = shares
                    old_cost_basis = 0 if shares == 0 or pd.isna(avg_cost) else avg_cost * shares
                    cash -= cost + fee
                    shares += buy_shares
                    avg_cost = (old_cost_basis + cost + fee) / shares if shares > 0 else np.nan
                    last_trade_date = date
                    trades.append({
                        "date": date,
                        "strategy_name": row["strategy_name"],
                        "action": "BUY",
                        "price": price,
                        "shares": buy_shares,
                        "position_before": before,
                        "position_after": shares,
                        "cash_flow": -(cost + fee),
                        "fee": fee,
                        "holding_cost_after": avg_cost,
                        "reason": row["reason"],
                    })

        elif diff_value <= -close * lot_size and shares > 0:
            price = close * (1 - slippage)
            sell_shares = min(int(abs(diff_value) / (price * lot_size)) * lot_size, shares)
            if target_weight == 0:
                sell_shares = shares
            if sell_shares > 0:
                before = shares
                proceeds = sell_shares * price
                fee = max(proceeds * commission, 5)
                tax = proceeds * stamp_tax
                cash += proceeds - fee - tax
                shares -= sell_shares
                if shares == 0:
                    avg_cost = np.nan
                last_trade_date = date
                trades.append({
                    "date": date,
                    "strategy_name": row["strategy_name"],
                    "action": "SELL" if shares == 0 else "REDUCE",
                    "price": price,
                    "shares": sell_shares,
                    "position_before": before,
                    "position_after": shares,
                    "cash_flow": proceeds - fee - tax,
                    "fee": fee + tax,
                    "holding_cost_after": avg_cost,
                    "reason": row["reason"],
                })

        equity = cash + shares * close
        daily_rows.append({
            "date": date,
            "strategy_name": row["strategy_name"],
            "close": close,
            "factor_close": signal_close,
            "cash": cash,
            "shares": shares,
            "equity": equity,
            "position_weight": shares * close / equity if equity else 0,
            "holding_cost": avg_cost,
            "avg_cost": avg_cost,
            "unrealized_return": close / avg_cost - 1 if shares > 0 and not pd.isna(avg_cost) and avg_cost else np.nan,
            "score": row["score"],
            "target_weight": row["target_weight"],
            "final_action": row["final_action"],
        })

    return pd.DataFrame(daily_rows), pd.DataFrame(trades)


def metrics(daily: pd.DataFrame, trades: pd.DataFrame, initial_cash: float) -> dict:
    if daily.empty:
        return {}
    equity = daily.set_index("date")["equity"]
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_cash - 1
    days = max((pd.to_datetime(equity.index[-1]) - pd.to_datetime(equity.index[0])).days, 1)
    annual_return = (equity.iloc[-1] / initial_cash) ** (365 / days) - 1
    drawdown = equity / equity.cummax() - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0.0

    paired = 0
    wins = 0
    if not trades.empty:
        buys = trades[trades["action"] == "BUY"].reset_index(drop=True)
        sells = trades[trades["action"].isin(["SELL", "REDUCE"])].reset_index(drop=True)
        paired = min(len(buys), len(sells))
        for i in range(paired):
            if sells.loc[i, "cash_flow"] + buys.loc[i, "cash_flow"] > 0:
                wins += 1

    buy_hold = daily["close"].iloc[-1] / daily["close"].iloc[0] - 1
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": float(sharpe),
        "n_trades": int(len(trades)),
        "paired_trades": int(paired),
        "win_rate": float(wins / paired) if paired else 0.0,
        "buy_hold_return": float(buy_hold),
        "excess_return": float(total_return - buy_hold),
        "final_equity": float(equity.iloc[-1]),
        "days": int(days),
    }
