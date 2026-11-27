"""Portfolio simulation for signals formed after the market close."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import floor, isfinite, sqrt
from typing import Any

import pandas as pd

from my_strategy.execution import BrokerSimulator, CostModel, Order


@dataclass(frozen=True)
class Strategy:
    top_n: int = 10
    rebalance_days: int = 5
    score_threshold: float = 0.70
    max_weight: float = 0.15
    max_industry_weight: float = 0.35


def _positive(value: Any) -> bool:
    return value is not None and pd.notna(value) and isfinite(float(value)) and float(value) > 0


def target_weights(rows: pd.DataFrame, strategy: Strategy) -> dict[str, float]:
    ranks = rows["score"].rank(pct=True)
    candidates = rows.loc[ranks >= strategy.score_threshold].sort_values(
        ["score", "stock"], ascending=[False, True]
    )
    target = {}
    exposure: dict[str, float] = {}
    weight = min(1 / strategy.top_n, strategy.max_weight)
    for row in candidates.itertuples(index=False):
        group = getattr(row, "industry", "unknown")
        known = pd.notna(group) and group not in ("", "unknown", "unclassified")
        remaining = strategy.max_industry_weight - exposure.get(group, 0.0) if known else weight
        size = min(weight, remaining)
        if size <= 1e-12:
            continue
        target[row.stock] = size
        if known:
            exposure[group] = exposure.get(group, 0.0) + size
        if len(target) == strategy.top_n:
            break
    return target


def _blocked(row: Any, old_close: float | None, side: str, board: str) -> str | None:
    if row is None or not _positive(row.trade_open) or not getattr(row, "has_trade_price", True):
        return "no_open"
    if not _positive(old_close):
        return "no_previous_close"
    limit_flag = getattr(row, "limit_up" if side == "BUY" else "limit_down", None)
    if limit_flag is not None and pd.notna(limit_flag):
        return "price_limit" if limit_flag else None
    limit = 0.195 if board in ("star", "chinext") else 0.295 if board == "bse" else 0.095
    move = row.trade_open / old_close - 1
    return "approximate_price_limit" if (move >= limit if side == "BUY" else move <= -limit) else None


def simulate(
    frame: pd.DataFrame,
    scores: pd.Series,
    sessions: pd.DatetimeIndex,
    benchmark_returns: pd.Series,
    strategy: Strategy,
    *,
    run_id: str,
    cost_multiplier: float = 1.0,
    initial_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    """Simulate one shared cash account, using the immediately preceding session's scores."""
    if len(sessions) < 2:
        raise ValueError("Portfolio simulation needs at least two trading sessions.")
    rows = frame.loc[frame["date"].isin(sessions)].copy()
    rows["score"] = scores.reindex(rows.index)
    day_rows = {day: group for day, group in rows.groupby("date", sort=False)}
    cost = CostModel()
    broker = BrokerSimulator(
        initial_cash=initial_cash,
        cost_model=CostModel(
            commission=cost.commission * cost_multiplier,
            stamp_tax=cost.stamp_tax,
            slippage=cost.slippage * cost_multiplier,
            min_commission=cost.min_commission * cost_multiplier,
        ),
        run_id=run_id,
    )
    targets: dict[str, float] = {}
    previous: dict[str, Any] = {}
    marks: dict[str, float] = {}
    records = []
    rejected = []
    benchmark = initial_cash
    benchmark_complete = True
    order_dates: dict[str, str] = {}
    fills = 0.0
    last_rebalance = -strategy.rebalance_days

    for index, day in enumerate(sessions):
        group = day_rows.get(day)
        current = {} if group is None else {row.stock: row for row in group.itertuples(index=False)}
        signal_day = sessions[index - 1] if index else None
        signals = day_rows.get(signal_day)
        if signals is not None and index - last_rebalance >= strategy.rebalance_days:
            known = signals.dropna(subset=["score"])
            if not known.empty:
                targets = target_weights(known, strategy)
                order_dates = {stock: str(signal_day.date()) for stock in set(targets) | set(broker.positions)}
                last_rebalance = index

        if index == last_rebalance:
            opening_marks = dict(marks)
            opening_marks.update({s: r.trade_open for s, r in current.items() if _positive(r.trade_open)})
            equity_open = broker.equity(opening_marks)
            intentions = []
            for stock in sorted(set(targets) | {s for s, shares in broker.positions.items() if shares}):
                row = current.get(stock)
                old = previous.get(stock)
                mark = opening_marks.get(stock)
                weight = targets.get(stock, 0.0)
                shares = broker.shares(stock)
                if not _positive(mark):
                    rejected.append(_rejection(day, stock, "BUY" if weight else "SELL", "no_open"))
                    continue
                desired = floor(equity_open * weight / mark / 100) * 100
                change = desired - shares
                if change:
                    intentions.append((change > 0, stock, change, row, old, weight))
            for buy, stock, change, row, old, weight in sorted(intentions):
                side = "BUY" if buy else "SELL"
                board = str(getattr(old, "board", "unknown")).lower()
                prior_close = getattr(row, "prev_trade_close", getattr(old, "trade_close", None))
                reason = _blocked(row, prior_close, side, board)
                if reason:
                    rejected.append(_rejection(day, stock, side, reason))
                    continue
                shares = abs(change)
                amount = getattr(old, "amount", None)
                if getattr(old, "date", None) != signal_day or not _positive(amount):
                    rejected.append(_rejection(day, stock, side, "no_previous_session_liquidity"))
                    continue
                basis = row.trade_open
                price = broker.cost_model.execution_price(side, basis)
                size_limit = floor(float(amount) * 0.02 / price / 100) * 100
                if buy:
                    fee_buffer = max(broker.cost_model.min_commission, broker.cash * broker.cost_model.commission)
                    affordable = max(0, floor((broker.cash - fee_buffer) / price / 100) * 100)
                    group = getattr(old, "industry", "unknown")
                    if pd.notna(group) and group not in ("", "unknown", "unclassified"):
                        group_value = sum(
                            held * opening_marks[symbol]
                            for symbol, held in broker.positions.items()
                            if held and getattr(previous.get(symbol), "industry", None) == group
                        )
                        room = max(0.0, equity_open * strategy.max_industry_weight - group_value)
                        industry_limit = floor(room / basis / 100) * 100
                        size_limit = min(size_limit, industry_limit)
                    shares = min(shares, affordable, size_limit)
                    minimum = 200 if board == "star" and not broker.shares(stock) else 100
                else:
                    shares = min(shares, size_limit)
                    minimum = min(100, broker.shares(stock))
                if shares < minimum:
                    rejected.append(_rejection(day, stock, side, "cash_or_liquidity_lot"))
                    continue
                signal_date = order_dates[stock]
                entry = broker.submit(
                    Order(str(day.date()), stock, side, shares, price, f"signal_date={signal_date}", weight),
                    mark_price=basis,
                    equity_before=equity_open,
                )
                if entry is None:
                    rejected.append(_rejection(day, stock, side, "broker_rejected"))
                else:
                    fills += shares * price / equity_open
                    if shares < abs(change):
                        rejected.append(_rejection(day, stock, side, "partial_fill"))

        for stock, row in current.items():
            if _positive(row.trade_close) and getattr(row, "has_trade_price", True):
                marks[stock] = row.trade_close
                previous[stock] = row

        if index:
            benchmark_return = benchmark_returns.get(day)
            benchmark_complete &= pd.notna(benchmark_return)
            if benchmark_complete:
                benchmark *= 1 + float(benchmark_return)
        records.append(
            {
                "date": str(day.date()),
                "equity": broker.equity(marks),
                "cash": broker.cash,
                "benchmark": benchmark if benchmark_complete else None,
            }
        )

    equity = pd.Series([row["equity"] for row in records])
    daily_returns = equity.pct_change().dropna()
    deviation = float(daily_returns.std(ddof=0))
    trades = broker.ledger.to_frame()
    final_benchmark = records[-1]["benchmark"]
    total_return = float(equity.iloc[-1] / initial_cash - 1)
    benchmark_return = final_benchmark / initial_cash - 1 if final_benchmark is not None else None
    metrics = {
        "total_return": total_return,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "sharpe": float(daily_returns.mean() / deviation * sqrt(252)) if deviation else 0.0,
        "turnover": fills,
        "trade_count": len(trades),
        "fees": float(trades["fee"].sum()) if len(trades) else 0.0,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return if benchmark_return is not None else None,
        "rejected_order_count": len(rejected),
        "rejected_by_reason": dict(Counter(item["reason"] for item in rejected)),
    }
    ledger = trades.to_dict("records")
    for trade in ledger:
        trade["signal_date"] = trade["reason"].removeprefix("signal_date=")
    return {"metrics": metrics, "equity": records, "trades": ledger, "rejected": rejected}


def _rejection(day: pd.Timestamp, stock: str, side: str, reason: str) -> dict[str, str]:
    return {"date": str(day.date()), "stock": stock, "side": side, "reason": reason}
