"""Read-only holding returns derived from an authorized portfolio replay."""

from __future__ import annotations

from math import fsum, isclose, isfinite
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd

from . import expert_backtest_activity as activity, expert_personal_replay as personal, expert_replay as replay, latent_api


_ACCOUNTS = ("full_pool", "watchlist")


def _query(query: str) -> str:
    try:
        values = parse_qs(query, max_num_fields=1)
        if set(values) - {"account"} or any(len(items) != 1 for items in values.values()):
            raise ValueError("Invalid holdings filters")
        account = values.get("account", ["full_pool"])[0].strip()
        if account not in _ACCOUNTS:
            raise ValueError("Invalid holdings account")
        return account
    except ValueError as error:
        raise latent_api.RouteError(400, "invalid_holdings_filter", "账户或查询参数不合法。") from error


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _day(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.to_datetime(value, errors="raise").strftime("%Y-%m-%d")


def _ledgers(root: Path, report: dict, account: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost = report["source"]["cost"]
    tables = []
    for kind in ("daily", "positions", "trades"):
        file_id = f"custom--{account}_{cost}_{kind}.parquet"
        path = replay._file(root, personal.authorized_download(report, file_id))
        table = pd.read_parquet(path)
        table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
        tables.append(table)
    return tuple(tables)


def _visible(daily: pd.DataFrame, positions: pd.DataFrame, trades: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage = report.get("coverage", {})
    begin = coverage.get("requested_start") or report["window"]["start"]
    end = coverage.get("requested_end") or report["window"]["end"]
    visible = {day for day in daily["date"] if begin <= day <= end}
    requested = report["source"].get("requested_model_sessions")
    if requested is not None:
        visible.intersection_update(requested)
    return tuple(frame.loc[frame["date"].isin(visible)].copy() for frame in (daily, positions, trades))


def _holding(position: pd.Series, history: pd.DataFrame) -> dict:
    stock = str(position["stock"])
    shares = _number(position.get("shares"))
    mark_price = _number(position.get("mark_price"))
    market_value = _number(position.get("market_value"))
    state = str(position.get("position_state") or "")
    row = {
        "stock": stock,
        "board": str(position.get("board") or ""),
        "industry": str(position.get("industry") or ""),
        "shares": int(shares) if shares is not None and shares.is_integer() else shares,
        "mark_price": mark_price,
        "mark_date": _day(position.get("mark_date")),
        "market_value": market_value,
        "market_status": str(position.get("market_status") or ""),
        "position_state": state,
        "detached_on": _day(position.get("detached_on")),
        "mark_is_stale": bool(position.get("mark_is_stale")),
        "stranded_delisted": state == "stranded_delisted",
        "cost_per_share": None,
        "cost_basis": None,
        "unrealized_pnl": None,
        "unrealized_return": None,
        "realized_pnl": None,
        "total_pnl": None,
        "zero_recovery_pnl": None,
        "last_trade_date": None,
        "last_target_weight": None,
        "status": "unavailable",
        "reason": "",
    }
    if shares is None or shares <= 0 or mark_price is None or mark_price < 0 or market_value is None or market_value < 0:
        row["reason"] = "期末股数、估值价或市值无效。"
        return row
    if history.empty:
        row["reason"] = "没有可核验的成交成本。"
        return row
    history = history.sort_values("date", kind="stable")
    last = history.iloc[-1]
    average_cost = _number(last.get("holding_cost_after"))
    position_after = _number(last.get("position_after"))
    cash_flows = [_number(value) for value in history["cash_flow"]]
    row["last_trade_date"] = last["date"]
    row["last_target_weight"] = _number(last.get("target_weight"))
    if average_cost is None or average_cost < 0 or position_after is None or not isclose(position_after, shares, abs_tol=1e-6):
        row["reason"] = "最后成交股数或每股成本无法与期末持仓对齐。"
        return row
    if any(value is None for value in cash_flows):
        row["reason"] = "成交现金流不完整。"
        return row
    cost_basis = average_cost * shares
    unrealized = market_value - cost_basis
    realized = fsum(cash_flows) + cost_basis
    if isclose(realized, 0.0, abs_tol=1e-9):
        realized = 0.0
    total = realized + unrealized
    row.update(
        cost_per_share=average_cost,
        cost_basis=cost_basis,
        unrealized_pnl=unrealized,
        unrealized_return=unrealized / cost_basis if cost_basis > 0 else None,
        realized_pnl=realized,
        total_pnl=total,
        zero_recovery_pnl=total - market_value if row["stranded_delisted"] else None,
        status="verified",
    )
    if row["stranded_delisted"]:
        row["research_advice"] = "退市滞留；核查估值与零回收情景，不提供正常买卖指令。"
    elif row["mark_is_stale"]:
        row["research_advice"] = "估值日期早于回测截止；先核查价格，再解释历史目标权重。"
    else:
        row["research_advice"] = "仅展示历史模拟目标权重和订单状态，不生成当前买卖建议。"
    return row


def _difference(actual: float | None, expected: float | None) -> float | None:
    if actual is None or expected is None:
        return None
    return actual - expected


def _summary(rows: list[dict], final_daily: pd.Series) -> tuple[dict, dict]:
    complete = all(row["status"] == "verified" for row in rows)
    market_complete = all(row["market_value"] is not None for row in rows)

    def total(field: str, *, verified: bool = True) -> float | None:
        if (verified and not complete) or any(row[field] is not None and not isfinite(row[field]) for row in rows):
            return None
        values = [row[field] for row in rows if row[field] is not None]
        return fsum(values) if len(values) == len(rows) else None

    market_value = total("market_value", verified=False) if market_complete else None
    listed_value = fsum(row["market_value"] for row in rows if not row["stranded_delisted"]) if market_complete else None
    stranded_value = fsum(row["market_value"] for row in rows if row["stranded_delisted"]) if market_complete else None
    differences = {
        "market_value": _difference(market_value, _number(final_daily.get("market_value"))),
        "listed_market_value": _difference(listed_value, _number(final_daily.get("listed_market_value"))),
        "stranded_delisted_value": _difference(stranded_value, _number(final_daily.get("stranded_delisted_value"))),
    }
    reconciled = all(value is not None and isclose(value, 0.0, abs_tol=0.01) for value in differences.values())
    cost_basis = total("cost_basis")
    unrealized = total("unrealized_pnl")
    summary = {
        "positions": len(rows),
        "listed_positions": sum(not row["stranded_delisted"] for row in rows),
        "stranded_delisted_positions": sum(row["stranded_delisted"] for row in rows),
        "cost_basis": cost_basis,
        "market_value": market_value,
        "unrealized_pnl": unrealized,
        "unrealized_return": unrealized / cost_basis if unrealized is not None and cost_basis else None,
        "realized_pnl": total("realized_pnl"),
        "total_pnl": total("total_pnl"),
        "zero_recovery_pnl": total("total_pnl") - stranded_value if complete and stranded_value is not None else None,
        "status": "verified" if complete and reconciled else "unavailable",
    }
    reasons = []
    if not complete:
        reasons.append("部分持仓成本或现金流未通过核验。")
    if not reconciled:
        reasons.append("持仓与日账本市值未完全对账。")
    reconciliation = {"passed": reconciled, "differences": differences, "reason": "；".join(reasons)}
    return summary, reconciliation


def _advice(summary: dict, reconciliation: dict) -> list[dict]:
    items = []
    if summary["stranded_delisted_positions"]:
        items.append({"code": "stranded_valuation", "level": "warning",
                      "text": "退市滞留估值不代表可变现资产；同时查看零回收情景，并核查退市处置和估值假设。"})
    if not summary["listed_positions"]:
        items.append({"code": "no_listed_positions", "level": "info",
                      "text": "期末无在市持仓，没有可给出的在市持仓买卖建议；旧信号和候选不能冒充当前推荐。"})
    else:
        items.append({"code": "historical_actions", "level": "info",
                      "text": "在市持仓只展示归档期末目标权重和最后订单状态，不生成当前买卖建议。"})
    if summary["status"] != "verified":
        items.append({"code": "verification_failed", "level": "warning",
                      "text": reconciliation["reason"] or "持仓收益未完全核验，请先检查账本。"})
    return items


def holdings(run_id: str, query: str = "") -> dict:
    account = _query(query)
    root, report, selected = activity._authorized(run_id, account)
    source = report["source"]
    context = {"run_id": run_id, "source_run_id": source.get("source_run_id"),
               "model_version": source.get("model_version"), "model_id": source.get("model_id"),
               "approval": source.get("approval")}
    if not selected.get("downloads"):
        return {"account": account, "as_of": None, "context": context, "summary": {"positions": 0, "status": "unavailable"},
                "reconciliation": {"passed": False, "reason": "此账户没有授权账本。"}, "holdings": [],
                "advice": [{"code": "missing_ledgers", "level": "warning", "text": "此账户没有可核验的持仓账本。"}],
                "downloads": {}}
    daily, positions, trades = _ledgers(root, report, account)
    daily, positions, trades = _visible(daily, positions, trades, report)
    daily = daily.sort_values("date")
    if daily.empty:
        return {"account": account, "as_of": None, "context": context, "summary": {"positions": 0, "status": "unavailable"},
                "reconciliation": {"passed": False, "reason": "没有共同回放日期。"}, "holdings": [],
                "advice": [{"code": "no_sessions", "level": "warning", "text": "此运行没有可核验的持仓日期。"}],
                "downloads": selected.get("downloads", {})}
    as_of = daily["date"].iloc[-1]
    final_daily = daily.iloc[-1]
    final_positions = positions.loc[positions["date"].eq(as_of) & pd.to_numeric(positions["shares"], errors="coerce").gt(0)]
    rows = [_holding(position, trades.loc[trades["symbol"].eq(position["stock"])]) for _, position in final_positions.iterrows()]
    rows.sort(key=lambda row: (-float(row["market_value"] or 0), row["stock"]))
    summary, reconciliation = _summary(rows, final_daily)
    return {"account": account, "as_of": as_of, "context": context, "summary": summary,
            "reconciliation": reconciliation, "holdings": rows, "advice": _advice(summary, reconciliation),
            "downloads": selected.get("downloads", {}),
            "research_notice": f"截至 {as_of} 的历史模拟持仓与收益，不构成当前投资建议。"}
