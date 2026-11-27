"""Read a completed portfolio replay's authorized ledgers and score evidence."""

from __future__ import annotations

from datetime import date
from math import isfinite
from pathlib import Path
import re
from urllib.parse import parse_qs

import pandas as pd

from . import expert_personal_replay as personal, expert_replay as replay, latent_api


PAGE_SIZE = 50
_ACCOUNTS = ("full_pool", "watchlist")
_STATES = ("filled", "partial", "rejected", "pretrade_rejected")
_FIELDS = {"account", "month", "symbol", "side", "status", "page", "after_last"}
_SYMBOL = re.compile(r"[0-9]{6}(?:\.(?:SH|SZ|BJ))?")
_FOLD = re.compile(r"oos_[1-3]|historical_replay")


def _query(query: str) -> dict:
    try:
        values = parse_qs(query, max_num_fields=len(_FIELDS))
        if set(values) - _FIELDS or any(len(items) != 1 for items in values.values()):
            raise ValueError("Invalid activity filters")
        options = {key: items[0].strip() for key, items in values.items()}
        account, month, symbol, side, status, page, after_last = (options.get(key, default) for key, default in (
            ("account", "full_pool"), ("month", ""), ("symbol", ""), ("side", "all"), ("status", "all"), ("page", "1"),
            ("after_last", ""),
        ))
        symbol = symbol.upper()
        if (account not in _ACCOUNTS or status not in (*_STATES, "all") or side not in ("all", "BUY", "SELL")
                or (symbol and not _SYMBOL.fullmatch(symbol))
                or (month and date.fromisoformat(f"{month}-01").isoformat()[:7] != month)
                or not re.fullmatch(r"[1-9][0-9]{0,5}", page) or after_last not in ("", "1")):
            raise ValueError("Invalid activity values")
    except ValueError as error:
        raise latent_api.RouteError(400, "invalid_activity_filter", "账户、月份、股票代码、方向、状态或页码不合法。") from error
    return {"account": account, "month": month, "symbol": symbol, "side": side, "status": status, "page": int(page),
            "after_last": after_last == "1"}


def _authorized(run_id: str, account: str) -> tuple[Path, dict, dict]:
    root = latent_api._run_folder(run_id)
    try:
        state = latent_api._read(replay._file(root, "status.json"))
        report = latent_api._read(replay._file(root, f"reports/{replay.REPORT}"))
    except (OSError, ValueError) as error:
        raise latent_api.RouteError(404, "unknown_activity_run", "没有此已完成的组合回测。") from error
    if (root.name != run_id or state.get("status") != "completed" or report.get("mode") != "expert_watchlist"
            or report.get("run", {}).get("run_id") != run_id or report.get("run", {}).get("task") != replay.TASK):
        raise latent_api.RouteError(404, "unknown_activity_run", "没有此已完成的组合回测。")
    selected = report.get("accounts", {}).get(account)
    if not isinstance(selected, dict):
        raise latent_api.RouteError(404, "unknown_activity_account", "此运行没有该账户。")
    return root, report, selected


def _ledgers(root: Path, report: dict, account: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost = report["source"]["cost"]
    tables = []
    for kind in ("daily", "orders", "trades"):
        file_id = f"custom--{account}_{cost}_{kind}.parquet"
        path = replay._file(root, personal.authorized_download(report, file_id))
        table = pd.read_parquet(path)
        table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
        tables.append(table)
    return tuple(tables)


def _score_counts(report: dict, account: str, sessions: list[str], visible: set[str]) -> tuple[dict, dict]:
    """Recover only from score/config files bound to the original replay."""
    source = report["source"]
    anchors = source.get("source_files", {})
    symbols = list(source.get("symbols", ())) if account == "watchlist" else []
    if symbols and any(not report.get("coverage", {}).get("symbols", {}).get(symbol, {}).get("in_original_pool") for symbol in symbols):
        return {}, {"basis": "unavailable", "reason": "关注股含池外股票，旧运行没有完整封存的账户级分数。"}
    if source.get("approval") not in {"candidate", "research_only", "production_fallback"} or not anchors or not visible:
        return {}, {"basis": "unavailable", "reason": "此运行缺少可校验的历史分数或评估日期。"}
    try:
        run = source["source_run_id"]
        root = latent_api._run_folder(run)
        config_file = replay._file(root, "config/study.json")
        config_sha = replay._sha(config_file)
        policy = source.get("selection_policy") or latent_api._read(config_file)["portfolio"]
        interval, threshold = policy["rebalance_days"], policy.get("min_score")
        selection_mode = policy.get("selection_mode", "absolute_q50")
        if (type(interval) is not int or interval < 1 or selection_mode not in {"absolute_q50", "cross_sectional_rank"}
                or (selection_mode == "absolute_q50" and not isfinite(float(threshold)))
                or (selection_mode == "cross_sectional_rank" and threshold is not None)):
            raise ValueError("Invalid policy")
        strategy_arm = source["arm"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", strategy_arm):
            raise ValueError("Invalid archived arm")
        dates = set()
        counts = {}
        for version in source["model_versions"]:
            fold = version["fold"]
            if version["run_id"] != run or not _FOLD.fullmatch(fold):
                raise ValueError("Unexpected archived fold")
            within = {day for day in visible if version["valid_from"] <= day <= version["valid_through"]}
            if not within:
                continue
            proof = anchors[f"{run}/{fold}"]
            path = replay._file(root, f"backtests/{fold}/{strategy_arm}_predictions.parquet")
            if proof["config_sha256"] != config_sha or proof["predictions_sha256"] != replay._sha(path) or dates.intersection(within):
                raise ValueError("Archived proof mismatch")
            dates.update(within)
            fields = ["date", "score", *(["stock"] if symbols else [])]
            frame = pd.read_parquet(path, columns=fields)
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
            present = frame["date"].isin(within)
            if symbols:
                present &= frame["stock"].isin(symbols)
            rows = frame.loc[present & frame["score"].map(isfinite), ["date", "score"]]
            qualifies = rows["score"].gt(threshold) if selection_mode == "absolute_q50" else True
            scored = rows.assign(qualifies=qualifies, negative=rows["score"].lt(0),
                                 zero=rows["score"].eq(0))
            groups = scored.groupby("date").agg(eligible=("score", "size"), qualified=("qualifies", "sum"),
                                                negative=("negative", "sum"), zero=("zero", "sum"),
                                                minimum=("score", "min"), maximum=("score", "max"))
            counts.update({day: {"eligible": int(row["eligible"]), "qualified": int(row["qualified"]),
                                 "negative": int(row["negative"]), "zero": int(row["zero"]),
                                 "min": float(row["minimum"]), "max": float(row["maximum"])}
                           for day, row in groups.iterrows()})
        if dates != visible:
            raise ValueError("Score evidence does not cover selected replay calendar")
        scheduled = set(sessions[::interval]) & visible
        market = _market_coverage(root, source, account)
        return (counts, {"basis": "pinned_archive", "rebalance_days": interval,
                         "selection_mode": selection_mode,
                         "score_threshold": float(threshold) if threshold is not None else None,
                         "scored_sessions": len(counts), "scheduled": scheduled, "market": market})
    except (KeyError, OSError, TypeError, ValueError, latent_api.RouteError):
        return {}, {"basis": "unavailable", "reason": "归档分数或原始调仓配置无法核验。"}


def _market_coverage(root: Path, source: dict, account: str) -> dict:
    """Reuse the full-pool coverage already sealed in the original report."""
    if account != "full_pool" or not source.get("source_report_sha256"):
        return {}
    try:
        path = replay._file(root, f"reports/{replay.SOURCE_REPORT}")
        if replay._sha(path) == source["source_report_sha256"]:
            return latent_api._read(path).get("input_coverage", {}).get("daily_coverage", {})
    except (OSError, ValueError, latent_api.RouteError):
        return {}
    return {}


def _research_status(report: dict) -> dict:
    """Read release evidence independently from score recovery."""
    source = report["source"]
    approval = source.get("approval")
    result = {
        "status": "research_only" if approval in {"candidate", "research_only"} else (approval or "unavailable"),
        "approval": approval,
        "source_run_id": source.get("source_run_id"),
        "release_gate": {"basis": "unavailable", "passed": None,
                         "reason": "来源研究报告或发布门状态无法核验。"},
    }
    if not source.get("source_report_sha256") or not source.get("source_run_id"):
        return result
    try:
        root = latent_api._run_folder(source["source_run_id"])
        path = replay._report_path(root)
        if replay._sha(path) != source["source_report_sha256"]:
            raise ValueError("Source report hash mismatch")
        if approval == "production_fallback":
            result["release_gate"] = {
                "basis": "pinned_archive", "passed": None, "state": "production_fallback_not_promotion",
                "selection": source.get("selection_evidence"),
                "reasons": ["按用户决定使用当前测试最优臂作为生产回退；未宣称通过独立前瞻晋升验收。"],
            }
            return result
        decision = latent_api._read(path).get("decision")
        if not isinstance(decision, dict):
            raise ValueError("Missing research decision")
        result["release_gate"] = {
            "basis": "pinned_archive",
            "passed": False if decision.get("state") == "research_only" else None,
            "state": decision.get("state"),
            "historical_numerical_gate": decision.get("historical_numerical_gate"),
            "point_in_time_verified": decision.get("point_in_time_verified"),
            "qualified_forward_results": decision.get("qualified_forward_results"),
            "reasons": [item for item in decision.get("reasons", []) if isinstance(item, str)],
        }
    except (OSError, TypeError, ValueError, latent_api.RouteError):
        pass
    return result


def _months(first: str, last: str) -> list[str]:
    if not first or not last or first > last:
        return []
    return pd.period_range(first, last, freq="M").astype(str).tolist()


def _number(value):
    return float(value) if pd.notna(value) and isfinite(float(value)) else None


def _entry(row: pd.Series) -> dict:
    shares = _number(row["filled_shares"])
    return {
        "signal_date": str(row["signal_date"]) if pd.notna(row["signal_date"]) else None,
        "date": row["date"], "symbol": row["stock"], "side": row["side"], "status": row["status"],
        "reason": str(row["reason"]) if pd.notna(row["reason"]) else None,
        "market_status": str(row["market_status"]) if pd.notna(row["market_status"]) else None,
        "requested_shares": _number(row["requested_shares"]), "filled_shares": shares,
        "price": _number(row["execution_price"]) if shares else None, "fee": _number(row["fee"]), "action": row["trade_action"],
        "source_row": int(row["source_row"]),
    }


def _inactivity(sessions: list[str], orders: pd.DataFrame, scores: dict, scheduled: set | None,
                evidence: dict, last: str | None, visible: set[str]) -> dict:
    whole_run = last is None
    after = orders if whole_run else orders.loc[orders["date"].gt(last)]
    zero = after.loc[after["filled_shares"].eq(0)]
    same_day = orders.loc[orders["date"].eq(last) & orders["filled_shares"].eq(0)] if last else orders.iloc[0:0]
    reasons = []
    if not zero.empty:
        grouped = zero.assign(reason=zero["reason"].where(zero["reason"].notna(), None)).groupby(
            ["status", "reason", "side"], dropna=False, as_index=False,
        ).agg(order_count=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
        grouped = grouped.sort_values(["order_count", "status", "side"], ascending=[False, True, True], kind="stable")
        for row in grouped.itertuples(index=False):
            reasons.append({"status": row.status, "reason": str(row.reason) if pd.notna(row.reason) else None,
                            "side": row.side, "count": int(row.order_count), "first_date": row.first_date,
                            "last_date": row.last_date})
    run_days = [day for day in sessions if day in visible]
    on = [day for day in run_days if scheduled is not None and day in scheduled]
    off = [day for day in run_days if scheduled is not None and day not in scheduled]

    def score_total(key, days):
        return sum(scores[day][key] for day in days if day in scores)

    on_scores = [scores[day] for day in on if day in scores]
    full_run = {
        "sessions": len(run_days),
        "score_days": sum(day in scores for day in run_days) if scheduled is not None else None,
        "rebalances": len(on) if scheduled is not None else None,
        "eligible_stock_days": score_total("eligible", run_days) if scheduled is not None else None,
        "eligible_on_rebalance": score_total("eligible", on) if scheduled is not None else None,
        "qualified_stock_days": score_total("qualified", run_days) if scheduled is not None else None,
        "qualified_on_rebalance": score_total("qualified", on) if scheduled is not None else None,
        "qualified_off_rebalance": score_total("qualified", off) if scheduled is not None else None,
        "negative_on_rebalance": score_total("negative", on) if scheduled is not None else None,
        "zero_on_rebalance": score_total("zero", on) if scheduled is not None else None,
        "all_negative_on_rebalance": (bool(on_scores) and _all_negative(on_scores)) if scheduled is not None else None,
        "score_min_on_rebalance": min((row["min"] for row in on_scores), default=None) if scheduled is not None else None,
        "score_max_on_rebalance": max((row["max"] for row in on_scores), default=None) if scheduled is not None else None,
        "orders": len(orders),
        "buy_orders": int(orders["side"].eq("BUY").sum()),
        "sell_orders": int(orders["side"].eq("SELL").sum()),
        "zero_fill_orders": int(orders["filled_shares"].eq(0).sum()),
        "rejected_orders": int(orders["status"].eq("rejected").sum()),
        "pretrade_rejected_orders": int(orders["status"].eq("pretrade_rejected").sum()),
        "reason_code": None,
    }
    if whole_run:
        if not orders.empty:
            full_run["reason_code"] = "UNFILLED_ORDERS"
        elif scheduled is not None:
            if not full_run["rebalances"]:
                full_run["reason_code"] = "NO_SCHEDULED_REBALANCES"
            elif not full_run["eligible_on_rebalance"]:
                full_run["reason_code"] = "NO_FINITE_SCORES"
            elif not full_run["qualified_on_rebalance"]:
                full_run["reason_code"] = "NO_QUALIFIED_CANDIDATES"
            else:
                full_run["reason_code"] = "QUALIFIED_CANDIDATES_WITHOUT_ORDERS"
    result = {
        "basis": evidence.get("basis", "unavailable"),
        "reason": evidence.get("reason"),
        "scope": "whole_run" if whole_run else "after_last_trade",
        "never_traded": whole_run,
        "selection_cause": None,
        "last_order_date": orders["date"].max() if not orders.empty else None,
        "orders_after_last": len(after),
        "buy_orders_after_last": int(after["side"].eq("BUY").sum()),
        "sell_orders_after_last": int(after["side"].eq("SELL").sum()),
        "zero_fill_after_last": len(zero),
        "zero_fill_on_last_trade_date": len(same_day) if last else None,
        "zero_fill_reasons": reasons,
        "sessions_after_last": None,
        "score_days_after_last": None,
        "rebalances_after_last": None,
        "positive_on_rebalance_after_last": None,
        "positive_off_rebalance_after_last": None,
        "full_run": full_run,
    }
    if scheduled is not None:
        days = [day for day in sessions if whole_run or day > last]
        on_after = [day for day in days if day in scheduled and not (whole_run and day == sessions[-1])]
        off_after = [day for day in days if day not in on_after]
        result.update(
            sessions_after_last=len(days), score_days_after_last=sum(day in scores for day in days),
            rebalances_after_last=len(on_after),
            positive_on_rebalance_after_last=sum(scores.get(day, {}).get("qualified", 0) for day in on_after),
            positive_off_rebalance_after_last=sum(scores.get(day, {}).get("qualified", 0) for day in off_after),
        )
    if whole_run:
        if not after.empty and after["filled_shares"].gt(0).any():
            result["selection_cause"] = "LEDGER_RECONCILIATION_REQUIRED"
        elif not after.empty:
            result["selection_cause"] = "ORDERS_NOT_FILLED"
        elif scheduled is not None:
            result["selection_cause"] = {
                "NO_QUALIFIED_CANDIDATES": "NO_QUALIFIED_CANDIDATES",
                "NO_FINITE_SCORES": "NO_FINITE_SCORES",
                "NO_SCHEDULED_REBALANCES": "NO_SCHEDULED_REBALANCES",
                "QUALIFIED_CANDIDATES_WITHOUT_ORDERS": "NO_ORDER_INTENTS_OR_UNVERIFIED_SELECTION",
            }.get(full_run["reason_code"])
    return result


def _all_negative(rows: list[dict]) -> bool:
    eligible = sum(row["eligible"] for row in rows)
    return eligible > 0 and sum(row["negative"] for row in rows) == eligible


def _result(report: dict, account: dict, options: dict, daily: pd.DataFrame, orders: pd.DataFrame, trades: pd.DataFrame) -> dict:
    coverage = report.get("coverage", {})
    sessions = daily["date"].dropna().sort_values().unique().tolist()
    begin = coverage.get("requested_start") or report["window"]["start"]
    end = coverage.get("requested_end") or report["window"]["end"]
    requested = report["source"].get("requested_model_sessions")
    visible = set(day for day in sessions if begin <= day <= end)
    if requested is not None:
        visible.intersection_update(requested)
    daily = daily.loc[daily["date"].isin(visible)].sort_values("date")
    orders = orders.assign(source_row=range(1, len(orders) + 1)).loc[orders["date"].isin(visible)].copy()
    trades = trades.loc[trades["date"].isin(visible)].copy()
    scores, proof = _score_counts(report, options["account"], sessions, visible)
    proof.update(
        approval=report["source"].get("approval"), source_arm=report["source"].get("arm"),
        source_run_id=report["source"].get("source_run_id"),
    )
    scheduled = proof.pop("scheduled", None)
    market = proof.pop("market", {})
    month_keys = _months(begin, end)
    months = []
    for month in month_keys:
        days = daily.loc[daily["date"].str.startswith(month)]
        entries = orders.loc[orders["date"].str.startswith(month)]
        executed = trades.loc[trades["date"].str.startswith(month)]
        states = entries["status"].value_counts()
        date_list = days["date"].tolist()
        month_counts = [scores[day] for day in date_list if day in scores]
        has_score_evidence = scheduled is not None and bool(date_list)
        has_market_evidence = bool(date_list) and all(
            isinstance(market.get(day, {}).get(key), int) for day in date_list for key in ("active_members", "valid_raw")
        )
        statuses = {key: int(states.get(key, 0)) if date_list else None for key in _STATES}
        months.append({"month": month, "sessions": len(days), "orders": len(entries) if date_list else None,
                       "trades": len(executed) if date_list else None, **statuses,
                       "fees": _number(executed["fee"].sum()) if date_list else None,
                       "cash_at_end": _number(days["cash"].iloc[-1]) if date_list else None,
                       "stale_positions_at_end": _number(days["stale_positions"].iloc[-1]) if date_list else None,
                       "delisted_positions_at_end": _number(days["delisted_positions"].iloc[-1])
                       if date_list and "delisted_positions" in days else None,
                       "first_date": date_list[0] if date_list else None, "last_date": date_list[-1] if date_list else None,
                       "rebalance_days": sum(day in scheduled for day in date_list) if has_score_evidence else None,
                       "score_days": len(month_counts) if has_score_evidence else None,
                       "active_stock_days": sum(market[day]["active_members"] for day in date_list) if has_market_evidence else None,
                       "valid_raw_stock_days": sum(market[day]["valid_raw"] for day in date_list) if has_market_evidence else None,
                       "eligible_stock_days": sum(row["eligible"] for row in month_counts) if has_score_evidence else None,
                       "positive_on_rebalance": sum(scores[day]["qualified"] for day in date_list if day in scores and day in scheduled) if has_score_evidence else None,
                       "positive_off_rebalance": sum(scores[day]["qualified"] for day in date_list if day in scores and day not in scheduled) if has_score_evidence else None})
    trades["stock"] = trades["symbol"].astype(object)
    trades["side"] = trades["action"].map({"BUY": "BUY", "ADD": "BUY", "SELL": "SELL", "REDUCE": "SELL"}).astype(object)
    orders["stock"] = orders["stock"].astype(object)
    orders["side"] = orders["side"].astype(object)
    trade_info = trades.groupby(["date", "stock", "side"], as_index=False).agg(fee=("fee", "sum"), trade_action=("action", "first"))
    for key in ("date", "stock", "side"):
        trade_info[key] = trade_info[key].astype(orders[key].dtype)
    rows = orders.merge(trade_info, on=["date", "stock", "side"], how="left", validate="many_to_one")
    rows["trade_action"] = rows["trade_action"].where(rows["trade_action"].notna(), None).astype(object)
    rows.loc[rows["trade_action"].isna(), "trade_action"] = None
    last = trades["date"].max() if not trades.empty else None
    diagnosis = _inactivity(sessions, orders, scores, scheduled, proof, last, visible)
    if options["after_last"]:
        rows = rows.loc[rows["date"].gt(last) & rows["filled_shares"].eq(0)] if last else rows.iloc[0:0]
    if options["month"]:
        rows = rows.loc[rows["date"].str.startswith(options["month"])]
    if options["symbol"]:
        prefix = options["symbol"] + "." if len(options["symbol"]) == 6 else options["symbol"]
        rows = rows.loc[rows["stock"].str.startswith(prefix)]
    if options["side"] != "all":
        rows = rows.loc[rows["side"].eq(options["side"])]
    if options["status"] != "all":
        rows = rows.loc[rows["status"].eq(options["status"])]
    rows = rows.sort_values(["date", "stock", "side"], ascending=[False, True, True], kind="stable")
    total = len(rows)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(options["page"], pages)
    offset = (page - 1) * PAGE_SIZE
    return {"account": options["account"], "summary": {"sessions": len(daily), "trades": len(trades), "orders": len(orders),
             "last_trade_date": last, "last_order_date": diagnosis["last_order_date"],
             "unfilled_after_last": diagnosis["zero_fill_after_last"] if last else None,
             "unfilled_on_last_trade_date": diagnosis["zero_fill_on_last_trade_date"]},
            "inactivity_diagnosis": diagnosis,
            "months": months, "entries": [_entry(row) for _, row in rows.iloc[offset : offset + PAGE_SIZE].iterrows()],
            "filtered": {"page": page, "page_size": PAGE_SIZE, "total": total, "pages": pages},
            "downloads": account.get("downloads", {}), "evidence": proof,
            "research_status": _research_status(report)}


def activity(run_id: str, query: str = "") -> dict:
    """Return bounded pages of an existing account's simulated events."""
    options = _query(query)
    root, report, account = _authorized(run_id, options["account"])
    if not account.get("downloads"):
        daily, orders, trades = (pd.DataFrame(columns=columns) for columns in (
            ["date", "cash", "stale_positions"], ["date", "stock", "side", "status", "filled_shares"], ["date", "symbol", "action", "fee"],
        ))
    else:
        daily, orders, trades = _ledgers(root, report, options["account"])
    return _result(report, account, options, daily, orders, trades)
