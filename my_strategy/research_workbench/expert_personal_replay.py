"""Run independent user portfolios against one pinned expert prediction tape."""

from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.latent_quant import forward
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, execution_evidence_identity, scaled_study_cost, simulate
from my_strategy.latent_quant.runner import _verify_ledger
from my_strategy.storage.expert_artifacts import _publish

from . import expert_replay as replay, latent_api, stock_chart_api
from .runner import _write_table, write_json


LABELS = {"full_pool": "全池组合", "watchlist": "关注股组合", "single_stock": "独立单股", "benchmark": "沪深300"}
SECTOR_LEDGERS = ("selection_candidates", "sector_candidate_daily")
ACTION_LABELS = {"buy": "买入", "add": "加仓", "hold": "持有", "reduce": "减仓", "sell": "卖出",
                 "watch": "观望", "unavailable": "暂不可评"}
STUDY_MODE = {"watchlist": "expert_watchlist", "single_stock": "expert_independent_stock"}


@dataclass
class Source:
    signature: dict
    config: dict
    strategy: Strategy
    cost: StudyCostModel
    candidate: replay.Inputs | None
    timeline: replay.Timeline | None
    approved: object | None
    roster: pd.DataFrame
    raw: tuple
    sources: dict


def _numbers(spec: dict, config: dict) -> tuple[float, StudyCostModel]:
    def number(key: str, fallback: float, *, lower: float, upper: float) -> float:
        value = spec.get(key)
        if value is None:
            return fallback
        if isinstance(value, bool):
            raise ValueError("回测资金和费用必须使用有限数值。")
        try:
            parsed = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError("回测资金和费用必须使用有限数值。") from error
        if not parsed.is_finite() or not Decimal(str(lower)) <= parsed <= Decimal(str(upper)):
            raise ValueError("回测资金或费用超出支持范围。")
        return float(parsed)

    base = StudyCostModel(**config["expense_model"])
    cash = number("initial_cash", config["initial_cash"], lower=0, upper=1_000_000_000_000)
    if cash <= 0:
        raise ValueError("初始资金需大于零。")
    commission = number("commission_bps", base.commission * 10_000, lower=0, upper=9_999.99999999) / 10_000
    slippage = number("slippage_bps", base.slippage * 10_000, lower=0, upper=9_999.99999999) / 10_000
    return cash, replace(base, commission=commission, slippage=slippage)


def _next_session(day: date, benchmark: pd.DatetimeIndex | None = None) -> str | None:
    if benchmark is not None:
        later = benchmark[benchmark > pd.Timestamp(day)]
        if len(later):
            return str(later[0].date())
    calendar = latent_api._calendar_path()
    if calendar.is_file():
        later = forward._ordered_sessions(forward._calendar(calendar), day + timedelta(days=1))
        if later:
            return str(later[0])
    return None


def _sources(spec: dict) -> Source:
    scope = spec["scope"]
    provider = spec.get("sector_provider", "capco")
    if provider not in {"capco", "ths_fuyao"}:
        raise ValueError("板块数据提供方无效。")
    if scope not in STUDY_MODE:
        raise ValueError("未知回放范围。")
    if scope == "single_stock":
        symbols = [spec["symbol"]]
    else:
        if not isinstance(spec.get("symbols"), list):
            raise ValueError("关注股必须是股票代码列表。")
        symbols = spec["symbols"]
    if any(not isinstance(symbol, str) or stock_chart_api.SYMBOL.fullmatch(symbol.upper().strip()) is None for symbol in symbols):
        raise ValueError("请输入标准股票代码。")
    symbols = sorted({symbol.upper().strip() for symbol in symbols})
    source_id = spec["source_run_id"]
    folder, _, config = replay._source(source_id)
    source_hash = replay._sha(replay._report_path(folder))
    if spec["release"].get("research_report_sha256", source_hash) != source_hash:
        raise ValueError("当前模型的登记报告与冻结版本不一致。")
    cash, rates = _numbers(spec, config)
    multiplier = config["cost_multipliers"][spec["cost"]]
    if rates.slippage * multiplier >= 1:
        raise ValueError("压力档滑点不能将卖出模拟价格降到零或负数。")
    approved = replay._is_forward_release(spec)
    candidate = timeline = verified = None
    raw = ([], [])
    archives = {}
    if approved:
        from . import expert_approved_replay, expert_forward

        verified = expert_approved_replay._archive({**spec, "symbol": symbols[0] if scope == "single_stock" else "000001.SZ"})
        model_key = verified.identity
        roster = pd.read_parquet(replay._file(folder, f"{expert_forward.ROOT}/lock/roster.parquet"))
        if "stock" in roster:
            roster = roster.set_index("stock")
        dates = verified.dates
        raw = verified.raw if scope == "single_stock" else raw
        next_day = None
        if dates:
            record = forward._json(replay._file(folder, f"{expert_forward.ROOT}/captures/{dates[-1]}/manifest.json"))
            next_day = record["next_trade_date"]
    else:
        candidate = replay._inputs({**spec, "symbol": symbols[0] if scope == "single_stock" else "000001.SZ"})
        timeline = replay._timeline(candidate)
        roster = replay._roster(candidate)
        dates = [str(day.date()) for day in timeline.requested]
        next_day = _next_session(date.fromisoformat(dates[-1]), timeline.benchmark.index) if dates else None
        model_key = dict(candidate.identity)
        if scope == "single_stock":
            raw = replay._extra_market(candidate, timeline, roster)
        for symbol in sorted(set(symbols) - set(roster.index)):
            single = replace(candidate, identity={**candidate.identity, "symbol": symbol})
            details, archived = replay._identity(single, timeline, roster, ([], []))
            archives[symbol] = (details, archived)
    requested_plan = spec.get("plan_next", False)
    candidate_of = spec.get("candidate_of")
    if type(requested_plan) is not bool:
        raise ValueError("次日预申报开关必须为布尔值。")
    if requested_plan and (scope != "watchlist" or not isinstance(candidate_of, str)
                           or latent_api.ID.fullmatch(candidate_of) is None or next_day is None):
        raise ValueError("全池次日候选需要有效的父回测和下一交易日。")
    model_key = {**model_key, "scope": scope, "symbols": symbols, "sector_provider": provider, "model_id": spec.get("model_id", spec["release_id"]),
                 "model_version": spec.get("model_version"), "initial_cash": cash,
                 "plan_next": spec.get("plan_next") is True, "candidate_of": spec.get("candidate_of"),
                 "execution_costs": asdict(scaled_study_cost(multiplier, rates)), "next_trade_date": next_day,
                 "execution_evidence": execution_evidence_identity(),
                 "personal_code_sha256": replay._sha(Path(__file__)), "source_report_sha256": source_hash,
                 "requested_model_sessions": dates, "personal_raw_sha256": stable_hash(raw),
                 "external_evidence": {symbol: details for symbol, (details, _) in archives.items()}}
    return Source(model_key, config, Strategy(**config["portfolio"]), rates, candidate, timeline, verified, roster, raw, archives)


def replay_identity(spec: dict) -> dict:
    return _sources(spec).signature


def _approved_tape(source: Source):
    from . import expert_forward

    archive = source.approved
    dates = ([day for day in archive.assessment["sessions"] if day <= archive.dates[-1]]
             if source.signature["scope"] == "single_stock" else archive.dates)
    panel_rows, status_rows, benchmarks = [], [], []
    for day in dates:
        folder = archive.source / expert_forward.ROOT / "captures" / day
        manifest = replay._file(folder, "manifest.json")
        if replay._sha(manifest) != archive.assessment["capture_manifests"][day]:
            raise ValueError("前瞻日快照在回放读取期间发生变化。")
        record = forward._json(manifest)
        for name, rows in (("panel", panel_rows), ("market_status", status_rows), ("benchmark", benchmarks)):
            evidence = record["files"][name]
            file = replay._file(folder, evidence["file"])
            if replay._sha(file) != evidence["sha256"]:
                raise ValueError("前瞻行情在回放读取期间发生变化。")
            rows.append(forward._table(file, forward.FIELDS[name]))
    frame, status, benchmark = (pd.concat(rows, ignore_index=True) for rows in (panel_rows, status_rows, benchmarks))
    for table in (frame, status, benchmark):
        table["date"] = pd.to_datetime(table["date"])
    arm = source.signature["arm"]
    relative = f"{arm}_predictions.parquet"
    filename = replay._file(archive.source, f"{expert_forward.ROOT}/acceptance/{relative}")
    if replay._sha(filename) != archive.assessment["artifacts"][relative]:
        raise ValueError("冻结前瞻预测已变化。")
    scores = pd.read_parquet(filename, columns=["date", "stock", *replay.QUANTILES])
    aligned = frame[["date", "stock"]].join(scores.set_index(["date", "stock"]), on=["date", "stock"], validate="one_to_one")
    sessions = pd.DatetimeIndex(dates)
    returns = benchmark.set_index("date")["close"].pct_change(fill_method=None).reindex(sessions)
    return frame, status, sessions, returns, aligned


def _common_tape(source: Source):
    if source.approved is not None:
        return _approved_tape(source)
    timeline = source.timeline
    if source.signature["scope"] == "watchlist":
        dates = timeline.requested
        timeline = replace(timeline, sessions=dates, assignment={day: timeline.assignment[day] for day in dates})
    frame, _, status, sessions, returns, scores = replay._tape(source.candidate, timeline)
    return frame, status, sessions, returns, scores


def _personal_tape(source: Source, stock: str, original: pd.DataFrame):
    details, archived = source.sources[stock]
    problems = [proof["unavailable"] for proof in details["native_experts"].values() if "unavailable" in proof]
    if problems:
        raise ValueError(problems[0])
    candidate = replace(source.candidate, identity={**source.candidate.identity, "symbol": stock})
    frame, roster, market, _, scores, timing, _ = replay._single_stock(candidate, source.timeline, archived, original)
    return frame, roster, market, scores, timing


def _points(daily: pd.DataFrame, column: str, initial: float, dates: list[str]) -> list[dict]:
    values = daily.set_index("date")[column].loc[dates] / initial
    drawdown = values / values.cummax().clip(lower=1.0) - 1
    return [{"date": day, "nav": float(value), "cumulative_return": float(value - 1), "drawdown": float(drawdown[day])}
            for day, value in values.items()]


def _simulate(source: Source, frame: pd.DataFrame, status: pd.DataFrame, roster: pd.DataFrame, scores: pd.Series,
              sessions: pd.DatetimeIndex, benchmark: pd.Series, run_id: str, *, plan_next: bool) -> tuple[dict | None, pd.DataFrame]:
    key = source.signature
    cash = key["initial_cash"]
    cost = source.config["cost_multipliers"][key["cost"]]
    executable = len(sessions) > 1
    next_day = pd.Timestamp(key["next_trade_date"]) if key["next_trade_date"] else None
    planned = None
    if (plan_next or not executable) and next_day is not None:
        pending_sessions = sessions.append(pd.DatetimeIndex([next_day]))
        planned = simulate(frame, scores, pending_sessions, benchmark, run_id=run_id,
                           initial_cash=cash, strategy=source.strategy, cost_multiplier=cost,
                           base_cost_model=source.cost, market_status=status, roster=roster)
        day = str(next_day.date())
        if planned["trades"]["date"].eq(day).any():
            raise ValueError("没有下一交易日行情，预申报不能生成成交。")
        pending = planned["orders"].loc[planned["orders"]["date"].eq(day)].copy() if plan_next else pd.DataFrame()
        if plan_next and len(pending):
            requested = pending["requested_shares"].gt(0)
            pending["status"] = np.where(requested, "pending", "pretrade_rejected")
            pending.loc[requested, "reason"] = None
    else:
        pending = pd.DataFrame()
    if executable:
        outcome = simulate(frame, scores, sessions, benchmark, run_id=run_id, initial_cash=cash,
                           strategy=source.strategy, cost_multiplier=cost, base_cost_model=source.cost,
                           market_status=status, roster=roster)
    elif planned is not None:
        # One observed day starts from cash. The second day only supplies the
        # calendar needed to preview a possible next-open order.
        candidates = planned["selection_candidates"].copy()
        candidates = candidates.loc[pd.to_datetime(candidates["date"]).isin(sessions)].copy()
        candidates[["order_count", "fill_count"]] = 0
        from my_strategy.latent_quant.sector_features import summarize_selection_candidates

        sector = summarize_selection_candidates(
            candidates, sessions=sessions,
            rebalance_dates=candidates.loc[candidates["planned_rebalance"], "date"].drop_duplicates().sort_values(),
        )
        outcome = {**planned, "daily": planned["daily"].iloc[:1], "orders": planned["orders"].iloc[:0],
                   "selection_candidates": candidates, "sector_candidate_daily": sector,
                   "metrics": {**planned["metrics"], "benchmark_return": 0.0, "excess_return": 0.0, "order_count": 0,
                               "rejected_order_count": 0, "pretrade_rejected_count": 0, "reject_rate": None, "reasons": {}}}
    else:
        return None, pending
    outcome["verification"] = _verify_ledger(outcome, sessions, initial_cash=cash)
    return outcome, pending


def _full_cache_root() -> Path:
    return latent_api.RUNS.parent / "expert_account_cache"


def _full_cache(source: Source, frame: pd.DataFrame, market: pd.DataFrame, pool: pd.DataFrame,
                dates: pd.DatetimeIndex, benchmark: pd.Series, *, plan_next: bool) -> tuple[dict | None, pd.DataFrame]:
    key = source.signature
    fixed = {name: key.get(name) for name in (
        "source_run_id", "source_report_sha256", "datasets", "source_files", "manifest_sha256", "fallback_manifest_sha256",
        "arm", "initial_cash",
        "execution_costs", "execution_evidence", "personal_code_sha256", "code_sha256",
    )}
    fixed.update(portfolio=asdict(source.strategy), sessions=[str(day.date()) for day in dates],
                 next_trade_date=key["next_trade_date"] if plan_next or len(dates) == 1 else None)
    account_id = f"expert-full-{stable_hash(fixed)}"
    destination = _full_cache_root() / account_id

    def read():
        record = forward._json(replay._file(destination, "manifest.json"))
        if record["identity"] != fixed or set(record["hashes"]) != {*latent_api.LEDGERS, *SECTOR_LEDGERS, "pending"}:
            raise ValueError("共用全池账户与固定模型、费用或交易日历不一致。")
        outputs = {}
        for name, digest in record["hashes"].items():
            path = replay._file(destination, f"{name}.parquet")
            if replay._sha(path) != digest:
                raise ValueError("共用全池账本工件哈希不一致。")
            outputs[name] = pd.read_parquet(path)
        pending = outputs.pop("pending")
        outputs.update(metrics=record["metrics"], verification=record["verification"], cache_folder=destination,
                       ledger_run_id=account_id, cache_hashes=record["hashes"])
        return outputs, pending

    if destination.exists():
        return read()
    outcome, pending = _simulate(source, frame, market, source.roster, pool["score"], dates, benchmark, account_id, plan_next=plan_next)
    if outcome is None:
        return outcome, pending
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        record = {"identity": fixed, "metrics": outcome["metrics"], "verification": outcome["verification"], "hashes": {}}
        for name in (*latent_api.LEDGERS, *SECTOR_LEDGERS, "pending"):
            path = stage / f"{name}.parquet"
            _write_table(path, pending if name == "pending" else outcome[name])
            record["hashes"][name] = replay._sha(path)
        write_json(stage / "manifest.json", record)
        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
    return read()


def _slice(frame: pd.DataFrame, scores: pd.DataFrame, days: pd.DatetimeIndex, symbols: set[str]) -> tuple:
    picked = frame.loc[frame["date"].isin(days) & frame["stock"].isin(symbols)].reset_index(drop=True)
    prediction_keys = scores.dropna(subset=["date", "stock"]).set_index(["date", "stock"])
    wanted = picked[["date", "stock"]].join(prediction_keys.drop(columns=["target", "label_end"], errors="ignore"),
                                             on=["date", "stock"], validate="one_to_one")
    return picked, wanted


def _position(outcome: dict | None, stock: str, day: str, initial: float) -> tuple[int, float]:
    if outcome is None:
        return 0, 0.0
    positions = outcome["positions"]
    stock_day = positions.loc[positions["stock"].eq(stock) & positions["date"].eq(day)]
    if stock_day.empty:
        return 0, 0.0
    equity = outcome["daily"].loc[outcome["daily"]["date"].eq(day), "equity"]
    return int(stock_day["shares"].item()), float(stock_day["market_value"].item() / (equity.item() if len(equity) else initial))


def _intent(orders: pd.DataFrame, symbol: str) -> dict | None:
    if orders.empty:
        return None
    rows = orders.loc[orders["stock"].eq(symbol)]
    if rows.empty:
        return None
    order = rows.iloc[-1]
    preview = {key: order[key] for key in ("signal_date", "date", "side", "reason")}
    if order["requested_shares"] > 0:
        preview["reason"] = None
    return {**preview, "requested_shares": int(order["requested_shares"]), "desired_shares": int(order["desired_shares"]),
            "target_weight": float(order["target_weight"]), "reference_close": float(order["reference_close"]),
            "status": "pending" if order["requested_shares"] > 0 else "pretrade_rejected"}


def _rank(pool: pd.DataFrame, symbol: str, when: pd.Timestamp, score: float | None) -> dict:
    today = pool.loc[pool["date"].eq(when) & pool["score"].notna(), ["stock", "score"]]
    total = len(today)
    member = today["stock"].eq(symbol).any()
    result = {"rank": None, "total": total, "member": bool(member),
              "rank_basis": "original_pool" if member else "hypothetical_insertion"}
    if score is not None and total:
        higher = today["score"].gt(score) | (today["score"].eq(score) & today["stock"].lt(symbol))
        result["rank"] = int(higher.sum()) + 1
    return result


def _latest(source: Source, case: dict | None, orders: pd.DataFrame, stock_frame: pd.DataFrame, predictions: pd.DataFrame,
            full: dict | None, full_orders: pd.DataFrame, pool: pd.DataFrame) -> dict:
    key = source.signature
    dates = key["requested_model_sessions"]
    stock = key["symbols"][0]
    when = pd.Timestamp(dates[-1]) if dates else None
    day = dates[-1] if dates else None
    cash = key["initial_cash"]
    shares, current = _position(case, stock, day, cash)
    original_shares, original_weight = _position(full, stock, day, cash)
    chosen = predictions.loc[predictions["date"].eq(when) & predictions["score"].notna()] if len(predictions) else pd.DataFrame()
    eligible = len(chosen) == 1
    q = {q: float(chosen.iloc[0][q]) if eligible else None for q in replay.QUANTILES}
    qualified = eligible and (source.strategy.selection_mode == "cross_sectional_rank"
                              or q["score"] > source.strategy.min_score)
    due = bool(dates and (len(dates) - 1) % source.strategy.rebalance_days == 0)
    preview = _intent(orders, stock)
    original = _intent(full_orders, stock)
    previous = full["orders"].loc[full["orders"]["stock"].eq(stock) & full["orders"]["date"].eq(day)] if full else pd.DataFrame()
    actual = None if previous.empty else previous.iloc[-1]["side"]
    reference = _rank(pool, stock, when, q["score"]) if day else {"rank": None, "total": 0, "member": False}
    reference.update(shares=original_shares, weight=original_weight,
                     action=(original["side"] if original else actual) or ("hold" if original_shares else "watch"),
                     pending_order=original)
    target = current
    if not eligible:
        state, reason, target = "unavailable", "最新可回放交易日缺少可评分的冻结特征或模型覆盖。", None
    elif not due:
        state = "hold" if shares else "watch"
        reason = ("当天未到独立账户调仓日。"
                  if shares or qualified else "当天未到独立账户调仓日，分数也未超过买入门槛。")
    elif not key["next_trade_date"]:
        state, reason = ("hold" if shares else "watch"), "未核验下一交易日，当前保留仓位和收盘预测。"
    elif preview and preview["requested_shares"] > 0:
        target = preview["target_weight"]
        if preview["side"] == "BUY":
            state = "add" if shares else "buy"
        else:
            state = "sell" if target == 0 else "reduce"
        reason = "按当日收盘可见价格固定申报股数，等待下一交易日模拟执行。"
    elif preview:
        target = preview["target_weight"]
        state = "hold" if shares else "watch"
        reason = "按上一交易日可见流动性或最小委托单位计算，未能形成有效申报。"
    else:
        state = "hold" if shares else "watch"
        if not qualified:
            reason = "最新分数未超过买入门槛。"
        else:
            reason = "已到调仓日，但在账户仓位及最小申报单位约束下没有新增申报。"
            if not stock_frame.empty:
                row = stock_frame.loc[stock_frame["date"].eq(when)].iloc[0]
                weight = (1 - source.strategy.cash_buffer) / source.strategy.top_n
                cap = source.strategy.max_industry_weight if pd.notna(row.get("industry")) else 1.0
                target = min(source.strategy.max_weight, weight, source.strategy.max_board_weight, cap)
    return {"state": state, "label": ACTION_LABELS[state], "signal_date": day, "next_trade_date": key["next_trade_date"],
            **q, "current_shares": shares, "current_weight": current, "target_weight": target,
            "rebalance_due": due, "eligible": eligible, "reason": reason, "original_pool": reference,
            "pending_order": preview if preview and preview["requested_shares"] > 0 else None}


def _write_sector_archive(root: Path, source: Source, full_frame: pd.DataFrame, benchmark: pd.Series, outcome: dict) -> dict:
    from .expert_sector import write_sector_artifacts

    market, close = full_frame, benchmark
    if source.candidate is not None:
        end = pd.to_datetime(full_frame["date"]).max()
        panel = pd.read_parquet(source.candidate.files["panel.parquet"])
        panel["date"] = pd.to_datetime(panel["date"], errors="raise")
        market = panel.loc[panel["date"].le(end)].copy()
        frozen_benchmark = pd.read_parquet(source.candidate.files["benchmark.parquet"])
        frozen_benchmark["date"] = pd.to_datetime(frozen_benchmark["date"], errors="raise")
        close = frozen_benchmark.set_index("date")["close"]
    return write_sector_artifacts(
        root, market, close, outcome["selection_candidates"], outcome["sector_candidate_daily"],
        provider=source.signature.get("sector_provider", "capco"),
    )


def _write_account(root: Path, account: str, cost: str, outcome: dict | None) -> dict:
    files = {}
    if outcome is not None:
        for name in (*latent_api.LEDGERS, *SECTOR_LEDGERS):
            relative = f"{account}_{cost}_{name}.parquet"
            destination = root / "backtests/custom" / relative
            if "cache_folder" in outcome:
                source = replay._file(outcome["cache_folder"], f"{name}.parquet")
                if replay._sha(source) != outcome["cache_hashes"][name]:
                    raise ValueError("导出期间共用全池账本哈希变化。")
                _publish(source, destination)
            else:
                _write_table(destination, outcome[name])
            files[name] = f"/api/expert/replays/{root.name}/downloads/custom--{relative}"
    return {"label": LABELS[account], "metrics": outcome["metrics"] if outcome else None,
            "verification": outcome["verification"] if outcome else None, "downloads": files,
            "ledger_run_id": outcome.get("ledger_run_id", root.name) if outcome else None}


def _write_scores(root: Path, account: str, values: pd.DataFrame) -> None:
    file = root / "backtests/custom" / f"{account}_predictions.parquet"
    scored = values.loc[values["score"].notna()] if "score" in values else pd.DataFrame()
    scored = scored.reindex(columns=["date", "stock", *replay.QUANTILES]).astype(
        {"stock": "string", **dict.fromkeys(replay.QUANTILES, "float64")},
    ).assign(label_end=pd.NaT, target=np.nan)
    scored["date"] = pd.to_datetime(scored["date"])
    _write_table(file, scored)


def _preview_event(root: Path, account: str, symbol: str, plan: dict | None) -> dict | None:
    if plan is None:
        return None
    path = f"backtests/custom/{account}_pending.parquet"
    _write_table(root / path, pd.DataFrame([{"stock": symbol, **plan}]))
    return {"id": f"{root.name}:{account}:pending:{symbol}:{plan['signal_date']}", "stage": "signal",
            "side": plan["side"], "action": plan["side"], "time": plan["signal_date"],
            "signal_date": plan["signal_date"], "execution_date": plan["date"], "status": "pending",
            "price": plan["reference_close"], "shares": plan["requested_shares"], "filled_shares": None,
            "requested_shares": plan["requested_shares"], "target_weight": plan["target_weight"], "fee": None,
            "reason": None, "source_file": path, "source_row": 0,
            "source_url": f"/api/expert/replays/{root.name}/downloads/custom--{account}_pending.parquet"}


def _chart(root: Path, source: Source, coverage: dict, frame: pd.DataFrame, market: pd.DataFrame, accounts: dict,
           latest: dict, warnings: list[str], has_scores: bool) -> dict:
    key = source.signature
    settings = replace(source.candidate, identity={**key, "arm": "single_stock"}) if source.candidate else replay.Inputs(
        {**key, "arm": "single_stock"}, {}, (), source.config, source.strategy, source.cost,
        source.config["cost_multipliers"][key["cost"]],
    )
    replay_info = {"mode": "expert_single_stock" if coverage["model_start"] else "expert_market_only", "run_id": root.name,
                   "approval": key["approval"], "source_run_id": key["source_run_id"], "coverage": coverage}
    result = replay._chart(root, settings, frame, market, source.raw, replay_info,
                           has_ledger=bool(accounts["single_stock"]["downloads"]), has_predictions=has_scores,
                           warnings=warnings)
    result["context"]["model_id"] = key["model_id"]
    result["latest_action"] = latest
    result["strategies"][0]["label"] = LABELS["single_stock"]
    results = result["strategies"]
    cost, symbol, start, end = (key[name] for name in ("cost", "symbol", "start", "end"))
    proof = f"/api/expert/replays/{root.name}/downloads/"
    full = bool(accounts["full_pool"]["downloads"])
    results.append({
        "id": "full_pool", "label": LABELS["full_pool"], "status": "replayed" if symbol in source.roster.index else "not_covered",
        "events": stock_chart_api._event_rows(root, "full_pool", "custom", cost, symbol, start, end, download_prefix=proof) if full else [],
        "positions": stock_chart_api._positions(root, "full_pool", "custom", cost, symbol, start, end) if full else [],
        "predictions": stock_chart_api._predictions(root, "baseline", "full_pool", "custom", symbol, start, end, replay=True) if full else [],
    })
    for strategy, order in zip(results, (latest["pending_order"], latest["original_pool"]["pending_order"]), strict=True):
        if order and order["status"] == "pending":
            strategy["events"].append(_preview_event(root, strategy["id"], symbol, order))
    return result


def _compare(source: Source, outcomes: dict, dates: list[str]) -> dict:
    initial = source.signature["initial_cash"]
    series = []
    for name, outcome in outcomes.items():
        if outcome is not None:
            series.append({"id": name, "label": LABELS[name], "points": _points(outcome["daily"], "equity", initial, dates)})
    if outcomes.get("full_pool") is not None:
        full = outcomes["full_pool"]["daily"]
        first_base = float(full.loc[full["date"].eq(dates[0]), "benchmark"].item())
        series.append({"id": "benchmark", "label": LABELS["benchmark"], "points": _points(full, "benchmark", first_base, dates)})
    return {"benchmark": "000300.SH", "series": series}


def run(*, run_id: str, spec: dict) -> Path:
    source = _sources(spec)
    key = source.signature
    if spec.get("input_identity", key) != key:
        raise ValueError("提交后模型、数据或账户配置发生变化。")
    root = latent_api._run_folder(run_id)
    if root.name != run_id or (root.exists() and any(path.name != "expert.log" for path in root.iterdir())):
        raise ValueError("回放运行目录已有工件。")
    dates = key["requested_model_sessions"]
    context = create_run_context(task=replay.TASK, run_id=run_id, as_of_date=dates[-1] if dates else key["end"], config=key,
                                 data_version=stable_hash(key["datasets"]), seed=source.config.get("seed", 42))
    context.subdir("backtests", "custom")
    write_json(root / "status.json", {"status": "running", "phase": "portfolio"})
    try:
        scope, cost = (key[name] for name in ("scope", "cost"))
        symbols = set(key["symbols"])
        days = pd.DatetimeIndex(dates)
        covered, stock_proofs = {}, {}
        warnings = []
        outcomes = {"full_pool": None}
        if symbols:
            outcomes[scope] = None
        full_frame = chosen_frame = market = chosen_market = pool = chosen_pred = pd.DataFrame()
        full_plan = chosen_plan = pd.DataFrame()
        if dates:
            full_frame, market, continuous_days, benchmark, pool = _common_tape(source)
            outcomes["full_pool"], full_plan = _full_cache(
                source, full_frame, market, pool, continuous_days, benchmark,
                plan_next=key["plan_next"] or (scope == "single_stock" and
                (len(continuous_days) - 1) % source.strategy.rebalance_days == 0),
            )
            if key["plan_next"]:
                latest_day = pool["date"].max()
                _write_scores(root, "full_pool", pool.loc[pool["date"].eq(latest_day)])
                _write_table(root / "backtests/custom/full_pool_pending.parquet", full_plan)
            in_pool = symbols.intersection(source.roster.index)
            covered = {stock: {"in_original_pool": True} for stock in sorted(in_pool)}
            chosen_frame, chosen_pred = _slice(full_frame, pool, days, in_pool)
            chosen_market = market.loc[market["stock"].isin(in_pool)]
            chosen_roster = source.roster.loc[sorted(in_pool)]
            for stock in sorted(symbols - in_pool):
                try:
                    if stock not in source.sources:
                        raise ValueError("已批准版本缺少该股同日封存输入。")
                    extra_frame, added_roster, extra_market, added_score, proof = _personal_tape(source, stock, pool)
                    extra_frame, extra_pred = _slice(extra_frame, added_score, days, {stock})
                    chosen_frame = pd.concat([chosen_frame, extra_frame], ignore_index=True)
                    chosen_pred = pd.concat([chosen_pred, extra_pred], ignore_index=True)
                    chosen_market = pd.concat([chosen_market, extra_market], ignore_index=True)
                    chosen_roster = pd.concat([chosen_roster, added_roster])
                    covered[stock] = {"in_original_pool": False}
                    stock_proofs[stock] = proof
                except (KeyError, OSError, ValueError) as error:
                    covered[stock] = {"in_original_pool": False, "reason": str(error)}
            if symbols and not chosen_pred.empty and chosen_pred["score"].notna().any():
                outcomes[scope], chosen_plan = _simulate(source, chosen_frame, chosen_market, chosen_roster, chosen_pred["score"],
                                                        days, benchmark, run_id, plan_next=scope == "single_stock"
                                                        and (len(days) - 1) % source.strategy.rebalance_days == 0)
            if symbols:
                counts = chosen_pred.loc[chosen_pred["score"].notna(), "stock"].value_counts()
                for stock in symbols:
                    covered[stock]["eligible_rows"] = int(counts.get(stock, 0))
            if dates[0] != key["start"] or dates[-1] != key["end"]:
                warnings.append("策略按冻结模型共同覆盖的交易日回放，其他日期仅显示可读行情。")
        else:
            warnings.append("当前所选日期没有可回放的冻结模型分数。")
            covered = {stock: {"in_original_pool": stock in source.roster.index, "eligible_rows": 0} for stock in sorted(symbols)}
        accounts = {name: _write_account(root, name, cost, outcome) for name, outcome in outcomes.items()}
        sector_report = None
        if outcomes["full_pool"] is not None and not full_frame.empty:
            sector_report = _write_sector_archive(root, source, full_frame, benchmark, outcomes["full_pool"])
        coverage = {"requested_start": key["start"], "requested_end": key["end"], "model_start": dates[0] if dates else None,
                    "model_end": dates[-1] if dates else None, "market_start": dates[0] if dates else None,
                    "market_end": dates[-1] if dates else None, "symbols": covered,
                    "unavailable_symbols": [stock for stock, detail in covered.items() if not detail.get("eligible_rows")],
                    "input_details": stock_proofs}
        latest = chart = None
        if scope == "single_stock":
            symbol = key["symbol"]
            latest = _latest(source, outcomes[scope], chosen_plan, chosen_frame, chosen_pred,
                             outcomes["full_pool"], full_plan, pool)
            if chosen_pred.empty or not chosen_pred["score"].notna().any():
                warnings.append(covered[symbol].get("reason", "该股票在回放窗口没有合格特征。"))
            if dates:
                _write_scores(root, "single_stock", chosen_pred)
                _write_scores(root, "full_pool", pool.loc[pool["stock"].eq(symbol)])
            chart = _chart(root, source, coverage, chosen_frame, chosen_market, accounts, latest, warnings, bool(dates))
        if execution_evidence_identity() != key["execution_evidence"]:
            raise ValueError("回放期间历史撮合规则发生变化，请重新提交。")
        write_json(root / "data_snapshots" / "source_references.json", {
            "source_run_id": key["source_run_id"], "model_id": key["model_id"], "input_hashes": key["datasets"],
            "symbols": key["symbols"], "per_stock": stock_proofs, "effective_costs": key["execution_costs"],
            "execution_evidence": key["execution_evidence"],
        })
        report = {
            "mode": STUDY_MODE[scope], "run": context.to_dict(), "source": key, "coverage": coverage, "chart": chart, "latest_action": latest,
            "window": {"start": key["start"], "end": key["end"], "sessions": len(dates), "stock_count": len(source.roster),
                       "initial_cash": key["initial_cash"], "initial_position": "cash",
                       "full_pool_anchor": (str(continuous_days[0].date()) if dates else None)},
            "accounts": accounts, "performance": _compare(source, outcomes, dates) if scope == "watchlist" else None,
            "sector_resonance": sector_report,
            "warnings": warnings,
            "qualification": "所有账户共用固定模型；盘后生成模拟信号，下一交易日开盘近似撮合。",
        }
        path = root / "reports" / replay.REPORT
        write_json(path, report)
        if key["plan_next"]:
            from .expert_candidates import write_replay

            write_replay(root)
        write_json(root / "status.json", {"status": "completed", "phase": "complete"})
        return path
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "phase": "evidence", "message": str(error)})
        raise


def authorized_download(report: dict, file_id: str) -> str:
    cost = report["source"]["cost"]
    if cost not in {"normal", "double"}:
        raise ValueError("费用档无效。")
    names = {f"{name}_{cost}_{kind}.parquet" for name, account in report["accounts"].items()
             if name in LABELS and account["downloads"] for kind in (*latent_api.LEDGERS, *SECTOR_LEDGERS)}
    if report["mode"] == "expert_independent_stock":
        names.update({f"{account}_{kind}.parquet" for account in ("single_stock", "full_pool") for kind in ("predictions", "pending")})
    if report["source"].get("plan_next"):
        names.update({f"full_pool_{kind}.parquet" for kind in ("predictions", "pending")})
        names.update({"candidates.parquet", "candidates.csv"})
    allowed = {f"custom--{name}": f"backtests/custom/{name}" for name in names}
    if report["source"].get("plan_next"):
        allowed["expert_candidates.json"] = "reports/expert_candidates.json"
    if file_id in allowed:
        return allowed[file_id]
    from .expert_sector import authorized_download as sector_download

    relative = sector_download(replay.latent_api._run_folder(report["run"]["run_id"]), report, file_id)
    if relative is None:
        raise KeyError(file_id)
    return relative
