"""Read-only, stock-scoped evidence for the research workbench chart."""

from __future__ import annotations

from contextlib import closing
from datetime import date
from functools import lru_cache
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import parse_qs

import pandas as pd

from my_strategy.core.paths import DATA_ROOT
from my_strategy.latent_quant.data import _status_evidence
from my_strategy.storage.db_paths import raw_db_path

from . import latent_api
from .data import _read_bars

DEFAULT_SYMBOL = "002650.SZ"
DEFAULT_RUN = "latent-quant-20261122-cpu-v2"
DEFAULT_FOLD = "oos_1"
DEFAULT_COST = "normal"
UNIVERSE = DATA_ROOT / "raw/workbench_universe/20261120"
ROSTER = UNIVERSE / "roster.csv"
STATUS_FILE = UNIVERSE / "verified_nontrading_evidence.csv"
SYMBOL = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
RAW_PRICES = ("trade_open", "trade_high", "trade_low", "trade_close")
PROFILE_NAMES = {"baseline": "原始基线", "domestic": "国内调优", "overseas": "海外特征"}
ARM_NAMES = {
    "historical_quantiles": "历史经验分位数", "supertrend_atr": "SuperTrend＋ATR",
    "catboost_base": "CatBoost 基础", "catboost_supertrend_atr": "CatBoost＋SuperTrend／ATR",
    "domestic_fixed": "国内模型 · 固定组合", "domestic_tuned": "国内模型 · 调优组合",
    "domestic": "国内特征", "overseas_raw": "国内＋海外原始特征", "overseas_pca": "国内＋海外特征及 PCA",
}
COST_NAMES = {"normal": "常规成本", "double": "双倍佣金与滑点"}


def _error(status: int, code: str, message: str) -> None:
    raise latent_api.RouteError(status, code, message)


def _connection():
    return sqlite3.connect(raw_db_path().resolve(strict=True).as_uri() + "?mode=ro", uri=True, timeout=5)


@lru_cache(maxsize=4)
def _names(path: Path, revision: int) -> dict[str, str]:
    table = pd.read_csv(path, usecols=["stock", "stock_name"], dtype=str).dropna()
    return dict(zip(table["stock"], table["stock_name"], strict=True))


def _roster_names() -> dict[str, str]:
    return _names(ROSTER, ROSTER.stat().st_mtime_ns) if ROSTER.is_file() else {}


def _securities(symbol: str | None = None) -> list[dict]:
    roster = _roster_names()
    clause = "AND stock = ?" if symbol else ""
    with closing(_connection()) as connection:
        records = connection.execute(
            f"SELECT stock, name, exchange, first_date, last_date FROM securities WHERE raw_rows>0 {clause} ORDER BY stock",
            (symbol,) if symbol else (),
        ).fetchall()
    return [
        {"symbol": stock, "name": name or roster.get(stock) or stock, "exchange": exchange,
         "first_date": first, "last_date": last}
        for stock, name, exchange, first, last in records
    ]


def _symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL.fullmatch(symbol):
        _error(400, "invalid_symbol", "请输入标准股票代码，例如 002650.SZ。")
    return symbol


def _security(symbol: str) -> dict:
    matches = _securities(symbol)
    if not matches:
        _error(404, "unknown_stock", "本地日线仓没有这只股票。")
    return matches[0]


def search(q: str) -> dict:
    query = q.strip()
    if len(query) > 40:
        _error(400, "invalid_search", "搜索词过长。")
    key = query.casefold()
    securities = _securities()
    if key:
        securities = [row for row in securities if key in row["symbol"].casefold() or key in row["name"].casefold()]
        securities.sort(key=lambda row: (not row["symbol"].casefold().startswith(key), row["symbol"]))
    else:
        securities.sort(key=lambda row: (row["symbol"] != DEFAULT_SYMBOL, row["symbol"]))
    return {"q": query, "symbols": securities[:24]}


def _iso(value: Any) -> str | None:
    return pd.Timestamp(value).date().isoformat() if pd.notna(value) else None


def _number(value: Any) -> float | None:
    return float(value) if pd.notna(value) and isfinite(float(value)) else None


def _whole(value: Any) -> int | None:
    found = _number(value)
    return int(found) if found is not None else None


def _text(value: Any) -> str | None:
    return str(value) if pd.notna(value) and str(value) else None


def _date(value: str) -> str:
    try:
        normalized = date.fromisoformat(value).isoformat()
    except ValueError:
        normalized = None
    if normalized != value:
        _error(400, "invalid_date", "日期必须使用 YYYY-MM-DD。")
    return value


def _file(folder: Path, relative: str, *, required: bool = True) -> Path | None:
    expected = folder / relative
    path = latent_api._safe(folder, expected)
    if path != expected:
        _error(404, "invalid_artifact", "工件路径与冻结记录不一致。")
    if path.is_file():
        return path
    if required:
        _error(409, "missing_artifact", f"缺少冻结工件 {expected.name}。")
    return None


def _fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _read_run(run_id: str) -> tuple[Path, str, dict]:
    folder = latent_api._run_folder(run_id)
    if folder.name != run_id:
        _error(404, "unknown_run", "运行路径与登记编号不一致。")
    profile = latent_api._profile(folder)
    if profile is None:
        _error(404, "unknown_run", "没有可用的 Latent-Quant 运行。")
    worker = latent_api._worker_status(run_id)
    if worker.get("status") != "completed":
        _error(409, "unfinished_run", "此运行尚无完成的冻结结果。")
    report = latent_api._read(_file(folder, f"reports/{latent_api.REPORTS[profile][0]}"))
    if report.get("run", {}).get("run_id") != run_id:
        _error(409, "run_lineage_mismatch", "报告运行编号和工件目录不一致。")
    return folder, profile, report


def _source_folder(folder: Path, profile: str) -> tuple[Path, dict[str, str]]:
    if profile != "domestic":
        return folder, {}
    config = latent_api._read(_file(folder, "config/study.json"))
    coverage = latent_api._read(_file(folder, "data_snapshots/coverage.json"))
    source_id = config.get("source_run_id")
    if not source_id or source_id != coverage.get("source_run_id") or source_id == folder.name:
        _error(409, "source_lineage_mismatch", "国内调优来源未通过运行编号核对。")
    source, source_profile, _ = _read_run(source_id)
    hashes = coverage.get("frozen_input_sha256", {})
    if source_profile != "baseline" or not all(name in hashes for name in ("panel.parquet", "roster.parquet", "market_status.parquet")):
        _error(409, "source_lineage_mismatch", "国内调优缺少基线来源哈希。")
    return source, hashes


def _snapshot(folder: Path, hashes: dict[str, str], name: str) -> Path:
    path = _file(folder, f"data_snapshots/{name}")
    if name in hashes and _fingerprint(path) != hashes[name]:
        _error(409, "source_hash_mismatch", f"冻结来源文件 {name} 的哈希不一致。")
    return path


def _stock_rows(path: Path, symbol: str, key: str = "stock") -> pd.DataFrame:
    return pd.read_parquet(path, filters=[(key, "=", symbol)])


def _known_folds(report: dict) -> dict[str, dict]:
    return {item["period"]: item for item in report["folds"] if item["period"] in latent_api.FOLDS}


def _prediction_file(folder: Path, profile: str, fold: str, arm: str) -> Path | None:
    filename = "domestic_predictions.parquet" if profile == "domestic" else f"{arm}_predictions.parquet"
    return _file(folder, f"backtests/{fold}/{filename}", required=False)


def _fold_description(folder: Path, profile: str, fold: str, evidence: dict) -> dict:
    allowed_arms = latent_api.REPORTS[profile][1]
    arms = [arm for arm in allowed_arms if arm in evidence["arms"]]
    return {
        "id": fold, "start": _iso(evidence["evaluation_start"]), "end": _iso(evidence["evaluation_end"]),
        "arms": [
            {"id": arm, "label": ARM_NAMES[arm], "has_predictions": _prediction_file(folder, profile, fold, arm) is not None}
            for arm in arms
        ],
        "costs": [{"id": cost, "label": label} for cost, label in COST_NAMES.items()
                  if any(cost in evidence["arms"][arm]["execution"] for arm in arms)],
    }


def catalog(symbol: str) -> dict:
    identity = _security(symbol)
    runs = []
    for folder in sorted(latent_api.RUNS.glob("*"), reverse=True):
        if not folder.is_dir() or not latent_api.ID.fullmatch(folder.name):
            continue
        try:
            if latent_api._profile(latent_api._run_folder(folder.name)) is None:
                continue
            folder, profile, report = _read_run(folder.name)
            source, hashes = _source_folder(folder, profile)
            roster_path = _snapshot(source, hashes, "roster.parquet")
            members = pd.read_parquet(roster_path)
            if "stock" not in members:
                members = members.reset_index()
            covered = members["stock"].eq(symbol).any()
            folds = [_fold_description(folder, profile, key, value) for key, value in _known_folds(report).items()]
        except latent_api.RouteError:
            continue
        runs.append({"run_id": folder.name, "profile": profile, "label": PROFILE_NAMES[profile],
                     "status": "completed", "coverage": "covered" if covered else "not_covered", "folds": folds})
    runs.sort(key=lambda item: item["run_id"] != DEFAULT_RUN)
    eligible = next((run for run in runs if run["coverage"] == "covered" and run["folds"]), None)
    fold = next((item for item in eligible["folds"] if item["id"] == DEFAULT_FOLD), eligible["folds"][0]) if eligible else None
    start, end = _raw_dates(identity, None, None)
    from . import stock_chart_legacy
    from .expert_api import chart_release

    release = chart_release()
    return {**identity, "runs": runs, "legacy_sources": stock_chart_legacy.catalog(symbol),
            "expert_release": release, "raw_defaults": {"start": start, "end": end},
            "defaults": {"run_id": eligible["run_id"] if eligible else None, "fold": fold["id"] if fold else None,
                         "cost": DEFAULT_COST if fold else None, "start": fold["start"] if fold else start,
                         "end": fold["end"] if fold else end}}


def _raw_dates(identity: dict, start: str | None, end: str | None) -> tuple[str, str]:
    finish = min(end or identity["last_date"], identity["last_date"])
    default = identity["first_date"]
    if finish >= identity["first_date"]:
        default = (date.fromisoformat(finish) - pd.DateOffset(months=3)).date().isoformat()
    first = max(start or default, identity["first_date"])
    return first, finish


def _range(start: str, end: str) -> tuple[str, str]:
    if start > end:
        _error(400, "invalid_range", "起始日期必须早于或等于结束日期。")
    return start, end


def _warehouse_series(identity: dict, start: str | None, end: str | None, *, fallback: bool = False) -> tuple:
    symbol = identity["symbol"]
    with closing(_connection()) as connection:
        market_start, market_end = connection.execute(
            "SELECT MIN(first_date), MAX(last_date) FROM securities WHERE raw_rows > 0",
        ).fetchone()
        files = [STATUS_FILE] if STATUS_FILE.is_file() else []
        market = _status_evidence(files, [symbol], pd.date_range(market_start, market_end))
        scope = dict(identity)
        if not market.empty:
            scope["first_date"] = min(scope["first_date"], _iso(market["date"].min()))
            scope["last_date"] = max(scope["last_date"], _iso(market["date"].max()))
        first, finish = _raw_dates(scope, start, end)
        if not fallback:
            _range(first, finish)
        frame = _read_bars(connection, [symbol], first, finish)
    return frame, market, first, finish, scope


def _bars(frame: pd.DataFrame, market: pd.DataFrame, start: str, end: str) -> tuple[list[dict], list[dict]]:
    states = {
        _iso(row.date): (row.market_status, _text(row.source))
        for row in market.itertuples(index=False)
        if start <= _iso(row.date) <= end
    }
    items = {}
    nontrading = {}
    for row in frame.itertuples(index=False):
        day = _iso(row.date)
        if not start <= day <= end:
            continue
        values = [_number(getattr(row, field)) for field in RAW_PRICES]
        good = all(value is not None and value > 0 for value in values)
        flag = getattr(row, "has_trade_price", True)
        status, source = states.get(day, (_text(getattr(row, "market_status", None)), _text(row.source)))
        if status in {"suspended", "delisted"} or not good or pd.isna(flag) or not flag:
            nontrading[day] = {"time": day, "status": status or "unavailable", "source": source}
        else:
            items[day] = {"time": day, "open": values[0], "high": values[1], "low": values[2],
                          "close": values[3], "volume": _number(row.volume), "status": "tradable"}
    for day, (status, source) in states.items():
        if day not in items:
            nontrading.setdefault(day, {"time": day, "status": status, "source": source})
    for day, evidence in nontrading.items():
        items[day] = {"time": day, "open": None, "high": None, "low": None, "close": None,
                      "volume": None, "status": evidence["status"]}
    return [items[day] for day in sorted(items)], [nontrading[day] for day in sorted(nontrading)]


def _event_id(run_id: str, arm: str, fold: str, cost: str, stock: str, field: str, row: int) -> str:
    return f"{run_id}:{fold}:{arm}:{cost}:{stock}:{field}:{row}"


def _event_rows(
    folder: Path, arm: str, fold: str, cost: str, stock: str, start: str, end: str,
    *, download_prefix: str | None = None,
) -> list[dict]:
    prefix = f"backtests/{fold}/{arm}_{cost}"
    order_path = _file(folder, f"{prefix}_orders.parquet")
    trade_path = _file(folder, f"{prefix}_trades.parquet")
    orders = pd.read_parquet(order_path).reset_index(drop=True)
    trades = pd.read_parquet(trade_path).reset_index(drop=True)
    orders = orders.loc[orders["stock"].eq(stock)]
    trades = trades.loc[trades["symbol"].eq(stock)]
    download_prefix = download_prefix or f"/api/latent/runs/{folder.name}/downloads/"
    by_order = {}
    events = []
    for row in orders.itertuples():
        index = int(row.Index)
        signal, execution = _iso(row.signal_date), _iso(row.date)
        requested = _whole(row.requested_shares)
        common = {"side": row.side, "signal_date": signal, "execution_date": execution,
                  "requested_shares": requested, "target_weight": _number(row.target_weight),
                  "source_file": order_path.relative_to(folder).as_posix(), "source_row": index,
                  "source_url": f"{download_prefix}{fold}--{order_path.name}",
                  "action": row.side}
        by_order[(signal, execution, row.side)] = row
        if requested and signal and start <= signal <= end:
            events.append({
                **common, "id": _event_id(folder.name, arm, fold, cost, stock, "signal", index),
                "stage": "signal", "time": signal, "price": _number(row.reference_close), "shares": requested,
                "reference_date": _iso(row.reference_date), "filled_shares": None, "fee": None,
                "transfer_fee": None, "status": "requested", "reason": None,
            })
        if row.status in {"rejected", "pretrade_rejected"}:
            pretrade = row.status == "pretrade_rejected"
            when = signal if pretrade else execution
            if when and start <= when <= end:
                events.append({
                    **common, "id": _event_id(folder.name, arm, fold, cost, stock, "rejected", index),
                    "stage": "rejected", "time": when,
                    "price": _number(row.reference_close if pretrade else row.open), "shares": None,
                    "filled_shares": 0, "fee": None, "transfer_fee": None, "status": row.status,
                    "reason": _text(row.reason),
                })
    for row in trades.itertuples():
        index = int(row.Index)
        signal, execution = _iso(row.signal_date), _iso(row.date)
        if not execution or not start <= execution <= end:
            continue
        side = "BUY" if row.action == "BUY" else "SELL"
        order = by_order.get((signal, execution, side))
        events.append({
            "id": _event_id(folder.name, arm, fold, cost, stock, "fill", index), "stage": "fill", "time": execution,
            "signal_date": signal, "execution_date": execution, "side": side, "action": row.action,
            "price": _number(row.price), "shares": _whole(row.shares), "filled_shares": _whole(row.shares),
            "requested_shares": _whole(order.requested_shares) if order is not None else None,
            "fee": _number(row.fee), "transfer_fee": _number(getattr(row, "transfer_fee_paid", None)),
            "target_weight": _number(row.target_weight), "status": order.status if order is not None else "filled",
            "reason": _text(order.reason if order is not None else row.reason),
            "source_file": trade_path.relative_to(folder).as_posix(), "source_row": index,
            "source_url": f"{download_prefix}{fold}--{trade_path.name}",
        })
    return sorted(events, key=lambda event: (event["time"], event["stage"], event["id"]))


def _predictions(
    folder: Path, profile: str, arm: str, fold: str, stock: str, start: str, end: str, *, replay: bool = False,
) -> list[dict]:
    path = (_file(folder, f"backtests/{fold}/{arm}_predictions.parquet") if replay
            else _prediction_file(folder, profile, fold, arm))
    if path is None:
        return []
    records = []
    for row in _stock_rows(path, stock).itertuples(index=False):
        when = _iso(row.date)
        if start <= when <= end:
            maturity = _iso(row.label_end)
            record = {
                "time": when, "q10": _number(getattr(row, "q10", None)),
                "q50": _number(getattr(row, "q50", None)), "q90": _number(getattr(row, "q90", None)),
                "score": _number(row.score), "label_end": maturity,
                "target": _number(row.target) if maturity and maturity <= end else None,
            }
            if arm == "knn60_multimodal40":
                record.update({
                    "score_semantics": "knn60_multimodal40_decision_rank",
                    "knn_q50": _number(getattr(row, "knn_q50", None)),
                    "multimodal_q50": _number(getattr(row, "multimodal_q50", None)),
                    "effective_knn_weight": _number(getattr(row, "effective_knn_weight", None)),
                    "effective_multimodal_weight": _number(getattr(row, "effective_multimodal_weight", None)),
                    "fallback_reason": _text(getattr(row, "fallback_reason", None)),
                })
            records.append(record)
    return records


def _positions(folder: Path, arm: str, fold: str, cost: str, stock: str, start: str, end: str) -> list[dict]:
    path = _file(folder, f"backtests/{fold}/{arm}_{cost}_positions.parquet")
    return [
        {"time": _iso(row.date), "shares": _whole(row.shares), "mark_price": _number(row.mark_price),
         "mark_date": _iso(row.mark_date), "market_value": _number(row.market_value)}
        for row in _stock_rows(path, stock).itertuples(index=False)
        if start <= _iso(row.date) <= end
    ]


def _context(run_id: str, fold: str, cost: str) -> tuple[Path, str, dict, dict]:
    folder, profile, report = _read_run(run_id)
    known = _known_folds(report)
    if fold not in known or cost not in COST_NAMES:
        _error(400, "invalid_context", "运行没有对应的窗口或费用档。")
    details = _fold_description(folder, profile, fold, known[fold])
    if not any(item["id"] == cost for item in details["costs"]):
        _error(400, "invalid_context", "窗口没有对应费用档。")
    return folder, profile, report, details


def view(
    symbol: str, *, run_id: str | None = None, fold: str | None = None, cost: str | None = None,
    start: str | None = None, end: str | None = None, arms: tuple[str, ...] = (), legacy: tuple[str, ...] = (),
) -> dict:
    identity = _security(symbol)
    warnings = []
    strategies = []
    sources = []
    profile = None
    covered_from, covered_through = identity["first_date"], identity["last_date"]
    price_source = "raw_warehouse"
    replay_context = None
    if run_id:
        fold = fold or DEFAULT_FOLD
        cost = cost or DEFAULT_COST
        folder, profile, _, details = _context(run_id, fold, cost)
        known_arms = {item["id"] for item in details["arms"]}
        if set(arms) - known_arms:
            _error(400, "invalid_context", "窗口没有对应策略。")
        selected = arms or tuple(item["id"] for item in details["arms"])
        covered_from, covered_through = details["start"], details["end"]
        start, end = _range(max(start or covered_from, covered_from), min(end or covered_through, covered_through))
        source, hashes = _source_folder(folder, profile)
        panel = _snapshot(source, hashes, "panel.parquet")
        status_path = _snapshot(source, hashes, "market_status.parquet")
        market = _stock_rows(status_path, symbol)
        frame = _stock_rows(panel, symbol)
        covered = not frame.empty
        cached = None
        if not covered:
            from . import stock_chart_replay

            cached = stock_chart_replay.lookup(folder, profile, symbol, fold, cost)
        if cached:
            archive, replay_report = cached
            frame = _stock_rows(_file(archive, "data_snapshots/panel.parquet"), symbol)
            market = _stock_rows(_file(archive, "data_snapshots/market_status.parquet"), symbol)
            replay_context = {
                "mode": "single_stock", "run_id": archive.name, "source_run_id": folder.name,
                "cached": True, "coverage": replay_report["coverage"], "devices": replay_report["devices"],
                "qualification": replay_report["qualification"],
            }
            price_source = "single_stock_replay"
            sources.append({
                "kind": "single_stock_replay", "label": "单股模拟回放", "run_id": archive.name,
                "report_url": stock_chart_replay.download_url(archive.name, "report.json"),
                "summary_url": stock_chart_replay.download_url(archive.name, "summary.csv"),
            })
            warnings.append("这只股票不在原组合中。图中策略点来自独立的单股模拟回放。")
        elif not covered:
            warnings.append("该股票不在此运行的冻结样本内，价格图使用本地未复权日线仓。")
            frame, market, _, _, _ = _warehouse_series(identity, start, end, fallback=True)
        series_root = archive if cached else folder
        available = covered or cached is not None
        for arm in selected:
            strategies.append({
                "id": arm, "label": ARM_NAMES[arm], "status": "replayed" if cached else "covered" if covered else "not_covered",
                "events": _event_rows(
                    series_root, arm, fold, cost, symbol, start, end,
                    download_prefix=stock_chart_replay.download_url(archive.name) if cached else None,
                ) if available else [],
                "positions": _positions(series_root, arm, fold, cost, symbol, start, end) if available else [],
                "predictions": _predictions(series_root, profile, arm, fold, symbol, start, end, replay=bool(cached))
                if available else [],
            })
        if covered:
            price_source = "snapshot"
            sources.append({"kind": "snapshot", "label": "冻结不复权日线", "run_id": source.name,
                            "path": "data_snapshots/panel.parquet"})
        elif not cached:
            price_source = "raw_warehouse_fallback"
            sources.append({"kind": "raw_warehouse", "label": "本地未复权日线（冻结样本未覆盖）", "run_id": None})
    else:
        if arms or fold or cost:
            _error(400, "invalid_context", "策略、窗口与费用档需要先选择研究运行。")
        frame, market, start, end, scope = _warehouse_series(identity, start, end)
        covered_from, covered_through = scope["first_date"], scope["last_date"]
        sources.append({"kind": "raw_warehouse", "label": "本地不复权日线仓", "run_id": None})
    bars, nontrading = _bars(frame, market, start, end)
    if legacy:
        from . import stock_chart_legacy

        for source_id in legacy:
            result = stock_chart_legacy.view(symbol, source_id, start, end, bars=bars)
            item = result["source"]
            strategies.append({"id": source_id, "label": item["label"], "status": "covered",
                               "events": result["events"], "positions": [], "predictions": []})
            sources.append(item)
            warnings.extend(result.get("warnings", []))
    from . import stock_chart_research

    return stock_chart_research.attach({
        **identity, "context": {"run_id": run_id, "profile": profile, "fold": fold, "cost": cost,
                                "start": start, "end": end, "price_basis": "raw", "price_source": price_source,
                                "replay": replay_context,
                                "coverage_start": covered_from, "coverage_end": covered_through},
        "bars": bars, "nontrading": nontrading, "strategies": strategies, "warnings": warnings, "sources": sources,
    })


def _params(query: str | dict[str, list[str]], allowed: set[str]) -> dict[str, list[str]]:
    if isinstance(query, str):
        try:
            query = parse_qs(query, keep_blank_values=True, max_num_fields=28)
        except ValueError:
            _error(400, "invalid_query", "查询参数超出接口范围。")
    if set(query) - allowed:
        _error(400, "invalid_query", "请求包含未识别的查询参数。")
    for key, values in query.items():
        if key not in {"arm", "source_id"} and len(values) != 1:
            _error(400, "invalid_query", "同一查询参数只能出现一次。")
    return query


def handle_get(handler: Any, path: str, query: str | dict[str, list[str]]) -> bool:
    if not path.startswith("/api/stock-chart/"):
        return False
    route = path.removeprefix("/api/stock-chart/")
    try:
        if route == "search":
            args = _params(query, {"q"})
            payload = search(args.get("q", [""])[0])
        elif route == "catalog":
            args = _params(query, {"symbol"})
            payload = catalog(_symbol(args.get("symbol", [DEFAULT_SYMBOL])[0]))
        elif route == "legacy-evidence":
            from . import stock_chart_legacy

            args = _params(query, {"symbol", "source_id", "event_id"})
            if set(args) != {"symbol", "source_id", "event_id"} or any(len(value) != 1 for value in args.values()):
                _error(400, "invalid_query", "下载历史成交需要股票、来源与账本行标识。")
            try:
                document = stock_chart_legacy.evidence(_symbol(args["symbol"][0]), args["source_id"][0], args["event_id"][0])
            except FileNotFoundError as error:
                raise latent_api.RouteError(404, "unknown_legacy_evidence", str(error)) from error
            except ValueError as error:
                raise latent_api.RouteError(400, "invalid_query", str(error)) from error
            handler._send(200, document, "text/csv; charset=utf-8")
            return True
        elif route == "view":
            args = _params(query, {"symbol", "run_id", "fold", "cost", "start", "end", "arm", "source_id"})
            values = {key: item[0] or None for key, item in args.items() if key not in {"arm", "source_id"}}
            symbol = _symbol(values.pop("symbol", None) or DEFAULT_SYMBOL)
            if not args:
                values["run_id"] = DEFAULT_RUN
            for key in ("start", "end"):
                if values.get(key):
                    values[key] = _date(values[key])
            payload = view(symbol, **values, arms=tuple(dict.fromkeys(args.get("arm", ()))),
                           legacy=tuple(dict.fromkeys(args.get("source_id", ()))))
        elif (parts := route.split("/")) and len(parts) == 4 and parts[0] == "replays" and parts[2] == "downloads":
            from . import stock_chart_replay

            _params(query, set())
            stock_chart_replay.download(handler, parts[1], parts[3])
            return True
        else:
            _error(404, "unknown_route", "没有此项股票图接口。")
        handler._json(200, payload)
    except latent_api.RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        handler._json(500, {"message": f"股票图证据读取失败：{error}", "code": "chart_read_failed"})
    return True


def _replay_request(handler: Any) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        if not (0 < length <= 2048) or handler.headers.get_content_type() != "application/json":
            raise ValueError
        request = json.loads(handler.rfile.read(length))
        if (not isinstance(request, dict) or set(request) - {"symbol", "run_id", "fold", "cost", "start", "end"}
                or not {"symbol", "run_id"}.issubset(request) or not all(isinstance(value, str) for value in request.values())):
            raise ValueError
        return request
    except (ValueError, TypeError) as error:
        raise latent_api.RouteError(400, "invalid_request", "请输入股票与原运行，其他字段可选。") from error


def handle_post(handler: Any, path: str) -> bool:
    if path != "/api/stock-chart/replay":
        return False
    try:
        from . import stock_chart_replay

        request = _replay_request(handler)
        symbol = _symbol(request["symbol"])
        _security(symbol)
        run_id = request["run_id"]
        fold, cost = request.get("fold", DEFAULT_FOLD), request.get("cost", DEFAULT_COST)
        folder, profile, report, details = _context(run_id, fold, cost)
        start = _date(request["start"]) if request.get("start") else details["start"]
        end = _date(request["end"]) if request.get("end") else details["end"]
        _range(max(start, details["start"]), min(end, details["end"]))
        source, hashes = _source_folder(folder, profile)
        roster = pd.read_parquet(_snapshot(source, hashes, "roster.parquet"))
        if "stock" not in roster:
            roster = roster.reset_index()
        if roster["stock"].eq(symbol).any():
            _error(409, "already_in_sample", "该股票已有原组合的冻结证据。")
        try:
            _, _, cached = stock_chart_replay.generate(
                folder, profile, report, symbol=symbol, fold=fold, cost=cost,
                start=details["start"], end=details["end"],
            )
        except (FileNotFoundError, ValueError) as error:
            raise latent_api.RouteError(422, "replay_unavailable", f"单股证据不足：{error}") from error
        result = view(symbol, run_id=run_id, fold=fold, cost=cost, start=start, end=end)
        if not result["context"]["replay"]:
            _error(409, "replay_evidence_changed", "回放来源发生变更，请重试。")
        result["context"]["replay"]["cached"] = cached
        handler._json(200, result)
    except latent_api.RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, KeyError, TypeError, sqlite3.Error) as error:
        handler._json(500, {"message": f"单股证据回放失败：{error}", "code": "stock_replay_failed"})
    return True
