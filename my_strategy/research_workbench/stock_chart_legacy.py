"""Read original simulated trade ledgers and replay a causal daily rule."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import csv
from datetime import date
from io import StringIO
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlencode

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.rule_strategies import MACrossStrategy
from my_strategy.storage.db_paths import processed_db_path, raw_db_path


_STOCK = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?")
_RULE = {
    "id": "rule:ma_cross", "label": "MA5/MA20 日线规则信号", "kind": "rule", "run_id": None,
    "execution": "signal_only", "price_basis": "有效原始收盘价；停牌及缺失日不产生信号",
    "method": "日线收盘后观察 MA5 是否高于 MA20，只保留连续两个有效均线状态发生切换的日期。",
    "source_file": "my_strategy/rule_strategies.py",
}


def _day(value: object) -> str | None:
    if isinstance(value, str) and _ISO_DATE.fullmatch(value.strip()):
        try:
            return date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError:
            return None
    return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _shares(value: object) -> int | None:
    count = _number(value)
    return int(count) if count is not None and count >= 0 and count.is_integer() else None


def _trade_event(row: dict, source_id: str, key: object) -> dict | None:
    when = _day(row.get("date"))
    action = str(row.get("action", "")).strip().upper()
    if when is None or action not in {"BUY", "SELL", "REDUCE"}:
        return None
    side = "BUY" if action == "BUY" else "SELL"
    shares = _shares(row.get("shares"))
    return {
        "id": f"{source_id}:{key}", "stage": "fill", "time": when, "side": side, "action": action,
        "signal_date": _day(row.get("signal_date")), "execution_date": when,
        "price": _number(row.get("price")), "requested_shares": _shares(row.get("requested_shares")),
        "filled_shares": shares, "shares": shares, "fee": _number(row.get("fee")), "status": "filled",
        "reason": row.get("reason") or None, "source_id": source_id,
        "run_id": row.get("run_id"), "source_file": row.get("source_file"), "source_row": row.get("row_number"),
    }


def _source(source_id: str, label: str, kind: str, events: list[dict], **metadata) -> dict:
    days = [event["time"] for event in events]
    return {
        "id": source_id, "label": label, "kind": kind, "event_count": len(events),
        "start": min(days, default=None), "end": max(days, default=None), **metadata,
    }


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def _workbench(symbol: str, folder: Path) -> tuple[dict | None, list[dict]]:
    run_id = folder.name
    status_path = folder / "workbench-status.json"
    report_path = folder / "reports" / "workbench.json"
    trades_path = folder / "backtests" / "trades.csv"
    files = (status_path, report_path, trades_path)
    expected = ARTIFACT_RUNS_ROOT.resolve() / run_id
    if (not _RUN.fullmatch(run_id) or folder.resolve() != expected
            or not all(path.is_file() and path.resolve() == expected / path.relative_to(folder) for path in files)):
        return None, []
    status = _json(status_path)
    if status.get("status") != "completed" or status.get("run_id") != run_id:
        return None, []
    report = _json(report_path)
    selected_model = report.get("summary", {}).get("selected_model")
    if report.get("run_id") != run_id or report.get("status") != "completed" or not isinstance(selected_model, str) or not selected_model:
        return None, []
    source_id = f"workbench:{run_id}"
    events = []
    with trades_path.open(encoding="utf-8-sig", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            if row.get("symbol", row.get("stock")) != symbol or row.get("run_id") not in (None, "", run_id):
                continue
            raw = {**row, "run_id": run_id, "source_file": str(trades_path), "row_number": row_number}
            if event := _trade_event(raw, source_id, row_number):
                events.append(event)
    mode = report.get("config", {}).get("mode")
    profile = "SeqLSTM 实验" if mode == "seqlstm" else "工作台实验"
    return _source(
        source_id, f"{profile} · {selected_model}", "workbench", events, run_id=run_id,
        price_basis="原账本模拟成交价格", selected_model=selected_model, source_file=str(trades_path),
        report_file=str(report_path), execution="simulated_fill", mode=mode,
    ), sorted(events, key=lambda event: event["time"])


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)


def _sqlite(symbol: str, run_id: str | None = None) -> dict[str, tuple[dict, list[dict]]]:
    path = processed_db_path()
    if not path.is_file():
        return {}
    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(backtest_trades)")}
        required = {"run_id", "stock", "date", "action", "price", "shares"}
        if not required.issubset(columns):
            return {}
        selected = sorted(required | {"fee", "reason", "source_file", "row_number", "signal_date", "requested_shares"} & columns)
        fields = ",".join(f"b.{column}" for column in selected)
        has_runs = {"run_id", "task"}.issubset({row["name"] for row in connection.execute("PRAGMA table_info(runs)")})
        join = "LEFT JOIN runs AS r ON r.run_id = b.run_id" if has_runs else ""
        task = "r.task" if has_runs else "NULL"
        where = "AND b.run_id IN (SELECT run_id FROM runs)" if has_runs else ""
        if run_id is not None:
            where = "AND b.run_id = ?"
        params = (symbol, run_id) if run_id is not None else (symbol,)
        rows = connection.execute(
            f"SELECT b.rowid AS ledger_rowid, {fields}, {task} AS task FROM backtest_trades AS b {join} "
            f"WHERE b.stock = ? {where} ORDER BY b.date, b.rowid", params,
        ).fetchall()
    events = defaultdict(list)
    tasks = {}
    for row in rows:
        current = str(row["run_id"])
        if not _RUN.fullmatch(current):
            continue
        if record := _trade_event(dict(row), f"sqlite:{current}", row["ledger_rowid"]):
            events[current].append(record)
            tasks[current] = row["task"]
    return {
        key: (_source(
            f"sqlite:{key}", f"SQLite 回测 · {tasks[key]} · {key}" if tasks[key] else f"SQLite 回测 · {key}", "sqlite", records,
            run_id=key, task=tasks[key], price_basis="原账本模拟成交价格", execution="simulated_fill",
            source_file=str(path), table="backtest_trades",
        ), records) for key, records in events.items()
    }


def _raw_daily(symbol: str, start: str, end: str, bars: list[dict] | None) -> pd.DataFrame:
    with closing(_connect(raw_db_path())) as connection:
        base = "FROM stock_daily_normalized WHERE stock = ? AND has_trade_price = 1 AND close > 0"
        warm = connection.execute(f"SELECT date, close {base} AND date < ? ORDER BY date DESC LIMIT 20", (symbol, start)).fetchall()
        if bars is None:
            current = connection.execute(f"SELECT date, close {base} AND date BETWEEN ? AND ?", (symbol, start, end)).fetchall()
        else:
            current = []
            for bar in bars:
                when, close = _day(bar.get("time")), _number(bar.get("close"))
                if when is not None and start <= when <= end and close is not None and close > 0 and bar.get("status") not in {"suspended", "delisted", "missing", "no_trade"}:
                    current.append((when, close))
    return pd.DataFrame([*reversed(warm), *current], columns=["date", "close"])


def _rule_signals(frame: pd.DataFrame, start: str, end: str) -> list[dict]:
    valid = frame.loc[frame["date"].le(end) & frame["close"].map(lambda value: _number(value) is not None and value > 0)]
    valid = valid.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    output = MACrossStrategy().generate(valid, {})
    switches = output["final_action"].isin({"buy", "sell"}) & output.index.to_series().ge(20)
    visible = output.loc[switches & output["date"].between(start, end)]
    return [{
        "id": f"{_RULE['id']}:{row.date}", "stage": "signal", "time": row.date, "side": row.final_action.upper(),
        "signal_date": row.date, "execution_date": None, "price": None, "signal_close": float(row.close),
        "requested_shares": None, "filled_shares": None, "shares": None, "fee": None, "status": "research_signal",
        "reason": row.reason, "source_id": _RULE["id"], "run_id": None, "source_file": _RULE["source_file"], "source_row": None,
    } for row in visible.itertuples(index=False)]


def catalog(symbol: str) -> list[dict]:
    """List B/S sources backed by the requested stock's own data or trade rows."""
    if not _STOCK.fullmatch(symbol):
        raise ValueError("股票代码须规范为 600000.SH 格式")
    with closing(_connect(raw_db_path())) as connection:
        first, last, count = connection.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM stock_daily_normalized WHERE stock = ? AND has_trade_price = 1 AND close > 0",
            (symbol,),
        ).fetchone()
    sources = []
    if count:
        sources.append({**_RULE, "start": first, "end": last, "bars": count, "event_count": None})
    for path in ARTIFACT_RUNS_ROOT.glob("*/workbench-status.json"):
        try:
            item, _ = _workbench(symbol, path.parent)
        except json.JSONDecodeError:
            continue
        if item is not None and item["event_count"]:
            sources.append(item)
    sources.extend(source for source, _ in _sqlite(symbol).values())
    return sorted(sources, key=lambda item: (item["kind"] == "rule", item["end"] or "", item["id"]), reverse=True)


def view(symbol: str, source_id: str, start: str, end: str, *, bars: list[dict] | None = None) -> dict:
    """Read original in-range trade events or replay the daily rule without storage writes."""
    if not _STOCK.fullmatch(symbol) or _day(start) != start or _day(end) != end or start > end:
        raise ValueError("股票代码或起止日期无效")
    kind, _, key = source_id.partition(":")
    if source_id == _RULE["id"]:
        signal = _rule_signals(_raw_daily(symbol, start, end, bars), start, end)
        return {"source": {**_RULE, "event_count": len(signal), "start": start, "end": end}, "events": signal}
    if not _RUN.fullmatch(key) or kind not in {"workbench", "sqlite"}:
        raise ValueError("未知 B/S 来源")
    if kind == "workbench":
        source, events = _workbench(symbol, ARTIFACT_RUNS_ROOT / key)
    else:
        source, events = _sqlite(symbol, key).get(key, (None, []))
    if source is None or not events:
        raise FileNotFoundError("该股票没有此来源的原始成交账本")
    items = []
    for event in events:
        if start <= event["time"] <= end:
            query = urlencode({"symbol": symbol, "source_id": source_id, "event_id": event["id"]})
            items.append({**event, "source_url": f"/api/stock-chart/legacy-evidence?{query}"})
    return {"source": source, "events": items}


def evidence(symbol: str, source_id: str, event_id: str) -> bytes:
    """Return one verified legacy ledger row without following imported source paths."""
    kind, marker, run_id = source_id.partition(":")
    if not _STOCK.fullmatch(symbol) or not marker or kind not in {"workbench", "sqlite"} or not _RUN.fullmatch(run_id):
        raise ValueError("无效的历史证据来源")
    if not event_id.startswith(f"{source_id}:"):
        raise FileNotFoundError("没有对应的历史账本行")
    if kind == "workbench":
        _, events = _workbench(symbol, ARTIFACT_RUNS_ROOT / run_id)
    else:
        _, events = _sqlite(symbol, run_id).get(run_id, (None, []))
    event = next((row for row in events if row["id"] == event_id), None)
    if event is None:
        raise FileNotFoundError("没有对应的历史账本行")
    exported = {"stock": symbol, **event}
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(exported))
    writer.writeheader()
    writer.writerow(exported)
    return stream.getvalue().encode("utf-8-sig")
