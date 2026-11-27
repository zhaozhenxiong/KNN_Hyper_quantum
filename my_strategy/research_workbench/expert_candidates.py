"""Project frozen replay predictions and next-open pending orders without changing them."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import expert_replay as replay, latent_api
from .runner import _write_table, write_json

SCHEMA = "khquant_expert_candidates_v1"
REPORT = "reports/expert_candidates.json"
TABLES = {
    "pending": "backtests/custom/full_pool_pending.parquet",
    "predictions": "backtests/custom/full_pool_predictions.parquet",
    "candidates": "backtests/custom/candidates.parquet",
    "candidates_csv": "backtests/custom/candidates.csv",
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("候选证据必须为 JSON 对象。")
    return value


def _scalar(value):
    if value is None or value is pd.NaT or np.isscalar(value) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{key: _scalar(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _research(root: Path, source: dict) -> tuple[Path, dict]:
    run_id = source["source_run_id"]
    if latent_api.ID.fullmatch(run_id) is None:
        raise ValueError("候选来源研究编号无效。")
    folder = latent_api._safe(root.parent, root.parent / run_id)
    if not folder.is_dir():
        raise ValueError("缺少候选来源研究目录。")
    path = replay._report_path(folder)
    if replay._sha(path) != source["source_report_sha256"]:
        raise ValueError("来源研究报告在候选生成后发生变化。")
    return path, _json(path)


def _verify_files(root: Path, payload: dict) -> None:
    for name, digest in payload["source_hashes"].items():
        if name == "research_report":
            continue
        relative = TABLES.get(name, f"reports/{name}.json")
        path = replay._file(root, relative)
        if replay._sha(path) != digest:
            raise ValueError(f"候选来源文件哈希不一致：{name}")
    research, _ = _research(root, payload["source"])
    if replay._sha(research) != payload["source_hashes"]["research_report"]:
        raise ValueError("候选绑定的来源研究报告哈希不一致。")


def write_replay(root: Path) -> Path:
    replay_path = replay._file(root, "reports/expert_stock_replay.json")
    replay_report = _json(replay_path)
    source = replay_report["source"]
    if source.get("plan_next") is not True or not source.get("candidate_of"):
        raise ValueError("只有显式全池次日预申报回放可以生成研究候选。")
    research_path, research = _research(root, source)
    pending_path = replay._file(root, TABLES["pending"])
    predictions_path = replay._file(root, TABLES["predictions"])
    pending = pd.read_parquet(pending_path)
    predictions = pd.read_parquet(predictions_path)
    required_orders = {"signal_date", "date", "stock", "filled_shares", "execution_price", "requested_shares", "status"}
    if not required_orders.issubset(pending.columns):
        raise ValueError("下一交易日预申报表缺少订单架构。")
    if predictions.empty or predictions["date"].nunique() != 1 or predictions["stock"].duplicated().any():
        raise ValueError("候选预测必须覆盖唯一信号日且每股票最多一行。")
    if pending["filled_shares"].fillna(0).ne(0).any() or pending["execution_price"].notna().any():
        raise ValueError("次日预申报不能包含成交数量或成交价格。")
    if len(pending) and (pending["date"].nunique() != 1 or pending["signal_date"].nunique() != 1):
        raise ValueError("预申报必须对应同一信号日和同一目标交易日。")
    signal_day = pd.Timestamp(predictions["date"].iloc[0])
    trade_day = pd.Timestamp(pending["date"].iloc[0]) if len(pending) else pd.Timestamp(source["next_trade_date"])
    if len(pending) and signal_day != pd.Timestamp(pending["signal_date"].iloc[0]) or trade_day <= signal_day:
        raise ValueError("候选信号日和目标交易日顺序无效。")
    quantiles = predictions[["q10", "q50", "q90", "score"]].to_numpy(dtype=float)
    if not np.isfinite(quantiles).all() or (predictions["q10"] > predictions["q50"]).any() \
            or (predictions["q50"] > predictions["q90"]).any() or not predictions["q50"].equals(predictions["score"]):
        raise ValueError("候选分位数无效或中位数与策略分数不一致。")
    ranked = predictions.sort_values(["score", "stock"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    score_columns = ["stock", "date", "q10", "q50", "q90", "score", "rank"]
    orders = pending.merge(ranked[score_columns], on="stock", how="left", suffixes=("", "_prediction"), validate="one_to_one")
    if orders["score"].isna().any():
        raise ValueError("预申报缺少对应的冻结预测。")
    selected = orders["side"].isin(["BUY", "ADD"]) & orders["status"].eq("pending") & orders["requested_shares"].gt(0)
    orders["selected"] = selected
    orders = orders.sort_values(["selected", "score", "stock"], ascending=[False, False, True])
    fields = ["stock", "board", "industry", "signal_date", "date", "side", "status", "reason", "target_weight",
              "requested_shares", "filled_shares", "execution_price", "reference_close", "reference_date", "market_status", "market_status_source",
              "limit_basis", "limit_rule_source", "risk_warning_status", "risk_warning_source", "q10", "q50", "q90",
              "score", "rank", "selected"]
    orders = orders.reindex(columns=fields)
    candidates = orders.loc[orders["selected"]].reset_index(drop=True)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    stale = now.date() > trade_day.date()
    blockers = ["研究观察名单来自滚动回放，不是最终拟合模型的正式前瞻推荐。"]
    if stale:
        blockers.append(f"目标交易日 {trade_day.date()} 已过，数据仅供历史研究。")
    payload = {
        "schema": SCHEMA, "run_id": root.name, "status": "research_only", "evidence_mode": "research_observation",
        "score_basis": "walk_forward_replay", "approval": source.get("approval", "research_only"),
        "source": source, "candidate_of": source["candidate_of"], "source_run_id": source["source_run_id"],
        "model_id": source["model_id"], "model_version": source.get("model_version"),
        "signal_date": str(signal_day.date()), "trade_date": str(trade_day.date()),
        "freshness": "stale" if stale else "current_for_trade_date", "generated_at": now.isoformat(timespec="seconds"),
        "research_decision": research.get("decision", {}), "blockers": blockers,
        "empty_reason": None if len(candidates) else "本次调仓、排名、账户和交易约束未形成有效买入预申报。",
        "candidates": _records(candidates), "orders": _records(orders),
        "source_hashes": {
            "research_report": replay._sha(research_path), "expert_stock_replay": replay._sha(replay_path),
            "pending": replay._sha(pending_path), "predictions": replay._sha(predictions_path),
        },
        "downloads": {
            "candidates": f"/api/expert/replays/{root.name}/downloads/custom--candidates.csv",
            "candidates_parquet": f"/api/expert/replays/{root.name}/downloads/custom--candidates.parquet",
            "report": f"/api/expert/replays/{root.name}/downloads/expert_candidates.json",
            "pending": f"/api/expert/replays/{root.name}/downloads/custom--full_pool_pending.parquet",
            "predictions": f"/api/expert/replays/{root.name}/downloads/custom--full_pool_predictions.parquet",
        },
    }
    table_path = latent_api._safe(root, root / TABLES["candidates"])
    _write_table(table_path, candidates)
    payload["source_hashes"]["candidates"] = replay._sha(table_path)
    csv_path = latent_api._safe(root, root / TABLES["candidates_csv"])
    candidates.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload["source_hashes"]["candidates_csv"] = replay._sha(csv_path)
    path = root / REPORT
    write_json(path, payload)
    return path


def read_replay(root: Path) -> dict:
    path = replay._file(root, REPORT)
    payload = _json(path)
    if payload.get("schema") != SCHEMA or payload.get("run_id") != root.name \
            or payload.get("status") != "research_only" or payload.get("score_basis") != "walk_forward_replay":
        raise ValueError("研究候选报告身份无效。")
    _verify_files(root, payload)
    return payload
