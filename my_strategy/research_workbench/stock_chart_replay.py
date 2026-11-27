"""Archive reproducible single-stock inference from a completed research fold."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path
import os
from threading import Lock
from typing import Any

import pandas as pd

from my_strategy.core.run_context import create_run_context, make_run_id, stable_hash
from my_strategy.latent_quant.execution import simulate
from my_strategy.latent_quant.runner import _verify_ledger

from . import latent_api
from .runner import _write_table, write_json

TASK = "stock-chart-single-stock-replay"
REPORT = "stock_replay.json"
VERSION = 1
_SUBMISSION = Lock()
_MODELS = {
    "baseline": ("catboost_base.cbm", "catboost_base.json", "catboost_supertrend_atr.cbm", "catboost_supertrend_atr.json"),
    "domestic": ("domestic.cbm", "domestic.json"),
    "overseas": ("domestic.cbm", "domestic.json", "overseas_raw.cbm", "overseas_raw.json",
                 "overseas_pca.cbm", "overseas_pca.json", "pca.json"),
}
_PREDICTIONS = ("q10", "q50", "q90", "score")


def _index_root() -> Path:
    return latent_api.RUNS.parent / "stock_chart_replays"


def _exact(folder: Path, relative: str) -> Path:
    expected = folder / relative
    actual = latent_api._safe(folder, expected)
    if actual != expected or not actual.is_file():
        raise FileNotFoundError(f"缺少确切冻结文件 {relative}")
    return actual


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _key(source: Path, profile: str, symbol: str, fold: str, cost: str) -> tuple[str, dict]:
    files = [f"reports/{latent_api.REPORTS[profile][0]}", "config/study.json"]
    files.extend(f"models/{fold}/{name}" for name in _MODELS[profile])
    if profile == "baseline":
        files.append("data_snapshots/panel.parquet")
    elif profile == "overseas":
        files.append("data_snapshots/overseas.parquet")
    evidence = {path: _hash(_exact(source, path)) for path in files}
    identity = {"version": VERSION, "source_run_id": source.name, "source_profile": profile,
                "symbol": symbol, "source_fold": fold, "source_cost": cost, "source_files": evidence}
    return stable_hash(identity), identity


@contextmanager
def _locked(key: str):
    root = _index_root()
    root.mkdir(parents=True, exist_ok=True)
    with _SUBMISSION, (root / f"{key}.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _report(run_id: str) -> tuple[Path, dict]:
    folder = latent_api._run_folder(run_id)
    if folder.name != run_id:
        raise FileNotFoundError("回放编号和目录不一致")
    status = latent_api._read(_exact(folder, "status.json"))
    report = latent_api._read(_exact(folder, f"reports/{REPORT}"))
    if (status.get("status") != "completed" or report.get("mode") != "single_stock"
            or report.get("run", {}).get("run_id") != run_id or report["run"].get("task") != TASK):
        raise FileNotFoundError("回放尚未完成")
    return folder, report


def _cached(key: str) -> tuple[Path, dict] | None:
    record = latent_api._read(_index_root() / f"{key}.json")
    if record.get("key") != key or not record.get("run_id"):
        return None
    try:
        folder, report = _report(record["run_id"])
    except (OSError, ValueError, KeyError, latent_api.RouteError):
        return None
    return (folder, report) if report.get("cache_key") == key else None


def lookup(source: Path, profile: str, symbol: str, fold: str, cost: str) -> tuple[Path, dict] | None:
    """Find an already completed replay without touching the original run."""
    try:
        key, _ = _key(source, profile, symbol, fold, cost)
        return _cached(key)
    except (OSError, ValueError, KeyError, latent_api.RouteError):
        return None


def _store_run(
    source: Path, profile: str, report: dict, identity: dict, key: str, start: str, end: str,
) -> tuple[Path, dict]:
    from .stock_replay_data import load_stock_replay_data
    from .stock_replay_models import replay

    symbol, fold, cost = (identity[name] for name in ("symbol", "source_fold", "source_cost"))
    config = latent_api._read(_exact(source, "config/study.json"))
    data = load_stock_replay_data(symbol, start, end)
    if data.frame.empty or not data.frame["signal_eligible"].any():
        raise ValueError("整折内没有符合冻结特征规则的个股日期")
    if len(data.sessions) < 2 or data.sessions[0].date().isoformat() != start or data.sessions[-1].date().isoformat() != end:
        raise ValueError("单股资料没有覆盖来源研究的完整交易日窗口")
    forecasts, strategies, devices = replay(source, profile, report, fold, data.frame, data.sessions)
    expected_arms = next(item["arms"] for item in report["folds"] if item["period"] == fold)
    if set(forecasts) != set(expected_arms) or set(strategies) != set(expected_arms):
        raise ValueError("冻结模型的对照组未完整重放")
    if not any(not prediction.empty for prediction in forecasts.values()):
        raise ValueError("冻结模型未产生符合证据要求的个股预测")
    if _key(source, profile, symbol, fold, cost)[0] != key:
        raise ValueError("来源模型或配置在回放时发生变更")
    settings = {
        **identity, "start": start, "end": end, "initial_cash": config["initial_cash"],
        "cost_multiplier": config["cost_multipliers"][cost],
        "portfolios": {arm: asdict(strategy) for arm, strategy in strategies.items()},
    }
    run_id = make_run_id("stock-chart-replay", end)
    context = create_run_context(
        task=TASK, as_of_date=end, config=settings, data_version=stable_hash(data.coverage),
        seed=config.get("seed", 42), run_id=run_id,
    )
    root = context.run_dir()
    reports = {}
    try:
        write_json(root / "status.json", {"status": "running", "phase": "simulation"})
        write_json(context.subdir("config") / "study.json", settings)
        write_json(context.subdir("data_snapshots") / "coverage.json", data.coverage)
        _write_table(context.subdir("data_snapshots") / "panel.parquet", data.frame)
        _write_table(context.subdir("data_snapshots") / "market_status.parquet", data.market_status)
        _write_table(context.subdir("data_snapshots") / "benchmark.parquet", pd.DataFrame({
            "date": data.sessions, "open": data.benchmark_open.reindex(data.sessions).to_numpy(),
            "close": data.benchmark_close.reindex(data.sessions).to_numpy(),
        }))
        ledger_root = context.subdir("backtests", fold)
        for arm, prediction in forecasts.items():
            rows = data.frame.loc[prediction.index, ["date", "stock"]].join(prediction[list(_PREDICTIONS)])
            rows["label_end"] = pd.NaT
            rows["target"] = float("nan")
            _write_table(ledger_root / f"{arm}_predictions.parquet", rows)
            outcome = simulate(
                data.frame, prediction["score"], data.sessions, data.benchmark_returns,
                run_id=run_id, cost_multiplier=settings["cost_multiplier"], initial_cash=settings["initial_cash"],
                strategy=strategies[arm], market_status=data.market_status, roster=data.roster,
            )
            verified = _verify_ledger(outcome, data.sessions)
            reports[arm] = {"metrics": outcome["metrics"], "verification": verified, "prediction_rows": len(rows)}
            for kind in latent_api.LEDGERS:
                _write_table(ledger_root / f"{arm}_{cost}_{kind}.parquet", outcome[kind])
        _write_table(context.subdir("reports") / "single_stock_summary.csv", pd.DataFrame([
            {"arm": arm, "cost": cost, "prediction_rows": info["prediction_rows"], **{
                name: info["metrics"][name] for name in ("trade_count", "order_count", "total_return", "max_drawdown", "fees")
            }} for arm, info in reports.items()
        ]))
        payload = {
            "mode": "single_stock", "cache_key": key, **identity, "run": context.to_dict(),
            "start": start, "end": end, "coverage": data.coverage, "devices": devices, "arms": reports,
            "qualification": "同一冻结规则的单股模拟复算；独立现金账户，未并入原组合或纸面留出。",
        }
        write_json(context.subdir("reports") / REPORT, payload)
        write_json(root / "status.json", {"status": "completed", "phase": "complete"})
        write_json(_index_root() / f"{key}.json", {"key": key, "run_id": run_id})
        return root, payload
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "phase": "simulation", "error": str(error)})
        raise


def generate(
    source: Path, profile: str, report: dict, *, symbol: str, fold: str, cost: str, start: str, end: str,
) -> tuple[Path, dict, bool]:
    key, identity = _key(source, profile, symbol, fold, cost)
    with _locked(key):
        saved = _cached(key)
        if saved is not None:
            folder, evidence = saved
            return folder, evidence, True
        folder, evidence = _store_run(source, profile, report, identity, key, start, end)
        return folder, evidence, False


def _downloads(report: dict) -> dict[str, str]:
    fold, cost, profile = (report[key] for key in ("source_fold", "source_cost", "source_profile"))
    if fold not in latent_api.FOLDS or cost not in {"normal", "double"} or profile not in latent_api.REPORTS:
        return {}
    allowed = {"report.json": f"reports/{REPORT}", "summary.csv": "reports/single_stock_summary.csv",
               "panel.parquet": "data_snapshots/panel.parquet", "market_status.parquet": "data_snapshots/market_status.parquet",
               "benchmark.parquet": "data_snapshots/benchmark.parquet"}
    for arm in latent_api.REPORTS[profile][1]:
        if arm in report["arms"]:
            files = [f"{arm}_predictions.parquet", *(f"{arm}_{cost}_{kind}.parquet" for kind in latent_api.LEDGERS)]
            allowed.update({f"{fold}--{file}": f"backtests/{fold}/{file}" for file in files})
    return allowed


def download_url(run_id: str, file_id: str = "") -> str:
    return f"/api/stock-chart/replays/{run_id}/downloads/{file_id}"


def download(handler: Any, run_id: str, file_id: str) -> None:
    try:
        root, report = _report(run_id)
        path = _exact(root, _downloads(report)[file_id])
    except (KeyError, OSError, ValueError) as error:
        raise latent_api.RouteError(404, "unknown_replay_artifact", "没有对应的单股回放证据。") from error
    latent_api._download(handler, path)
