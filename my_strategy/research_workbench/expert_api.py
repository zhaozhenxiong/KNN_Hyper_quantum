"""Local job and result routes for the integrated expert research workflow."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Thread

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import make_run_id, stable_hash
from my_strategy.model_registry.expert_release import chart_release, current_strategy, record_candidate
from my_strategy.research_workbench import latent_api
from my_strategy.research_workbench.runner import write_json

RUNS = ARTIFACT_RUNS_ROOT
JOBS = RUNS.parent / "expert_workbench_jobs"
RESEARCH_REPORT = "reports/expert_research.json"
REPLAY_REPORT = "reports/expert_stock_replay.json"
RouteError = latent_api.RouteError
PRESET = {"id": "frozen_120", "label": "120 股冻结样本", "start": "2025-03-03", "end": "2026-07-28"}
FULL_END = "2026-11-20"
STUDY_FIELDS = {"preset", "start", "end", "pool", "boards", "symbols", "stock_cap", "seed", "commission_bps",
                "universe_policy", "sector_view", "sector_provider", "rebalance_days", "top_n", "max_weight", "max_industry_weight",
                "knn_device", "knn_query_chunk"}
STRATEGY_BOUNDS = {"rebalance_days": (1, 30, int), "top_n": (1, 100, int),
                   "max_weight": (0.01, 1.0, float), "max_industry_weight": (0.01, 1.0, float)}
EXPERIMENT_SCHEMA = 4
REPLAY_FIELDS = {"release_id", "symbol", "start", "end", "cost", "sector_provider"}
ACCOUNT_SETTINGS = {"start", "end", "cost", "initial_cash", "commission_bps", "slippage_bps", "sector_provider"}
STOCK_CODE = re.compile(r"\d{6}\.(SH|SZ|BJ)")
RESEARCH_ARMS = {"domestic_gpu", "fusion", "online_knn_volume_trend", "knn60_multimodal40"}
ACTIVE = {"queued", "running"}


def _folder(run_id: str) -> Path:
    if not latent_api.ID.fullmatch(run_id):
        raise RouteError(400, "invalid_id", "运行编号无效。")
    return latent_api._safe(RUNS, RUNS / run_id)


def _job_file(job_id: str) -> Path:
    if not latent_api.ID.fullmatch(job_id):
        raise RouteError(400, "invalid_id", "任务编号无效。")
    return latent_api._safe(JOBS, JOBS / f"{job_id}.json")


def _save_job(job: dict) -> None:
    write_json(_job_file(job["job_id"]), job)


def _status(job: dict) -> dict:
    state = job["status"]
    report = RESEARCH_REPORT if job["kind"] == "research" else REPLAY_REPORT
    folder = _folder(job["run_id"])
    progress = latent_api._read(folder / "status.json")
    phase = progress.get("phase") or state
    message = progress.get("message") or job["message"]
    if state in ACTIVE and (not job.get("process_token") or latent_api._process_token(job["pid"]) != job["process_token"]):
        state = "completed" if (folder / report).is_file() and progress.get("status") == "completed" else "failed"
        if state == "failed":
            phase = "failed"
            message = progress.get("error") or "任务已结束，未生成可用结果。"
    if state in {"completed", "failed"}:
        phase = state
        message = (progress.get("error") or job["message"]) if state == "failed" else "研究结果已就绪。"
    route = "runs" if job["kind"] == "research" else "replays"
    return {key: job[key] for key in ("job_id", "run_id", "kind", "created_at")} | {
        "status": state, "phase": phase, "message": message,
        "reused_predictions_from": job.get("reuse_run_id"),
        "status_url": f"/api/expert/jobs/{job['job_id']}", "result_url": f"/api/expert/{route}/{job['run_id']}",
    }


def _jobs() -> list[dict]:
    jobs = []
    for file in JOBS.glob("*.json"):
        if not latent_api.ID.fullmatch(file.stem):
            continue
        job = latent_api._read(_job_file(file.stem))
        if job.get("job_id") == file.stem and job.get("kind") in {"research", "replay"}:
            jobs.append(job)
    return sorted(jobs, key=lambda item: item["created_at"], reverse=True)


def _finish(process: subprocess.Popen, job: dict) -> None:
    code = process.wait()
    report = RESEARCH_REPORT if job["kind"] == "research" else REPLAY_REPORT
    complete = code == 0 and (_folder(job["run_id"]) / report).is_file()
    _save_job(job | {"status": "completed" if complete else "failed",
                     "message": "研究结果已就绪。" if complete else f"任务退出码 {code}，请查看运行日志。"})


def _submit(kind: str, spec: dict, identity: dict, as_of: str) -> dict:
    key = stable_hash({"kind": kind, "identity": identity})
    with latent_api._submission():
        running = 0
        reuse_run_id = None
        for previous in _jobs():
            if previous["kind"] != kind:
                continue
            prior = _status(previous)
            if previous["key"] == key and prior["status"] in {*ACTIVE, "completed"}:
                return prior | {"cached": True}
            running += prior["status"] in ACTIVE
            reusable_failure = (kind == "research" and prior["status"] == "failed"
                                and (_folder(previous["run_id"]) / RESEARCH_REPORT).is_file())
            if (kind == "research" and reuse_run_id is None
                    and (prior["status"] == "completed" or reusable_failure)
                    and spec.get("prediction_key") and previous["spec"].get("prediction_key") == spec["prediction_key"]):
                reuse_run_id = previous["run_id"]
        if running >= (1 if kind == "research" else 2):
            raise RouteError(409, "job_in_progress", "已有同类任务运行中，请稍后重试。")
        run_id = make_run_id(f"expert-{kind}", as_of)
        JOBS.mkdir(parents=True, exist_ok=True)
        job = {"job_id": run_id, "run_id": run_id, "kind": kind, "key": key, "spec": spec, "reuse_run_id": reuse_run_id,
               "status": "queued", "message": "正在启动任务。", "created_at": latent_api._timestamp()}
        _save_job(job)
        try:
            with (JOBS / f"{run_id}.log").open("xb") as log:
                child = subprocess.Popen(
                    [sys.executable, "-B", "-m", __name__, "--worker", run_id], cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
        except OSError as error:
            _save_job(job | {"status": "failed", "message": "研究进程未能启动。"})
            raise RouteError(503, "worker_unavailable", "研究进程未能启动。") from error
        job.update(status="running", pid=child.pid, process_token=latent_api._process_token(child.pid))
        _save_job(job)
        Thread(target=_finish, args=(child, job), daemon=True).start()
        return _status(job) | {"cached": False}


def _json_body(handler, *, limit: int) -> dict:
    try:
        size = int(handler.headers.get("Content-Length", "0"))
        if not 0 < size <= limit or handler.headers.get_content_type() != "application/json":
            raise ValueError("Invalid body size or content type")
        body = json.loads(handler.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("Expected JSON object")
        return body
    except (ValueError, TypeError) as error:
        raise RouteError(400, "invalid_request", "请求字段或 JSON 格式无效。") from error


def _body(handler, fields: set[str]) -> dict:
    body = _json_body(handler, limit=4096)
    if set(body) != fields or not all(isinstance(value, str) for value in body.values()):
        raise RouteError(400, "invalid_request", "请求字段无效。")
    return body


def _basis_points(value, *, field: str, label: str) -> str:
    try:
        raw = str(value)
        if not re.fullmatch(r"[0-9]{1,10}(?:\.[0-9]{1,8})?", raw):
            raise ValueError("Invalid commission")
        rate = Decimal(raw)
        if not 0 <= rate < 10000:
            raise ValueError("Invalid commission")
        return format(rate.normalize(), "f")
    except (ValueError, InvalidOperation) as error:
        raise RouteError(400, f"invalid_{field}", f"{label}使用万分之数字，需大于等于 0 且小于 10000。") from error


def _commission(value) -> str:
    return _basis_points(value, field="commission", label="佣金")


def _study_spec(body: dict) -> dict:
    from my_strategy.latent_quant.runner import PORTFOLIO
    from my_strategy.research_workbench.expert_config import normalize_expert_spec

    if set(body) - STUDY_FIELDS or body.get("preset", PRESET["id"]) not in (PRESET["id"], "custom"):
        raise RouteError(400, "invalid_research_settings", "研究配置包含未知字段或预设。")
    data = {key: value for key, value in body.items()
            if key not in {"preset", "commission_bps", "knn_device", "knn_query_chunk", *STRATEGY_BOUNDS}}
    default_end = FULL_END if data.get("pool") == "full" else PRESET["end"]
    data = {"pool": "frozen_120", "start": PRESET["start"], "end": default_end, **data}
    fixed = (data["pool"] == "frozen_120" and data.get("seed", 42) == 42
             and (data["start"], data["end"]) == (PRESET["start"], PRESET["end"]))
    if fixed:
        data["preset"] = "frozen_v2"
    spec = normalize_expert_spec(data)
    for key, (minimum, maximum, kind) in STRATEGY_BOUNDS.items():
        value = body.get(key, getattr(PORTFOLIO, key))
        allowed = (int, float) if kind is float else (int,)
        if isinstance(value, bool) or not isinstance(value, allowed) or not minimum <= value <= maximum:
            raise RouteError(400, "invalid_strategy_settings", f"{key} 需在 {minimum} 至 {maximum} 之间。")
        spec[key] = value
    spec["commission_bps"] = _commission(body.get("commission_bps", "3"))
    device = body.get("knn_device", "cuda")
    chunk = body.get("knn_query_chunk", 256)
    if device not in {"auto", "cpu", "cuda"} or type(chunk) is not int or not 1 <= chunk <= 4096:
        raise RouteError(400, "invalid_knn_device", "KNN 设备或查询分块无效。")
    spec["knn_device"], spec["knn_query_chunk"] = device, chunk
    return spec


def _study_defaults() -> dict:
    from my_strategy.latent_quant.execution import StudyCostModel
    from my_strategy.latent_quant.runner import PORTFOLIO

    return {"pool": "full", "start": PRESET["start"], "end": FULL_END, "frozen_end": PRESET["end"], "seed": 42,
            "stock_cap": 120, "commission_bps": "3", "boards": ["sh_main", "sz_main", "star"], "sector_view": False,
            "knn_device": "cuda", "knn_query_chunk": 256,
            "base_cost": asdict(StudyCostModel()),
            **{key: getattr(PORTFOLIO, key) for key in STRATEGY_BOUNDS}}


def _replay_spec(body: dict, *, diagnostic: bool = False) -> dict:
    provider = body.get("sector_provider", "ths_fuyao")
    if provider not in {"capco", "ths_fuyao"}:
        raise RouteError(400, "invalid_sector_provider", "板块数据提供方无效。")
    body = {**body, "sector_provider": provider}
    try:
        start, end = date.fromisoformat(body["start"]), date.fromisoformat(body["end"])
        if (start.isoformat() != body["start"] or end.isoformat() != body["end"] or start > end
                or (end - start).days > 3653 or body["cost"] not in {"normal", "double"}
                or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", body["symbol"])):
            raise ValueError("Invalid dates, cost or security")
    except (ValueError, TypeError) as error:
        raise RouteError(400, "invalid_scope", "股票、日期或成本范围无效。") from error
    if diagnostic:
        source_run = body["run_id"]
        folder, report = _result(source_run, "runs")
        if (report.get("run", {}).get("run_id") != source_run or report.get("decision", {}).get("state") != "research_only"
                or latent_api._read(folder / "status.json").get("status") != "completed"):
            raise RouteError(409, "diagnostic_unavailable", "没有可复核的候选研究结果。")
        arm = report.get("final_portfolio", {}).get("arm", "fusion")
        if arm not in RESEARCH_ARMS:
            raise RouteError(409, "diagnostic_unavailable", "没有可复核的主策略账本。")
        release = {"status": "candidate", "source_run_id": source_run, "release_id": source_run, "arm": arm}
        return body | {"diagnostic": True, "release": release, **{name: release[name] for name in ("source_run_id", "release_id", "arm")}}
    release = chart_release()
    if release["status"] != "approved":
        raise RouteError(409, "release_unavailable", release["reason"])
    if body["release_id"] != release["release_id"]:
        raise RouteError(409, "release_changed", "当前发布版本已变化，请刷新后重跑。")
    return body | {"release": release, "source_run_id": release["source_run_id"], "arm": release["arm"]}


def _pinned_strategy(source_run_id: str | None, model_version: str | None) -> dict:
    if source_run_id is None:
        if model_version is not None:
            raise RouteError(400, "invalid_model_context", "模型版本必须同时指定来源研究运行。")
        return current_strategy()
    if not latent_api.ID.fullmatch(source_run_id) or model_version is not None and not isinstance(model_version, str):
        raise RouteError(400, "invalid_model_context", "来源研究运行或模型版本无效。")
    folder, report = _result(source_run_id, "runs")
    if (report.get("run", {}).get("run_id") != source_run_id
            or report.get("decision", {}).get("state") != "research_only"
            or latent_api._read(folder / "status.json").get("status") != "completed"):
        raise RouteError(409, "diagnostic_unavailable", "没有可复核的候选研究结果。")
    arm = report.get("final_portfolio", {}).get("arm", "fusion")
    if arm not in RESEARCH_ARMS:
        raise RouteError(409, "diagnostic_unavailable", "没有可复核的主策略账本。")
    manifest_path = folder / "reports/unified_expert.json"
    manifest = None
    if manifest_path.is_file():
        from .expert_final_fit import verify_final_fit

        try:
            manifest = verify_final_fit(folder)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RouteError(409, "model_closure_invalid", "统一模型包或研究报告哈希校验失败。") from error
    if model_version is not None and (not manifest or manifest.get("model_version") != model_version):
        raise RouteError(409, "model_version_mismatch", "统一模型版本与来源研究运行不一致。")
    if manifest and manifest.get("run_id") != source_run_id:
        raise RouteError(409, "model_version_mismatch", "统一模型清单与来源研究运行不一致。")
    coverage = report.get("input_coverage", {})
    from .expert_replay import _sha

    return {
        "status": "research_only", "run_id": source_run_id, "source_run_id": source_run_id,
        "release_id": source_run_id, "model_id": source_run_id, "model_version": manifest.get("model_version") if manifest else None,
        "arm": arm, "label": "多模态专家 · 指定统一研究候选",
        "research_report_sha256": manifest["research_report_sha256"] if manifest else _sha(folder / RESEARCH_REPORT),
        "analysis_start": coverage.get("first_evaluation_session"), "as_of": coverage.get("last_evaluation_session"),
    }


def _current_replay_spec(body: dict, *, single: bool) -> dict:
    field = "symbol" if single else "symbols"
    if set(body) - (ACCOUNT_SETTINGS | {field, "source_run_id", "model_version"}) or not {"start", "end", field} <= body.keys():
        raise RouteError(400, "invalid_request", "请提供股票及日期、有效费用和模型上下文。")
    try:
        start, end = (date.fromisoformat(body[key]) for key in ("start", "end"))
        cost = body.get("cost", "normal")
        if ((start.isoformat(), end.isoformat()) != (body["start"], body["end"])
                or start > end or (end - start).days > 3653 or cost not in ("normal", "double")):
            raise ValueError("Invalid dates or cost scenario")
        values = [body[field]] if single else body[field]
        if not isinstance(values, list) or len(values) > 10000:
            raise ValueError("Invalid watchlist")
        symbols = []
        for value in values:
            if not isinstance(value, str) or not STOCK_CODE.fullmatch(value.strip().upper()):
                raise ValueError("Invalid stock symbol")
            symbols.append(value.strip().upper())
        symbols = sorted(set(symbols))
    except (TypeError, ValueError) as error:
        raise RouteError(400, "invalid_scope", "股票、日期或成本范围无效。") from error

    settings = {"sector_provider": body.get("sector_provider", "ths_fuyao")}
    if settings["sector_provider"] not in {"capco", "ths_fuyao"}:
        raise RouteError(400, "invalid_sector_provider", "板块数据提供方无效。")
    if "initial_cash" in body:
        cash = body["initial_cash"]
        if type(cash) not in (int, float) or not 0 < cash <= 1_000_000_000_000:
            raise RouteError(400, "invalid_cash", "初始资金需大于 0 且不超过 1 万亿元。")
        settings["initial_cash"] = float(cash)
    if "commission_bps" in body:
        settings["commission_bps"] = _commission(body["commission_bps"])
    if "slippage_bps" in body:
        settings["slippage_bps"] = _basis_points(body["slippage_bps"], field="slippage", label="滑点")

    strategy = _pinned_strategy(body.get("source_run_id"), body.get("model_version"))
    if strategy["status"] == "unavailable":
        raise RouteError(409, "strategy_unavailable", strategy["reason"])
    diagnostic = strategy["status"] == "research_only"
    release = {key: strategy[key] for key in ("source_run_id", "release_id", "arm")}
    detail = release | {"status": "candidate"} if diagnostic else (
        strategy["publication"] if strategy["status"] == "approved" else strategy
    )
    if diagnostic and strategy.get("research_report_sha256"):
        detail["research_report_sha256"] = strategy["research_report_sha256"]
    return {
        "scope": "single_stock" if single else "watchlist", field: symbols[0] if single else symbols,
        "start": start.isoformat(), "end": end.isoformat(), "cost": cost, "diagnostic": diagnostic,
        "release": detail, "model_id": strategy["model_id"], "model_version": strategy.get("model_version"),
        **release, **settings,
    }


def _submit_candidates(body: dict) -> dict:
    if set(body) != {"backtest_run_id"} or not isinstance(body["backtest_run_id"], str) \
            or not latent_api.ID.fullmatch(body["backtest_run_id"]):
        raise RouteError(400, "invalid_request", "请提供一个已完成的全池回测编号。")
    parent_id = body["backtest_run_id"]
    parent = next((job for job in _jobs() if job["run_id"] == parent_id and job["kind"] == "replay"), None)
    if parent is None or _status(parent)["status"] != "completed":
        raise RouteError(409, "backtest_incomplete", "父回测不存在或尚未完成。")
    _, report = _result(parent_id, "replays")
    source = report.get("source", {})
    if source.get("scope") != "watchlist" or source.get("plan_next") or "full_pool" not in report.get("accounts", {}):
        raise RouteError(409, "invalid_parent_backtest", "次日候选必须引用未生成预申报的全池组合回测。")
    spec = copy.deepcopy(parent["spec"])
    spec.pop("input_identity", None)
    spec.update(plan_next=True, candidate_of=parent_id)
    from my_strategy.research_workbench.expert_replay import replay_identity

    identity = replay_identity(spec)
    return _submit("replay", spec | {"input_identity": identity}, identity, spec["end"])


def _result(run_id: str, kind: str) -> tuple[Path, dict]:
    folder = _folder(run_id)
    report = RESEARCH_REPORT if kind == "runs" else REPLAY_REPORT
    file = latent_api._safe(folder, folder / report)
    payload = latent_api._read(file)
    if not payload:
        raise RouteError(404, "unknown_result", "没有此专家任务结果。")
    return folder, payload


def _download(handler, parts: list[str], folder: Path, report: dict) -> None:
    file_id = parts[-1]
    individual = {"report.json": RESEARCH_REPORT if parts[0] == "runs" else REPLAY_REPORT,
                  "costs.csv": "reports/cost_and_prediction.csv", "parameters.json": "reports/selected_hyperparameters.json"}
    relative = individual.get(file_id)
    if not relative and (match := re.fullmatch(
        r"(oos_[1-3]|historical_replay)--((fusion|domestic_gpu|online_knn_volume_trend|knn60_multimodal40)_(?:normal|double)_(?:daily|orders|positions|trades)\.parquet)", file_id,
    )):
        reported = [*report.get("folds", []), report.get("historical_review") or {}]
        if any(fold.get("period") == match[1] and match[3] in fold.get("arms", {}) for fold in reported):
            relative = f"backtests/{match[1]}/{match[2]}"
    if not relative:
        raise RouteError(404, "unknown_download", "没有此证据文件。")
    latent_api._download(handler, latent_api._safe(folder, folder / relative))


def handle_get(handler, path: str, query: str = "") -> bool:
    if not path.startswith("/api/expert/"):
        return False
    try:
        parts = path.removeprefix("/api/expert/").split("/")
        if parts == ["release"]:
            handler._json(200, chart_release())
        elif parts == ["current"]:
            handler._json(200, current_strategy())
        elif parts == ["overview"]:
            latest = next((job for job in _jobs() if job["kind"] == "research"), None)
            handler._json(200, {"release": chart_release(), "preset": PRESET, "study_defaults": _study_defaults(),
                                "latest_job": _status(latest) if latest else None})
        elif len(parts) == 2 and parts[0] == "jobs":
            job = latent_api._read(_job_file(parts[1]))
            if not job or job.get("job_id") != parts[1]:
                raise RouteError(404, "unknown_job", "没有此专家任务。")
            handler._json(200, _status(job))
        elif len(parts) in {3, 4} and parts[0] == "runs" and parts[2] == "explain":
            from my_strategy.research_workbench.expert_explain import explain

            _result(parts[1], "runs")
            symbol = parts[3] if len(parts) == 4 else None
            if symbol and not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", symbol):
                raise RouteError(400, "invalid_symbol", "请输入标准股票代码。")
            handler._json(200, explain(parts[1], symbol=symbol))
        elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "forward-status":
            from my_strategy.research_workbench.expert_forward import status

            _result(parts[1], "runs")
            handler._json(200, status(parts[1]))
        elif len(parts) == 3 and parts[0] == "replays" and parts[2] == "candidates":
            from my_strategy.research_workbench.expert_candidates import read_replay

            handler._json(200, read_replay(_folder(parts[1])))
        elif len(parts) == 3 and parts[0] == "replays" and parts[2] == "holdings":
            from my_strategy.research_workbench.expert_backtest_holdings import holdings

            handler._json(200, holdings(parts[1], query))
        elif len(parts) == 3 and parts[0] == "replays" and parts[2] == "activity":
            from my_strategy.research_workbench.expert_backtest_activity import activity

            handler._json(200, activity(parts[1], query))
        elif len(parts) in {3, 5} and parts[0] == "replays" and parts[2] == "sectors":
            from my_strategy.research_workbench import expert_sector

            if len(parts) == 3:
                handler._json(200, expert_sector.overview(parts[1], query))
            elif parts[4] == "chart":
                handler._json(200, expert_sector.chart(parts[1], parts[3], query))
            elif parts[4] == "candidates":
                handler._json(200, expert_sector.candidates(parts[1], parts[3], query))
            else:
                raise RouteError(404, "unknown_route", "没有此板块接口。")
        elif len(parts) in {2, 4} and parts[0] in {"runs", "replays"}:
            folder, data = _result(parts[1], parts[0])
            if len(parts) == 2:
                if parts[0] == "runs":
                    from my_strategy.research_workbench.expert_score_health import inspect_archive

                    data = dict(data)
                    unified = folder / "reports/unified_expert.json"
                    if unified.is_file():
                        data["unified"] = latent_api._read(unified)
                    try:
                        data["score_health"] = inspect_archive(folder, data)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        data["score_health"] = {"status": "unavailable", "reason": str(error)}
                handler._json(200, data)
            elif parts[2] == "downloads":
                if parts[0] == "replays":
                    from my_strategy.research_workbench.expert_replay import download

                    download(handler, parts[1], parts[3])
                else:
                    _download(handler, parts, folder, data)
            else:
                raise RouteError(404, "unknown_route", "没有此专家接口。")
        else:
            raise RouteError(404, "unknown_route", "没有此专家接口。")
    except RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, ValueError, KeyError, TypeError) as error:
        handler._json(500, {"message": f"专家结果读取失败。{error}", "code": "expert_read_failed"})
    return True


def handle_post(handler, path: str) -> bool:
    if not path.startswith("/api/expert/"):
        return False
    try:
        if path in {"/api/expert/runs", "/api/expert/preflight"}:
            from my_strategy.research_workbench.expert_config import expert_preflight
            from my_strategy.research_workbench.expert_hybrid_study import prediction_identity

            spec = _study_spec(_json_body(handler, limit=16384))
            preview = expert_preflight({key: value for key, value in spec.items()
                                        if key not in {*STRATEGY_BOUNDS, "knn_device", "knn_query_chunk"}})
            if path.endswith("/preflight"):
                handler._json(200, preview | {"spec": spec})
                return True
            if preview["status"] == "blocked":
                details = [*preview.get("blockers", []), *(f"{stock} · {reason}" for stock, reason in preview["rejections"].items())]
                raise RouteError(422, "preflight_blocked", "研究预检未通过。" + "；".join(details[:8]))
            inputs = prediction_identity(spec)
            prepared = {"engine": "hybrid", "research": spec, "prediction_identity": inputs}
            identity = {"schema": EXPERIMENT_SCHEMA, "engine": "hybrid", "predictions": inputs, "research": spec}
            job = _submit("research", prepared, identity, spec["end"])
        elif path == "/api/expert/candidates":
            job = _submit_candidates(_json_body(handler, limit=4096))
        elif path in {"/api/expert/stock-replays", "/api/expert/portfolio-replays"}:
            from my_strategy.research_workbench.expert_replay import replay_identity

            spec = _current_replay_spec(_json_body(handler, limit=131072), single=path.endswith("/stock-replays"))
            identity = replay_identity(spec)
            job = _submit("replay", spec | {"input_identity": identity}, identity, spec["end"])
        elif path in {"/api/expert/replays", "/api/expert/diagnostic-replays"}:
            from my_strategy.research_workbench.expert_replay import replay_identity

            diagnostic = path.endswith("/diagnostic-replays")
            fields = (REPLAY_FIELDS - {"release_id"}) | {"run_id"} if diagnostic else REPLAY_FIELDS
            body = _body(handler, fields)
            spec = _replay_spec(body, diagnostic=diagnostic)
            identity = replay_identity(spec)
            job = _submit("replay", spec | {"input_identity": identity}, identity, body["end"])
        else:
            raise RouteError(404, "unknown_route", "没有此专家接口。")
        handler._json(200 if job["status"] == "completed" else 202, job)
    except RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, ValueError, KeyError, TypeError) as error:
        handler._json(422, {"message": f"专家任务无法执行。{error}", "code": "expert_inputs_unavailable"})
    return True


def _work(job_id: str) -> None:
    job = latent_api._read(_job_file(job_id))
    folder = _folder(job_id)
    try:
        if job["kind"] == "research":
            saved = job["spec"]
            spec = saved.get("research")
            if saved.get("engine") in {"online_knn", "hybrid"}:
                if saved.get("engine") == "hybrid":
                    from my_strategy.research_workbench.expert_hybrid_study import prediction_identity, run
                else:
                    from my_strategy.research_workbench.expert_knn_study import prediction_identity, run

                if prediction_identity(spec) != saved["prediction_identity"]:
                    raise ValueError("输入文件在任务领取前变更，请重新提交。")
                report = run(run_id=job_id, spec=spec)
                from my_strategy.model_registry import expert_fallback

                activation = expert_fallback.activate(folder)
                job["activation"] = activation
            else:
                from my_strategy.research_workbench.expert_research import prediction_identity, research_inputs, run

                if spec is not None:
                    if prediction_identity(spec) != saved["prediction_identity"]:
                        raise ValueError("输入文件在任务领取前变更，请重新提交。")
                    report = run(run_id=job_id, spec=spec, reuse_run_id=job.get("reuse_run_id"))
                else:
                    if research_inputs() != saved["input_identity"]:
                        raise ValueError("输入文件在任务领取前变更，请重新提交。")
                    report = run(run_id=job_id)
                record_candidate(job_id, Path(report))
        else:
            from my_strategy.research_workbench.expert_replay import run

            run(run_id=job_id, spec=job["spec"])
        write_json(folder / "status.json", {"run_id": job_id, "status": "completed", "phase": "completed"})
    except Exception as error:
        write_json(folder / "status.json", {"run_id": job_id, "status": "failed", "phase": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    _work(parser.parse_args().worker)
