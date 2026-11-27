"""Authenticated-local browser routes for explicit raw-warehouse data updates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import make_run_id
from my_strategy.research_workbench import latent_api
from my_strategy.research_workbench.data_update import PROGRESS_PREFIX, latest_complete_session

JOBS = ARTIFACT_RUNS_ROOT.parent / "data_workbench_jobs"
ACTIVE = {"queued", "running"}
TERMINAL = {"completed", "partial", "failed"}
RouteError = latent_api.RouteError


def _timestamp() -> str:
    return latent_api._timestamp()


def _job_file(job_id: str) -> Path:
    if not latent_api.ID.fullmatch(job_id):
        raise RouteError(400, "invalid_id", "更新任务编号无效。")
    return latent_api._safe(JOBS, JOBS / f"{job_id}.json")


def _result_file(job_id: str) -> Path:
    return latent_api._safe(JOBS, _job_file(job_id).with_suffix(".result.json"))


def _save(job: dict) -> None:
    path = _job_file(job["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _jobs() -> list[dict]:
    jobs = []
    for path in JOBS.glob("*.json"):
        if path.name.endswith(".result.json") or not latent_api.ID.fullmatch(path.stem):
            continue
        try:
            job = latent_api._read(_job_file(path.stem))
        except (OSError, ValueError, RouteError):
            continue
        if job.get("job_id") == path.stem and job.get("kind") == "data_update":
            jobs.append(job)
    return sorted(jobs, key=lambda item: item["created_at"], reverse=True)


def _result(job: dict) -> dict | None:
    try:
        result = latent_api._read(_result_file(job["job_id"]))
    except (OSError, ValueError):
        return None
    if (result.get("schema") != "khquant_data_update_v1"
            or result.get("job_id") != job["job_id"]
            or result.get("target_date") != job.get("target_date")
            or result.get("status") not in TERMINAL):
        return None
    return result


def _log_tail(job_id: str, limit: int = 65536) -> list[str]:
    path = _job_file(job_id).with_suffix(".log")
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace").strip().splitlines()


def _progress(job: dict) -> dict | None:
    for line in reversed(_log_tail(job["job_id"])):
        if not line.startswith(PROGRESS_PREFIX):
            continue
        try:
            value = json.loads(line[len(PROGRESS_PREFIX):])
        except ValueError:
            return None
        if isinstance(value.get("processed"), int) and isinstance(value.get("total"), int):
            return {key: value.get(key) for key in ("processed", "total", "last_code", "last_status")}
    return None


def _derive(job: dict) -> dict:
    if job.get("status") in TERMINAL:
        return job
    token = job.get("process_token")
    alive = bool(token and job.get("pid") and latent_api._process_token(job["pid"]) == token)
    if alive:
        return {**job, "status": "running", "phase": "updating"}
    result = _result(job)
    if result is not None:
        return {**job, "status": result["status"], "phase": result["status"],
                "message": result["message"], "updated_at": result["finished_at"], "result": result}
    return {**job, "status": "failed", "phase": "failed",
            "message": "更新进程已退出，未生成可用结果。", "updated_at": _timestamp()}


def _public(job: dict) -> dict:
    current = _derive(job)
    result = current.get("result")
    return {key: current.get(key) for key in (
        "job_id", "kind", "status", "phase", "message", "created_at", "updated_at", "target_date", "mode",
    )} | {
        "scope_label": "配置股票池" if job.get("mode") == "custom" else "原始仓全部股票",
        "status_url": f"/api/data/jobs/{job['job_id']}",
        "progress": _progress(job) if current.get("status") in ACTIVE else None,
        "result": result,
    }


def _finish(process: subprocess.Popen, job: dict) -> None:
    code = process.wait()
    result = _result(job) if code == 0 else None
    if result is not None:
        status, message = result["status"], result["message"]
    else:
        status = "failed"
        tail = [line for line in _log_tail(job["job_id"]) if not line.startswith(PROGRESS_PREFIX)]
        message = f"更新子进程退出码 {code}，未生成有效结果。"
        if tail:
            message += f" {tail[-1][:400]}"
    _save({**job, "status": status, "phase": status, "message": message,
           "updated_at": _timestamp(), "result": result})


def _target() -> str:
    try:
        return latest_complete_session()
    except (OSError, ValueError) as error:
        raise RouteError(503, "update_unavailable", str(error)) from error


def _submit() -> dict:
    target = _target()
    with latent_api._submission():
        if any(_derive(job)["status"] in ACTIVE for job in _jobs()):
            raise RouteError(409, "job_in_progress", "已有数据更新任务正在运行，请等待其结束。")
        job_id = make_run_id("data-update", target)
        JOBS.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": job_id, "kind": "data_update", "mode": "db", "target_date": target,
            "status": "queued", "phase": "starting", "message": "正在启动数据更新。",
            "created_at": _timestamp(), "updated_at": _timestamp(),
        }
        command = [
            sys.executable, "-B", "-m", "my_strategy.research_workbench.data_update",
            "--job-id", job_id, "--end-date", target, "--result", str(_result_file(job_id)),
        ]
        try:
            with _job_file(job_id).with_suffix(".log").open("xb") as log:
                child = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
        except OSError as error:
            _save({**job, "status": "failed", "phase": "failed", "message": str(error)})
            raise RouteError(503, "worker_unavailable", "数据更新子进程未能启动。") from error
        job.update(status="running", phase="updating", pid=child.pid,
                   process_token=latent_api._process_token(child.pid), updated_at=_timestamp())
        _save(job)
        Thread(target=_finish, args=(child, job), daemon=True).start()
        return _public(job)


def _status() -> dict:
    jobs = _jobs()
    return {
        "scope": "db",
        "scope_label": "原始仓全部股票",
        "target_date": _target(),
        "research_snapshot_date": "2026-11-20",
        "research_notice": "原始行情更新不扩展冻结研究快照，也不更新模型。",
        "job": _public(jobs[0]) if jobs else None,
    }


def _empty_body(handler: Any) -> None:
    try:
        size = int(handler.headers.get("Content-Length", "0"))
        if not 0 < size <= 1024 or handler.headers.get_content_type() != "application/json":
            raise ValueError("Invalid request body")
        body = json.loads(handler.rfile.read(size))
        if not isinstance(body, dict) or body:
            raise ValueError("Expected an empty object")
    except (ValueError, TypeError) as error:
        raise RouteError(400, "invalid_request", "数据更新仅接受空 JSON 对象。") from error


def _error(handler: Any, error: RouteError) -> None:
    handler._json(error.status, {"code": error.code, "message": str(error)})


def handle_get(handler: Any, path: str) -> bool:
    if not path.startswith("/api/data/"):
        return False
    try:
        if path == "/api/data/status":
            handler._json(200, _status())
        elif path.startswith("/api/data/jobs/"):
            job_id = path.removeprefix("/api/data/jobs/")
            if not latent_api.ID.fullmatch(job_id):
                raise RouteError(400, "invalid_id", "更新任务编号无效。")
            job = next((item for item in _jobs() if item["job_id"] == job_id), None)
            if job is None:
                raise RouteError(404, "unknown_job", "没有找到该数据更新任务。")
            handler._json(200, _public(job))
        else:
            handler._json(404, {"message": "页面不存在。"})
    except RouteError as error:
        _error(handler, error)
    return True


def handle_post(handler: Any, path: str) -> bool:
    if not path.startswith("/api/data/"):
        return False
    try:
        if path != "/api/data/updates":
            raise RouteError(404, "unknown_route", "页面不存在。")
        _empty_body(handler)
        handler._json(202, _submit())
    except RouteError as error:
        _error(handler, error)
    return True
