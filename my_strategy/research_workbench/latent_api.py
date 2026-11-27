"""Local research-workbench routes for Latent-Quant runs and paper research."""

from __future__ import annotations

from contextlib import contextmanager
from ctypes import wintypes
from datetime import date, datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock, Thread
from typing import Any

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, DATA_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import make_run_id
from my_strategy.latent_quant import forward
from my_strategy.latent_quant.forward_prepare import MARKET_INPUTS, MODEL_ROOT

RUNS = ARTIFACT_RUNS_ROOT
JOBS = RUNS.parent / "latent_workbench_jobs"
PAPER_INPUTS = DATA_ROOT / "raw/latent_quant_forward/inputs"
PAPER_CALENDAR = PAPER_INPUTS.parent / "calendar_20261231.csv"
PAPER_ROOT = forward.ROOT
PAPER_BASELINE = forward.BASELINE
HISTORICAL_END = "2026-11-20"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}")
FOLDS = ("oos_1", "oos_2", "oos_3", "historical_replay")
REPORTS = {
    "baseline": ("latent_quant.json", ("historical_quantiles", "supertrend_atr", "catboost_base", "catboost_supertrend_atr")),
    "overseas": ("overseas_research.json", ("domestic", "overseas_raw", "overseas_pca")),
    "domestic": ("domestic_research.json", ("domestic_fixed", "domestic_tuned")),
}
PRESETS = {
    "baseline_replay": ("baseline", "原始基线历史复跑", "my_strategy.latent_quant.runner"),
    "domestic_tune": ("domestic", "本地模型与组合调优", "my_strategy.latent_quant.domestic_runner"),
}
LEDGERS = ("daily", "orders", "positions", "trades")
CONTENT_TYPE = {
    ".csv": "text/csv; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".parquet": "application/octet-stream", ".png": "image/png", ".svg": "image/svg+xml",
}
SUBMISSION_LOCK = Lock()


class RouteError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe(root: Path, path: Path) -> Path:
    """Resolve symlinks and Windows junctions before serving an allowed file."""
    canonical = path.resolve()
    if not canonical.is_relative_to(root.resolve()):
        raise RouteError(404, "unknown_artifact", "文件不在研究工件目录内。")
    return canonical


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON document: {path.name}")
    return value


def _write_job(job: dict) -> None:
    path = JOBS / f"{job['job_id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _submission():
    """Serialize submissions across both the staging and production listeners."""
    JOBS.mkdir(parents=True, exist_ok=True)
    with SUBMISSION_LOCK, (JOBS / ".submit.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _process_token(pid: int) -> str | None:
    """Identify a live process by creation time as well as its reusable PID."""
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        time_pointer = ctypes.POINTER(wintypes.FILETIME)
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (time_pointer,) * 4
        handle = kernel.OpenProcess(0x00101000, False, pid)
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if kernel.WaitForSingleObject(handle, 0) != 258 or not kernel.GetProcessTimes(handle, *times):
                return None
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    proc = Path(f"/proc/{pid}/stat")
    if sys.platform.startswith("linux"):
        try:
            fields = proc.read_text().rsplit(")", maxsplit=1)[1].split()
            return fields[19] if fields[0] != "Z" else None
        except (OSError, IndexError):
            return None
    command = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, check=False)
    return command.stdout.strip() or None


def _job_index() -> list[dict]:
    jobs = []
    for file in JOBS.glob("*.json"):
        try:
            if (job := _read(_safe(JOBS, file))) and job.get("job_id") == file.stem:
                jobs.append(job)
        except (RouteError, OSError, ValueError):
            continue
    return sorted(jobs, key=lambda job: job["created_at"], reverse=True)


def _status(job: dict) -> dict:
    return {key: job[key] for key in ("status", "phase", "message", "created_at", "updated_at")}


def _run_folder(run_id: str) -> Path:
    if not ID.fullmatch(run_id):
        raise RouteError(400, "invalid_run_id", "运行编号无效。")
    return _safe(RUNS, RUNS / run_id)


def _worker_status(run_id: str) -> dict:
    folder = _run_folder(run_id)
    return _read(_safe(folder, folder / "status.json"))


def _reconcile(job: dict, worker: dict | None = None) -> dict:
    worker = worker or {}
    state = worker.get("status")
    if job["status"] != "failed" and state in {"completed", "failed"}:
        return {**job, "status": state, "phase": worker.get("phase") or state,
                "message": worker.get("error") or worker.get("message") or ("研究运行完成" if state == "completed" else "研究进程失败")}
    if job["status"] == "running":
        token = job.get("process_token")
        if token is None or _process_token(job["pid"]) != token:
            return {**job, "status": "failed", "phase": "failed", "message": "研究进程已退出，未返回完成状态。"}
        if job["kind"] == "run":
            return {**job, "phase": worker.get("phase") or "starting", "message": worker.get("message") or job["message"]}
    return job


def _finish(process: subprocess.Popen, job: dict) -> None:
    code = process.wait()
    status = "completed" if code == 0 else "failed"
    message = "纸面输入已冻结" if job["kind"] == "paper" else "研究运行完成"
    if code:
        log = JOBS / f"{job['job_id']}.log"
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 4096))
            tail = handle.read().decode("utf-8", errors="replace").strip().splitlines()
        message = f"子进程退出码 {code}。"
        if tail:
            message += f" {tail[-1][:400]}"
        if job["kind"] == "run":
            message = _worker_status(job["run_id"]).get("error") or message
    _write_job({**job, "status": status, "phase": status, "message": message, "updated_at": _timestamp()})


def _launch(kind: str, command: list[str], **fields: str) -> dict:
    with _submission():
        for existing in _job_index():
            if existing["kind"] == kind:
                worker = _worker_status(existing["run_id"]) if kind == "run" else {}
                if _reconcile(existing, worker)["status"] == "running":
                    raise RouteError(409, "job_in_progress", "同类任务正在运行，请等待其结束。")
        job_id = make_run_id(f"latent-web-{kind}")
        message = "正在准备当日纸面订单" if kind == "paper" else "已启动 CPU 研究进程"
        job = {
            "job_id": job_id, "kind": kind, **fields, "status": "running", "phase": "starting",
            "message": message, "created_at": _timestamp(), "updated_at": _timestamp(),
        }
        try:
            with (JOBS / f"{job_id}.log").open("xb") as log:
                child = subprocess.Popen(
                    [sys.executable, "-B", *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
        except OSError as error:
            _write_job({**job, "status": "failed", "phase": "failed", "message": str(error)})
            raise RouteError(503, "worker_unavailable", "研究子进程未能启动。") from error
        job.update(pid=child.pid, process_token=_process_token(child.pid))
        _write_job(job)
        Thread(target=_finish, args=(child, job), daemon=True).start()
        return job


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None and importlib.util.find_spec("catboost") is not None


def _profile(folder: Path) -> str | None:
    for name, (filename, _) in REPORTS.items():
        if _safe(folder, folder / "reports" / filename).is_file():
            return name
    return None


def _summary(run_id: str, job: dict | None = None) -> dict:
    folder = _run_folder(run_id)
    profile = job["profile"] if job else _profile(folder)
    if profile is None:
        raise RouteError(404, "unknown_run", "没有对应的 Latent-Quant 研究运行。")
    worker = _worker_status(run_id)
    current = _reconcile(job, worker) if job else worker
    if current.get("status") == "completed" and not _safe(folder, folder / "reports" / REPORTS[profile][0]).is_file():
        current = {**current, "status": "failed", "phase": "artifact_validation", "message": "总报告缺失。"}
    return {
        "run_id": run_id, "profile": profile, "status": current.get("status") or "incomplete",
        "phase": current.get("phase"), "message": current.get("error") or current.get("message"),
        "updated_at": current.get("updated_at"), "detail_url": f"/api/latent/runs/{run_id}",
    }


def _artifacts(folder: Path, profile: str) -> dict[str, tuple[Path, dict]]:
    items: dict[str, tuple[Path, dict]] = {}

    def add(relative: Path, label: str, kind: str, fold: str | None = None,
            arm: str | None = None, cost: str | None = None) -> None:
        expected = folder / relative
        try:
            path = _safe(folder, expected)
        except RouteError:
            return
        if path == expected and path.is_file():
            key = relative.name if fold is None else f"{fold}--{relative.name}"
            url = f"/api/latent/runs/{folder.name}/downloads/{key}"
            items[key] = (path, {"id": key, "label": label, "url": url, "filename": relative.name,
                                 "kind": kind, "fold": fold, "arm": arm, "cost": cost})

    add(Path("reports/cost_and_prediction.csv"), "预测和成本对照", "cost_summary")
    add(Path("reports/selected_hyperparameters.json"), "验证窗口胜出参数", "parameters")
    for fold in FOLDS:
        for arm in REPORTS[profile][1]:
            prefix = Path("backtests") / fold
            add(prefix / f"{arm}_predictions.parquet", "逐股预测", "predictions", fold, arm)
            for cost in ("normal", "double"):
                for kind in LEDGERS:
                    add(prefix / f"{arm}_{cost}_{kind}.parquet", f"{kind} · {cost}", kind, fold, arm, cost)
    return items


def _detail(run_id: str, job: dict | None = None) -> dict:
    detail = _summary(run_id, job)
    folder = _run_folder(run_id)
    filename = REPORTS[detail["profile"]][0]
    report = _read(_safe(folder, folder / "reports" / filename)) if detail["status"] == "completed" else {}
    if detail["status"] == "completed" and not report:
        detail.update(status="failed", phase="artifact_validation", message="子进程结束，但缺少可读取的总报告。")
    artifacts = _artifacts(folder, detail["profile"]) if report else {}
    return {**detail, "report": report or None, "downloads": [meta for _, meta in artifacts.values()],
            "job": _status(_reconcile(job, _worker_status(run_id))) if job else None}


def _calendar_path() -> Path:
    return PAPER_CALENDAR if PAPER_CALENDAR.is_file() else forward.CALENDAR


def _paper_job(jobs: list[dict]) -> dict | None:
    today = forward._now().date().isoformat()
    for job in jobs:
        if job["kind"] == "paper" and job["bundle_id"] == today:
            folder = PAPER_ROOT / "captures" / today
            worker = {"status": "completed", "message": "纸面输入已冻结"} if (folder / "manifest.json").is_file() else {}
            return {**_status(_reconcile(job, worker)), "bundle_id": today}
    return None


def _date_id(value: str) -> date:
    try:
        result = date.fromisoformat(value)
        if result.isoformat() == value:
            return result
    except (ValueError, TypeError):
        pass
    raise RouteError(400, "invalid_date", "日期必须使用 YYYY-MM-DD。")


def _bundle(bundle_id: str) -> Path:
    day = _date_id(bundle_id)
    folder = _safe(PAPER_INPUTS, PAPER_INPUTS / str(day))
    path = _safe(folder, folder / "market_bundle.json")
    payload = _read(path)
    if not payload:
        raise RouteError(404, "missing_market_bundle", "可信目录内还没有当日真实市场数据包。")
    inputs = payload.get("inputs", {})
    if payload.get("trade_date") != bundle_id or set(inputs) != set(MARKET_INPUTS):
        raise RouteError(409, "market_bundle_incomplete", "当日数据包须包含日历、行情面板、基准、交易状态与公告。")
    for role, spec in inputs.items():
        if not isinstance(spec, dict) or not all(isinstance(spec.get(key), str) and spec[key].strip()
                                                for key in ("path", "source", "observed_at")):
            raise RouteError(409, "market_bundle_incomplete", f"当日数据包缺少 {role} 的来源或时间。")
        relative = Path(spec["path"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise RouteError(404, "unknown_artifact", "市场数据包只能引用同一日期目录内的相对路径。")
        file = _safe(folder, folder / relative)
        if file.suffix.lower() not in {".csv", ".parquet"} or not file.is_file():
            raise RouteError(409, "market_bundle_incomplete", f"当日数据包缺少 {role} 的 CSV 或 Parquet。")
    return path


def _paper_state(jobs: list[dict]) -> dict:
    now = forward._now()
    today = now.date().isoformat()
    try:
        state = forward.status(root=PAPER_ROOT, baseline=PAPER_BASELINE, calendar_path=_calendar_path())
    except (OSError, ValueError, KeyError) as error:
        state = {"status": "invalid_evidence", "message": str(error), "ready_for_scoring": False}
    for key in ("acceptance_window_end", "all_signal_labels_end_if_known", "calendar_known_through"):
        if isinstance(state.get(key), date):
            state[key] = state[key].isoformat()
    if "missing_past_captures" in state:
        state["missing_past_captures"] = [str(day) for day in state["missing_past_captures"]]
    bundles = []
    try:
        _bundle(today)
        bundles.append({"id": today, "date": today, "label": f"{today} 真实盘后数据包"})
        input_message = ""
    except RouteError as error:
        input_message = str(error)
    except (OSError, ValueError, TypeError) as error:
        input_message = f"数据包读取失败：{error}"
    job = _paper_job(jobs)
    window_open = forward.CLOSE <= now.time() <= forward.DEFAULT_CUTOFF
    model_ready = (MODEL_ROOT / "baseline_models.json").is_file() and _available("my_strategy.latent_quant.forward_prepare")
    captured = (PAPER_ROOT / "captures" / today / "manifest.json").is_file()
    blocked = (
        "当日纸面证据已被冻结。" if captured else
        "当日纸面任务正在运行。" if job and job["status"] == "running" else
        "盘后 15:00–18:00 才可冻结，历史日期不能补写。" if not window_open else
        "已有前瞻快照缺失或校验不通过。" if state.get("status") in {"invalid_evidence", "missed_or_invalid_capture"} else
        "前瞻模型锁或运行依赖缺失。" if not model_ready else input_message
    )
    captures = [
        {"date": folder.name, "detail_url": f"/api/latent/paper/captures/{folder.name}"}
        for folder in sorted((PAPER_ROOT / "captures").glob("????-??-??"), reverse=True)
        if (folder / "manifest.json").is_file()
    ]
    return {"state": state, "bundles": bundles, "captures": captures, "job": job, "window_open": window_open,
            "prepare_available": not blocked and state.get("status") != "invalid_evidence", "prepare_blocker": blocked,
            "input_status": {"ready": bool(bundles), "message": input_message},
            "required_market_inputs": list(MARKET_INPUTS), "template_url": "/api/latent/paper/template"}


def _paper_detail(day: str) -> tuple[dict, dict]:
    trade_date = _date_id(day)
    folder = _safe(PAPER_ROOT, PAPER_ROOT / "captures" / day)
    if not (folder / "manifest.json").is_file():
        raise RouteError(404, "unknown_capture", "没有当日真实纸面快照。")
    valid, invalid = forward._capture_index(PAPER_ROOT, forward._policy(PAPER_BASELINE), trade_date)
    if trade_date not in valid:
        if day in invalid:
            raise RouteError(409, "invalid_capture", "当日纸面证据哈希校验失败。")
        raise RouteError(404, "unknown_capture", "没有当日真实纸面快照。")
    manifest = valid[trade_date]
    paths = {"manifest.json": _safe(folder, folder / "manifest.json")}
    for role, entry in manifest["files"].items():
        filename = entry["file"]
        suffixes = {"baseline_config": {".json"}, "baseline_roster": {".parquet"}}
        if Path(filename).stem == role and Path(filename).suffix.lower() in suffixes.get(role, {".csv", ".parquet"}):
            file = _safe(folder, folder / filename)
            if file == folder / filename:
                paths[filename] = file
    downloads = [{"id": key, "filename": key, "label": key, "kind": "capture",
                  "url": f"/api/latent/paper/captures/{day}/downloads/{key}"} for key in paths]
    return {"date": day, "manifest": manifest, "downloads": downloads}, paths


def _overview(jobs: list[dict]) -> dict:
    index = {job["run_id"]: job for job in reversed(jobs) if job["kind"] == "run"}
    for candidate in RUNS.glob("*"):
        try:
            if candidate.is_dir() and ID.fullmatch(candidate.name) and _profile(_safe(RUNS, candidate)) is not None:
                index.setdefault(candidate.name, None)
        except RouteError:
            continue
    summaries = [_summary(run_id, job) for run_id, job in index.items()]
    summaries.sort(key=lambda run: (bool(index[run["run_id"]]), run["updated_at"] or run["run_id"]), reverse=True)
    busy = any(row["status"] == "running" and index.get(row["run_id"]) for row in summaries)
    return {
        "version": 1, "historical_end": HISTORICAL_END, "runs": summaries[:50],
        "presets": [{"id": key, "label": label, "enabled": _available(module), "busy": busy}
                    for key, (_, label, module) in PRESETS.items()],
        "paper": _paper_state(jobs),
    }


def _download(handler: Any, file: Path) -> None:
    if file.suffix.lower() not in CONTENT_TYPE or not file.is_file():
        raise RouteError(404, "unknown_download", "没有此项证据下载。")
    handler._send(200, file.read_bytes(), CONTENT_TYPE[file.suffix.lower()])


def _parts(path: str) -> list[str]:
    return path.removeprefix("/api/latent/").split("/")


def handle_get(handler: Any, path: str) -> bool:
    if not path.startswith("/api/latent/"):
        return False
    try:
        parts = _parts(path)
        if parts == ["overview"]:
            handler._json(200, _overview(_job_index()))
        elif parts == ["paper", "template"]:
            template = forward._template()
            for key in ("inputs", "required_columns"):
                template[key] = {role: item for role, item in template[key].items() if role in MARKET_INPUTS}
            handler._json(200, template)
        elif len(parts) in (2, 4) and parts[0] == "runs":
            run_id = parts[1]
            job = next((item for item in _job_index() if item.get("run_id") == run_id), None)
            if len(parts) == 2:
                handler._json(200, _detail(run_id, job))
            elif parts[2] == "downloads":
                summary = _summary(run_id, job)
                items = _artifacts(_run_folder(run_id), summary["profile"]) if summary["status"] == "completed" else {}
                if parts[3] not in items:
                    raise RouteError(404, "unknown_download", "没有此项证据下载。")
                _download(handler, items[parts[3]][0])
            else:
                raise RouteError(404, "unknown_route", "没有此 Latent-Quant 接口。")
        elif len(parts) in (3, 5) and parts[:2] == ["paper", "captures"]:
            detail, files = _paper_detail(parts[2])
            if len(parts) == 3:
                handler._json(200, detail)
            elif parts[3] == "downloads" and parts[4] in files:
                _download(handler, files[parts[4]])
            else:
                raise RouteError(404, "unknown_download", "没有此项纸面证据下载。")
        else:
            raise RouteError(404, "unknown_route", "没有此 Latent-Quant 接口。")
    except RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, ValueError, KeyError, TypeError) as error:
        handler._json(500, {"message": f"研究证据读取失败：{error}", "code": "artifact_read_failed"})
    return True


def _body(handler: Any, key: str) -> str:
    try:
        size = int(handler.headers.get("Content-Length", 0))
        if not 0 < size <= 1024 or handler.headers.get_content_type() != "application/json":
            raise ValueError("Invalid request body")
        value = json.loads(handler.rfile.read(size))
        if not isinstance(value, dict) or set(value) != {key} or not isinstance(value[key], str):
            raise ValueError("Unexpected request fields")
        return value[key]
    except (ValueError, TypeError) as error:
        raise RouteError(400, "invalid_request", f"请求仅接受 JSON 字段 {key}。") from error


def handle_post(handler: Any, path: str) -> bool:
    if not path.startswith("/api/latent/"):
        return False
    try:
        if path == "/api/latent/runs":
            preset = _body(handler, "preset")
            if preset not in PRESETS:
                raise RouteError(400, "unknown_preset", "请选择固定研究预设。")
            profile, _, module = PRESETS[preset]
            if not _available(module):
                raise RouteError(503, "missing_dependency", "预设的本地运行环境尚未就绪。")
            run_id = make_run_id(f"latent-web-{profile}", HISTORICAL_END)
            job = _launch("run", ["-m", module, "--run-id", run_id], run_id=run_id, preset=preset, profile=profile)
            handler._json(202, {"run_id": run_id, "status": job["status"], "message": job["message"],
                                "detail_url": f"/api/latent/runs/{run_id}", "job": _status(job)})
        elif path == "/api/latent/paper/prepare":
            bundle_id = _body(handler, "bundle_id")
            now = forward._now()
            if _date_id(bundle_id) != now.date() or not forward.CLOSE <= now.time() <= forward.DEFAULT_CUTOFF:
                raise RouteError(409, "outside_capture_window", "只接受当日北京时间 15:00–18:00 的真实数据包。")
            if (PAPER_ROOT / "captures" / bundle_id / "manifest.json").is_file():
                raise RouteError(409, "already_captured", "当日已冻结，不能覆盖。")
            path = _bundle(bundle_id)
            module = "my_strategy.latent_quant.forward_prepare"
            if not (MODEL_ROOT / "baseline_models.json").is_file() or not _available(module):
                raise RouteError(503, "missing_dependency", "前瞻模型锁或进程环境尚未就绪。")
            args = ["-m", module, "--market-bundle", str(path), "--calendar", str(_calendar_path()),
                    "--baseline", str(PAPER_BASELINE), "--models", str(MODEL_ROOT), "--root", str(PAPER_ROOT)]
            job = _launch("paper", args, bundle_id=bundle_id)
            handler._json(202, {"bundle_id": bundle_id, "status": job["status"],
                                "message": job["message"], "job": _status(job)})
        else:
            raise RouteError(404, "unknown_route", "没有此 Latent-Quant 接口。")
    except RouteError as error:
        handler._json(error.status, {"message": str(error), "code": error.code})
    except (OSError, ValueError, KeyError, TypeError) as error:
        handler._json(500, {"message": f"研究任务启动失败：{error}", "code": "job_launch_failed"})
    return True
