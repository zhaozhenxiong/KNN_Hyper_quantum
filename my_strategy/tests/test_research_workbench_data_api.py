from contextlib import nullcontext
from email.message import Message
from io import BytesIO
import json
import os
from types import SimpleNamespace

import pytest

from my_strategy.research_workbench import data_api


class Request:
    def __init__(self, data=None):
        payload = json.dumps({} if data is None else data).encode()
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = BytesIO(payload)

    def _json(self, status, payload):
        self.status = status
        self.payload = json.loads(json.dumps(payload))


def _get(path):
    request = Request()
    assert data_api.handle_get(request, path)
    return request


def _post(path, data=None):
    request = Request(data)
    assert data_api.handle_post(request, path)
    return request


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    monkeypatch.setattr(data_api, "JOBS", root)
    monkeypatch.setattr(data_api, "latest_complete_session", lambda: "2026-11-23")
    monkeypatch.setattr(data_api.latent_api, "_submission", nullcontext)
    monkeypatch.setattr(data_api.latent_api, "_process_token", lambda pid: "live-token" if pid == os.getpid() else None)
    return root


def _result(job_id, status="completed"):
    return {
        "schema": "khquant_data_update_v1", "job_id": job_id, "target_date": "2026-11-23",
        "status": status, "message": "更新结果", "finished_at": "2026-11-24T10:00:00+08:00",
        "summary": {"total": 2, "success": 1, "failed": 1, "skipped": 0, "target_success": 1},
        "before": {"latest_date": "2026-11-20"}, "after": {"latest_date": "2026-11-23"},
        "validation": {"passed": True}, "failure_examples": [],
    }


def test_get_status_is_read_only_and_returns_target_without_job(jobs):
    response = _get("/api/data/status")
    assert response.status == 200
    assert response.payload["target_date"] == "2026-11-23"
    assert response.payload["job"] is None
    assert response.payload["research_snapshot_date"] == "2026-11-20"
    assert not jobs.exists()


def test_post_has_fixed_command_active_dedupe_and_post_terminal_retry(jobs, monkeypatch):
    commands = []
    names = iter(("update-one", "update-two"))

    class Process:
        pid = os.getpid()

        def wait(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write((data_api.PROGRESS_PREFIX + json.dumps({
            "processed": 10, "total": 100, "last_code": "000001.SZ", "last_status": "success",
        }) + "\n").encode())
        return Process()

    monkeypatch.setattr(data_api, "make_run_id", lambda *args: next(names))
    monkeypatch.setattr(data_api.subprocess, "Popen", popen)
    monkeypatch.setattr(data_api, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    assert _post("/api/data/updates", {"end_date": "2026-11-24"}).status == 400
    response = _post("/api/data/updates")
    assert response.status == 202
    assert response.payload["status"] == "running"
    assert response.payload["progress"]["processed"] == 10
    assert commands[0][1:4] == ["-B", "-m", "my_strategy.research_workbench.data_update"]
    assert commands[0][commands[0].index("--end-date") + 1] == "2026-11-23"
    assert "--today-realtime" not in commands[0]
    assert _post("/api/data/updates").status == 409
    assert _get("/api/data/status").payload["job"]["status"] == "running"

    job = data_api._jobs()[0]
    data_api._result_file(job["job_id"]).write_text(json.dumps(_result(job["job_id"], "partial")), encoding="utf-8")
    data_api._finish(Process(), job)
    loaded = _get(response.payload["status_url"])
    assert loaded.status == 200
    assert loaded.payload["status"] == "partial"
    assert loaded.payload["result"]["summary"]["failed"] == 1
    assert _post("/api/data/updates").status == 202
    assert len(commands) == 2


def test_crashed_process_recovers_result_without_get_writing_job(jobs, monkeypatch):
    jobs.mkdir()
    job = {
        "job_id": "crashed-update", "kind": "data_update", "mode": "custom", "target_date": "2026-11-23",
        "status": "running", "phase": "updating", "message": "running", "created_at": "old", "updated_at": "old",
        "pid": 999999, "process_token": "old-token",
    }
    path = data_api._job_file(job["job_id"])
    data_api._save(job)
    data_api._result_file(job["job_id"]).write_text(json.dumps(_result(job["job_id"], "partial")), encoding="utf-8")
    before = path.read_bytes()
    response = _get("/api/data/jobs/crashed-update")
    assert response.status == 200
    assert response.payload["status"] == "partial"
    assert response.payload["result"]["after"]["latest_date"] == "2026-11-23"
    assert path.read_bytes() == before


def test_missing_or_failed_process_is_explicit(jobs):
    jobs.mkdir()
    job = {
        "job_id": "lost-update", "kind": "data_update", "mode": "custom", "target_date": "2026-11-23",
        "status": "running", "phase": "updating", "message": "running", "created_at": "old", "updated_at": "old",
        "pid": 999999, "process_token": "old-token",
    }
    data_api._save(job)
    response = _get("/api/data/jobs/lost-update")
    assert response.payload["status"] == "failed"
    assert "未生成可用结果" in response.payload["message"]
    assert _get("/api/data/jobs/not-found").status == 404
    assert _get("/api/data/jobs/../invalid").status == 400
