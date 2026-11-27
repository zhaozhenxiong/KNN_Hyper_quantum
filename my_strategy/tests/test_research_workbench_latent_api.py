from datetime import date, datetime
from email.message import Message
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_strategy.research_workbench import latent_api


class Request:
    def __init__(self, data=None):
        payload = json.dumps(data).encode()
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = BytesIO(payload)

    def _json(self, status, payload):
        self.status = status
        self.payload = json.loads(json.dumps(payload))

    def _send(self, status, content, content_type):
        self.status, self.payload, self.content_type = status, content, content_type


def _get(path):
    request = Request()
    assert latent_api.handle_get(request, path)
    return request


def _post(path, payload):
    request = Request(payload)
    assert latent_api.handle_post(request, path)
    return request


@pytest.fixture
def folders(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    jobs = tmp_path / "jobs"
    inputs = tmp_path / "inputs"
    paper = tmp_path / "forward"
    monkeypatch.setattr(latent_api, "RUNS", runs)
    monkeypatch.setattr(latent_api, "JOBS", jobs)
    monkeypatch.setattr(latent_api, "PAPER_INPUTS", inputs)
    monkeypatch.setattr(latent_api, "PAPER_ROOT", paper)
    monkeypatch.setattr(latent_api, "_available", lambda module: True)
    monkeypatch.setattr(latent_api.forward, "status", lambda **options: {
        "status": "awaiting_market_close", "calendar_known_through": date(2026, 12, 31),
        "missing_past_captures": [], "valid_daily_captures": 0, "target_sessions": 60,
    })
    return runs, jobs, inputs, paper


def _write(path: Path, value: dict):
    path.write_text(json.dumps(value), encoding="utf-8")


def _history(runs, name, profile="baseline"):
    folder = runs / name
    (folder / "reports").mkdir(parents=True)
    _write(folder / "status.json", {"status": "completed", "phase": "complete"})
    _write(folder / "reports" / latent_api.REPORTS[profile][0], {"run": {"run_id": name}, "decision": {}})
    return folder


def test_catalog_serves_supported_reports_and_only_whitelisted_downloads(folders, tmp_path):
    runs, jobs, _, _ = folders
    first = _history(runs, "baseline-result")
    _history(runs, "domestic-tune", "domestic")
    _history(runs, "overseas-compare", "overseas")
    (first / "reports/notes.csv").write_bytes(b"unlisted")
    (first / "reports/cost_and_prediction.csv").write_bytes(b"fold,metric\noos_1,1\n")
    overview = _get("/api/latent/overview")
    assert overview.status == 200
    assert overview.payload["paper"]["state"]["calendar_known_through"] == "2026-12-31"
    assert {run["profile"] for run in overview.payload["runs"]} == {"baseline", "domestic", "overseas"}
    assert not jobs.exists()
    detail = _get("/api/latent/runs/baseline-result")
    assert detail.status == 200
    assert [item["filename"] for item in detail.payload["downloads"]] == ["cost_and_prediction.csv"]
    artifact = detail.payload["downloads"][0]["url"]
    assert _get(artifact).payload == b"fold,metric\noos_1,1\n"
    assert _get("/api/latent/runs/baseline-result/downloads/notes.csv").status == 404
    assert _get("/api/latent/runs/../downloads/cost_and_prediction.csv").status == 400
    outside = tmp_path / "elsewhere.csv"
    outside.write_bytes(b"not allowed")
    expected = first / "reports/cost_and_prediction.csv"
    expected.unlink()
    try:
        expected.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    assert _get(artifact).status == 404


def test_post_allowlist_job_deduplication_and_failure_reload(folders, monkeypatch):
    runs, jobs, _, _ = folders
    _history(runs, "independently-generated", "domestic")
    commands = []
    active = os.getpid()
    assert latent_api._process_token(active)

    def popen(args, **kwargs):
        commands.append(args)
        kwargs["stdout"].write(b"Sample worker error\n")
        return SimpleNamespace(pid=active, wait=lambda: 1)

    monkeypatch.setattr(
        latent_api, "subprocess",
        SimpleNamespace(Popen=popen, DEVNULL=-3, STDOUT=-2, CREATE_NO_WINDOW=0),
    )
    monkeypatch.setattr(latent_api, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    names = iter(("new-run", "new-job", "blocked-run"))
    monkeypatch.setattr(latent_api, "make_run_id", lambda *args: next(names))
    assert _post("/api/latent/runs", {"preset": "baseline_replay", "path": "arbitrary"}).status == 400
    assert _post("/api/latent/runs", {"preset": "../danger"}).status == 400
    result = _post("/api/latent/runs", {"preset": "baseline_replay"})
    assert result.status == 202
    assert result.payload["run_id"] == "new-run"
    assert commands[0][-4:] == ["-m", "my_strategy.latent_quant.runner", "--run-id", "new-run"]
    assert not (runs / "new-run").exists()
    assert (jobs / "new-job.json").exists()
    assert _get(result.payload["detail_url"]).payload["status"] == "running"
    assert _get("/api/latent/overview").payload["presets"][0]["busy"]
    assert _post("/api/latent/runs", {"preset": "domestic_tune"}).status == 409
    saved = latent_api._job_index()[0]
    latent_api._finish(SimpleNamespace(wait=lambda: 1), saved)
    loaded = _get(result.payload["detail_url"]).payload
    assert loaded["status"] == "failed"
    assert "Sample worker error" in loaded["message"]
    assert loaded["report"] is None


def test_paper_requires_same_day_window_and_contained_market_bundle(folders, tmp_path, monkeypatch):
    _, _, inputs, _ = folders
    now = [datetime.fromisoformat("2026-11-23T10:00:00+08:00")]
    monkeypatch.setattr(latent_api.forward, "_now", lambda: now[0])
    url = "/api/latent/paper/prepare"
    assert _post(url, {"bundle_id": "2026-11-23"}).payload["code"] == "outside_capture_window"
    assert _post(url, {"bundle_id": "../../windows"}).status == 400
    now[0] = datetime.fromisoformat("2026-11-23T16:00:00+08:00")
    assert _post(url, {"bundle_id": "2026-11-23"}).payload["code"] == "missing_market_bundle"

    day = inputs / "2026-11-23"
    day.mkdir(parents=True)
    base = {"trade_date": day.name, "inputs": {}}
    for role in latent_api.MARKET_INPUTS:
        path = day / f"{role}.csv"
        path.write_text("header\n", encoding="utf-8")
        base["inputs"][role] = {
            "source": "fixture", "observed_at": "2026-11-23T15:35:00+08:00", "path": path.name,
        }
    bundle = day / "market_bundle.json"
    base["inputs"]["panel"]["path"] = str(tmp_path / "other.csv")
    _write(bundle, base)
    assert _post(url, {"bundle_id": day.name}).payload["code"] == "unknown_artifact"
    base["inputs"]["panel"]["path"] = "panel.csv"
    _write(bundle, base)
    model = tmp_path / "model"
    model.mkdir()
    (model / "baseline_models.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(latent_api, "MODEL_ROOT", model)
    launches = []

    def launch(kind, command, **fields):
        launches.append((kind, command, fields))
        return dict(status="running", phase="starting", message="preparing", created_at="now", updated_at="now")

    monkeypatch.setattr(latent_api, "_launch", launch)
    assert _get("/api/latent/overview").payload["paper"]["prepare_available"]
    assert set(_get("/api/latent/paper/template").payload["inputs"]) == set(latent_api.MARKET_INPUTS)
    assert _post(url, {"bundle_id": day.name}).status == 202
    assert launches[0][1][:4] == ["-m", "my_strategy.latent_quant.forward_prepare", "--market-bundle", str(bundle)]
    assert _get("/api/latent/paper/captures/2026-11-23").status == 404
