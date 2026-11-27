from contextlib import nullcontext
from email.message import Message
import json
from pathlib import Path

import pytest

from my_strategy.model_registry import expert_fallback, expert_release
from my_strategy.model_registry.registry import ModelRegistry
from my_strategy.research_workbench import expert_api, expert_hybrid_study, expert_knn_study, expert_replay


def _candidate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(expert_api, "RUNS", tmp_path)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", tmp_path)
    run_id = "fixture-expert-research"
    report = tmp_path / run_id / "reports/expert_research.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "run": {"run_id": run_id},
        "decision": {"state": "research_only", "historical_numerical_gate": False, "point_in_time_verified": False},
        "final_portfolio": {"arm": "fusion"},
    }), encoding="utf-8")
    (report.parent.parent / "status.json").write_text('{"status":"completed"}', encoding="utf-8")
    return run_id, report


def test_only_approved_release_can_fill_the_main_chart(tmp_path, monkeypatch):
    run_id, report = _candidate(tmp_path, monkeypatch)
    registry = ModelRegistry(tmp_path / "registry.sqlite", snapshot_path=None)
    entry = expert_release.record_candidate(run_id, report, registry)
    assert entry.stage == "candidate"
    assert expert_release.record_candidate(run_id, report, registry).model_id == entry.model_id
    assert expert_release.chart_release(registry)["status"] == "unavailable"
    with pytest.raises((ValueError, FileNotFoundError)):
        expert_release.promote_verified(run_id, registry)
    registry.promote(entry.model_id)
    assert expert_release.chart_release(registry)["status"] == "unavailable"


def test_diagnostic_candidate_replay_is_separate_from_approved_requests(tmp_path, monkeypatch):
    run_id, _ = _candidate(tmp_path, monkeypatch)
    params = {"symbol": "002650.SZ", "start": "2025-09-22", "end": "2025-10-14", "cost": "normal"}
    monkeypatch.setattr(expert_api, "chart_release", lambda: {"status": "unavailable", "reason": "not released"})
    with pytest.raises(expert_api.RouteError) as error:
        expert_api._replay_spec(params | {"release_id": run_id})
    assert error.value.code == "release_unavailable"
    spec = expert_api._replay_spec(params | {"run_id": run_id}, diagnostic=True)
    assert spec["diagnostic"] is True and spec["release"]["status"] == "candidate"
    assert spec["start"] == params["start"] and spec["end"] == params["end"]
    with pytest.raises(expert_api.RouteError) as invalid:
        expert_api._replay_spec(params | {"run_id": run_id, "start": "20250922"}, diagnostic=True)
    assert invalid.value.code == "invalid_scope"


def test_research_settings_normalize_existing_preset_and_custom_scope():
    old = expert_api._study_spec({"preset": "frozen_120"})
    assert old["preset"] == "frozen_v2" and old["commission_bps"] == "3"
    custom = expert_api._study_spec({
        "pool": "manual", "symbols": ["002650.sz", "002436.SZ", "002650.SZ"],
        "start": "2026-04-01", "end": "2026-11-20", "commission_bps": "2.00",
    })
    assert custom["preset"] == "custom" and custom["commission_bps"] == "2"
    assert custom["symbols"] == ["002436.SZ", "002650.SZ"] and custom["end"] == "2026-11-20"
    assert expert_api._study_spec({"seed": 7})["preset"] == "custom"


def test_full_default_uses_latest_cutoff_and_keeps_old_fixed_preset():
    assert expert_api._study_defaults()["pool"] == "full"
    spec = expert_api._study_spec({"pool": "full"})
    assert (spec["pool"], spec["start"], spec["end"], spec["stock_cap"]) == ("full", "2025-03-03", "2026-11-20", None)
    assert expert_api._study_spec({"preset": "frozen_120"})["end"] == "2026-07-28"


def test_candidate_replay_uses_the_selected_single_expert(tmp_path, monkeypatch):
    run_id, report = _candidate(tmp_path, monkeypatch)
    content = json.loads(report.read_text(encoding="utf-8"))
    content["final_portfolio"]["arm"] = "domestic_gpu"
    report.write_text(json.dumps(content), encoding="utf-8")
    selection = expert_api._replay_spec({"run_id": run_id, "symbol": "002650.SZ", "start": "2025-09-22",
                                        "end": "2025-10-14", "cost": "normal"}, diagnostic=True)
    assert selection["arm"] == selection["release"]["arm"] == "domestic_gpu"


def test_current_strategy_requires_registered_full_pool_evidence(tmp_path, monkeypatch):
    run_id, report = _candidate(tmp_path, monkeypatch)
    registry = ModelRegistry(tmp_path / "registry.sqlite", snapshot_path=None)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["input_coverage"] = {"pool": "full", "first_evaluation_session": "2025-03-03",
                              "last_evaluation_session": "2026-11-20"}
    report.write_text(json.dumps(data), encoding="utf-8")
    entry = expert_release.record_candidate(run_id, report, registry)
    manual_id = "manual-expert"
    manual = tmp_path / manual_id / expert_release.REPORT
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps(data | {"run": {"run_id": manual_id}, "input_coverage": {"pool": "manual"}}),
                      encoding="utf-8")
    expert_release.record_candidate(manual_id, manual, registry)
    picked = expert_release.current_strategy(registry)
    assert (picked["status"], picked["run_id"], picked["model_id"]) == ("research_only", run_id, entry.model_id)
    assert (picked["analysis_start"], picked["as_of"]) == ("2025-03-03", "2026-11-20")
    assert picked["publication"]["status"] == "unavailable"
    report.write_text(json.dumps(data | {"changed": True}), encoding="utf-8")
    assert expert_release.current_strategy(registry)["status"] == "unavailable"


def test_unified_replay_scope_pins_current_version_and_normalizes_costs(monkeypatch):
    current = {
        "status": "research_only", "source_run_id": "frozen", "release_id": "frozen", "arm": "fusion",
        "model_id": "registered-id", "research_report_sha256": "hash",
    }
    monkeypatch.setattr(expert_api, "current_strategy", lambda: current)
    dates = {"start": "2026-11-18", "end": "2026-11-20"}
    portfolio = expert_api._current_replay_spec(dates | {"symbols": ["002650.sz", "002436.SZ", "002650.SZ"],
                                                         "initial_cash": 100_000, "commission_bps": "2.00",
                                                         "slippage_bps": 0}, single=False)
    assert portfolio["symbols"] == ["002436.SZ", "002650.SZ"]
    assert portfolio["scope"] == "watchlist" and portfolio["diagnostic"] is True
    assert portfolio["initial_cash"] == 100_000 and portfolio["commission_bps"] == "2" and portfolio["slippage_bps"] == "0"
    assert portfolio["release"]["research_report_sha256"] == "hash" and portfolio["model_id"] == "registered-id"
    single = expert_api._current_replay_spec(dates | {"symbol": "002436.sz"}, single=True)
    assert (single["scope"], single["symbol"], single["source_run_id"]) == ("single_stock", "002436.SZ", "frozen")
    assert expert_api._current_replay_spec(dates | {"symbols": []}, single=False)["symbols"] == []
    with pytest.raises(expert_api.RouteError, match="股票及日期"):
        expert_api._current_replay_spec(dates | {"symbols": [], "run_id": "old"}, single=False)
    with pytest.raises(expert_api.RouteError, match="初始资金"):
        expert_api._current_replay_spec(dates | {"symbols": [], "initial_cash": True}, single=False)


def test_pinned_replay_rejects_context_without_matching_unified_version(tmp_path, monkeypatch):
    run_id, _ = _candidate(tmp_path, monkeypatch)
    request = {"symbols": [], "start": "2026-11-18", "end": "2026-11-20", "source_run_id": run_id}
    spec = expert_api._current_replay_spec(request, single=False)
    assert spec["source_run_id"] == run_id and spec["diagnostic"] is True
    assert spec["model_id"] == run_id and spec["release"]["research_report_sha256"]
    with pytest.raises(expert_api.RouteError) as mismatch:
        expert_api._current_replay_spec(request | {"model_version": "not-the-sealed-version"}, single=False)
    assert mismatch.value.code == "model_version_mismatch"


def test_research_submission_reuses_only_sealed_final_fit_failure(tmp_path, monkeypatch):
    previous_id = "old-research"
    report = tmp_path / previous_id / expert_api.RESEARCH_REPORT
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    previous = {"job_id": previous_id, "run_id": previous_id, "kind": "research", "key": "old-key",
                "spec": {"prediction_key": "prediction-key"}, "status": "failed"}
    monkeypatch.setattr(expert_api, "RUNS", tmp_path)
    monkeypatch.setattr(expert_api, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(expert_api, "_jobs", lambda: [previous])
    monkeypatch.setattr(expert_api, "_status", lambda job: {"status": "failed"} if job["run_id"] == previous_id else {**job, "status": "running"})
    monkeypatch.setattr(expert_api.latent_api, "_submission", nullcontext)
    monkeypatch.setattr(expert_api.latent_api, "_process_token", lambda pid: "token")
    names = iter(("new-research", "ordinary-failure-research"))
    monkeypatch.setattr(expert_api, "make_run_id", lambda *args: next(names))

    class Process:
        pid = 123

    class Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(expert_api.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(expert_api, "Thread", Thread)
    reused = expert_api._submit("research", {"prediction_key": "prediction-key"}, {"request": 1}, "2026-11-20")
    assert reused["reuse_run_id"] == previous_id
    report.unlink()
    fresh = expert_api._submit("research", {"prediction_key": "prediction-key"}, {"request": 2}, "2026-11-20")
    assert fresh["reuse_run_id"] is None


def test_candidate_submission_reuses_completed_parent_identity(monkeypatch):
    parent = {"job_id": "job-parent", "run_id": "parent-backtest", "kind": "replay",
              "spec": {"scope": "watchlist", "symbols": [], "start": "2026-11-18", "end": "2026-11-20",
                       "source_run_id": "research", "model_version": "version", "input_identity": {"old": True}}}
    monkeypatch.setattr(expert_api, "_jobs", lambda: [parent])
    monkeypatch.setattr(expert_api, "_status", lambda job: {**job, "status": "completed"})
    monkeypatch.setattr(expert_api, "_result", lambda run_id, kind: (None, {
        "source": {"scope": "watchlist"}, "accounts": {"full_pool": {}},
    }))
    monkeypatch.setattr(expert_replay, "replay_identity", lambda spec: {"candidate_of": spec["candidate_of"]})
    captured = {}

    def submit(kind, spec, identity, end):
        captured.update(kind=kind, spec=spec, identity=identity, end=end)
        return {"run_id": "candidate-replay", "status": "queued"}

    monkeypatch.setattr(expert_api, "_submit", submit)
    result = expert_api._submit_candidates({"backtest_run_id": "parent-backtest"})
    assert result["run_id"] == "candidate-replay"
    assert captured["kind"] == "replay" and captured["end"] == "2026-11-20"
    assert captured["spec"]["plan_next"] is True
    assert captured["spec"]["candidate_of"] == "parent-backtest"
    assert captured["spec"]["model_version"] == "version"
    assert captured["spec"]["input_identity"] == {"candidate_of": "parent-backtest"}
    assert parent["spec"]["input_identity"] == {"old": True}
    with pytest.raises(expert_api.RouteError) as invalid:
        expert_api._submit_candidates({"backtest_run_id": "parent-backtest", "plan_next": True})
    assert invalid.value.code == "invalid_request"


def test_online_knn_worker_never_registers_multimodal_candidate(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = runs / "knn-job"
    root.mkdir(parents=True)
    monkeypatch.setattr(expert_api, "RUNS", runs)
    monkeypatch.setattr(expert_api, "JOBS", tmp_path / "jobs")
    job = {
        "job_id": "knn-job", "run_id": "knn-job", "kind": "research", "status": "running",
        "spec": {"engine": "online_knn", "research": {"pool": "full"}, "prediction_identity": {"pinned": True}},
    }
    expert_api._save_job(job)
    monkeypatch.setattr(expert_knn_study, "prediction_identity", lambda spec: {"pinned": True})
    called = {}

    def train(*, run_id, spec):
        called["training"] = (run_id, spec)
        return root / expert_api.RESEARCH_REPORT

    monkeypatch.setattr(expert_knn_study, "run", train)
    monkeypatch.setattr(expert_fallback, "activate", lambda folder: {"activated": True, "run_id": folder.name})
    monkeypatch.setattr(
        expert_api, "record_candidate",
        lambda *args: pytest.fail("纯 KNN 不应登记为多模态候选"),
    )
    expert_api._work("knn-job")
    assert called["training"] == ("knn-job", {"pool": "full"})
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_hybrid_worker_trains_and_activates_without_shadowed_candidate(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = runs / "hybrid-job"
    root.mkdir(parents=True)
    monkeypatch.setattr(expert_api, "RUNS", runs)
    monkeypatch.setattr(expert_api, "JOBS", tmp_path / "jobs")
    job = {
        "job_id": "hybrid-job", "run_id": "hybrid-job", "kind": "research", "status": "running",
        "spec": {"engine": "hybrid", "research": {"pool": "full", "knn_device": "cuda"},
                 "prediction_identity": {"pinned": True}},
    }
    expert_api._save_job(job)
    monkeypatch.setattr(expert_hybrid_study, "prediction_identity", lambda spec: {"pinned": True})
    called = {}

    def train(*, run_id, spec):
        called["training"] = (run_id, spec)
        return root / expert_api.RESEARCH_REPORT

    monkeypatch.setattr(expert_hybrid_study, "run", train)
    monkeypatch.setattr(expert_fallback, "activate", lambda folder: called.setdefault("activation", folder.name))
    monkeypatch.setattr(expert_api, "record_candidate", lambda *args: pytest.fail("混合任务不应写入被指针遮蔽的普通候选"))
    expert_api._work("hybrid-job")
    assert called == {"training": ("hybrid-job", {"pool": "full", "knn_device": "cuda"}), "activation": "hybrid-job"}
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_hybrid_submission_freezes_device_and_weight_identity(tmp_path, monkeypatch):
    captured = {}

    class Handler:
        headers = Message()
        headers["Content-Type"] = "application/json"

    body = {"pool": "full", "knn_device": "cuda", "knn_query_chunk": 64}
    raw = json.dumps(body).encode()
    Handler.headers["Content-Length"] = str(len(raw))
    handler = Handler()
    handler.rfile = __import__("io").BytesIO(raw)
    handler._json = lambda status, payload: captured.setdefault("response", (status, payload))
    monkeypatch.setattr(expert_api, "_submit", lambda kind, spec, identity, end: captured.update(
        kind=kind, spec=spec, identity=identity, end=end,
    ) or {"status": "queued"})

    class Preflight:
        def __call__(self, spec):
            assert not {"knn_device", "knn_query_chunk", "top_n"}.intersection(spec)
            return {"status": "ready", "blockers": [], "rejections": {}}

    import my_strategy.research_workbench.expert_config as config

    monkeypatch.setattr(config, "expert_preflight", Preflight())
    monkeypatch.setattr(expert_hybrid_study.research, "prediction_identity", lambda spec: {
        "spec": spec, "research": {"code_sha256": {}},
    })
    assert expert_api.handle_post(handler, "/api/expert/runs")
    assert captured["spec"]["engine"] == "hybrid"
    assert captured["spec"]["research"]["knn_device"] == "cuda"
    assert captured["spec"]["prediction_identity"]["hybrid"]["knn_weight"] == 0.6
    assert captured["spec"]["prediction_identity"]["hybrid"]["multimodal_weight"] == 0.4
    assert captured["spec"]["prediction_identity"]["knn"]["query_chunk"] == 64

@pytest.mark.parametrize("value", ["3%", -2, False, None, float("nan")])
def test_research_rejects_ambiguous_commission_values(value):
    with pytest.raises(expert_api.RouteError) as failure:
        expert_api._study_spec({"commission_bps": value})
    assert failure.value.code == "invalid_commission"
