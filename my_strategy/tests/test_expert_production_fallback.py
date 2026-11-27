import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from my_strategy.model_registry import expert_fallback, expert_release
from my_strategy.model_registry.registry import ModelRegistry
from my_strategy.research_workbench import expert_api, expert_backtest_activity, expert_replay
from my_strategy.tests.test_research_workbench_expert_replay import archive


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def fallback_archive(archive):
    runs, base = archive
    source = runs / "expert"
    source_config = json.loads((source / "config/study.json").read_text(encoding="utf-8"))
    source_report = json.loads((source / "reports/expert_research.json").read_text(encoding="utf-8"))
    source_report["input_coverage"] = {"pool": "full"}
    _json(source / "reports/expert_research.json", source_report)
    root = runs / expert_fallback.RUN_ID
    rank_policy = source_config["portfolio"] | {"selection_mode": "cross_sectional_rank", "min_score": None}

    def execution(value):
        return {cost: {"total_return": value, "verification": {"cash": True, "equity": True}}
                for cost in ("normal", "double")}

    folds = []
    for index, fold in enumerate(source_report["folds"]):
        period = fold["period"]
        destination = root / "backtests" / period
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / f"backtests/{period}/fusion_predictions.parquet",
                        destination / f"{expert_fallback.ARM}_predictions.parquet")
        folds.append({
            "period": period, "evaluation_start": fold["evaluation_start"], "evaluation_end": fold["evaluation_end"],
            "train": fold["models"]["train"], "calibration": fold["models"]["calibration"],
            "arms": {
                expert_fallback.ARM: {"strategy": rank_policy, "execution": execution([0.10, 0.20][index])},
                "fusion_rank": {"strategy": rank_policy, "execution": execution(0.05)},
            },
        })
    for period, start, end, fit_end in (
        ("oos_3", "2025-03-10", "2025-03-11", "2025-03-07"),
        ("historical_replay", "2025-03-12", "2025-03-13", "2025-03-11"),
    ):
        destination = root / "backtests" / period
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["date", "stock", "q10", "q50", "q90", "score"]).to_parquet(
            destination / f"{expert_fallback.ARM}_predictions.parquet", index=False,
        )
        folds.append({
            "period": period, "evaluation_start": start, "evaluation_end": end,
            "train": {"last_label_end": fit_end}, "calibration": {"last_label_end": fit_end},
            "arms": {expert_fallback.ARM: {"strategy": rank_policy,
                                           "execution": execution(-0.05 if period == "oos_3" else -0.02)}},
        })
    _json(root / "status.json", {"status": "completed"})
    _json(root / "config/study.json", {
        "source_run_id": source.name, "development_periods": list(expert_fallback.DEVELOPMENT),
        "rank_policy": rank_policy, "code_sha256": {"signal_experts.py": "archived"},
    })
    _json(root / expert_fallback.REPORT, {
        "run": {"run_id": root.name}, "folds": folds,
        "decision": {"status": "research_only", "registry_written": False},
        "artifact_verification": {"ledger_tables": 8, "matrix_rows": 8,
                                  "all_daily_dates_unique": True, "all_trade_counts_match": True},
    })
    expert_fallback.write_manifest(root)
    return runs, base


def test_current_strategy_uses_verified_fallback_without_registry_promotion(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    monkeypatch.setattr(expert_fallback, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", runs)
    registry = ModelRegistry(runs / "registry.sqlite", snapshot_path=None)
    current = expert_release.current_strategy(registry)
    assert current["status"] == "production_fallback"
    assert current["arm"] == expert_fallback.ARM
    assert current["selection_evidence"]["mean_return"] == pytest.approx((0.10 + 0.20 - 0.05) / 3)
    assert current["portfolio"]["selection_mode"] == "cross_sectional_rank"
    assert current["portfolio"]["min_score"] is None
    assert registry.entries() == []
    assert expert_release.chart_release(registry)["status"] == "unavailable"

    spec = expert_api._current_replay_spec({"symbols": [], "start": "2025-03-03", "end": "2025-03-07"}, single=False)
    assert spec["diagnostic"] is False
    assert spec["release"]["status"] == "production_fallback"
    assert spec["arm"] == expert_fallback.ARM


def test_fallback_executes_ranked_predictions_and_activity_uses_rank_semantics(fallback_archive):
    runs, base = fallback_archive
    release = expert_fallback.load(runs / expert_fallback.RUN_ID)
    spec = {
        "scope": "watchlist", "symbols": [], "source_run_id": release["source_run_id"],
        "release_id": release["release_id"], "arm": release["arm"], "release": release,
        "model_id": release["model_id"], "model_version": None, "diagnostic": False,
        "start": base["start"], "end": base["end"], "cost": "normal",
    }
    identity = expert_replay.replay_identity(spec)
    assert identity["approval"] == "production_fallback"
    assert identity["selection_policy"]["selection_mode"] == "cross_sectional_rank"
    output = expert_replay.run(run_id="fallback-view", spec=spec | {"input_identity": identity})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"]["arm"] == expert_fallback.ARM
    assert report["accounts"]["full_pool"]["metrics"]["trade_count"] > 0
    assert all(report["accounts"]["full_pool"]["verification"].values())

    activity = expert_backtest_activity.activity("fallback-view", "")
    assert activity["evidence"]["basis"] == "pinned_archive"
    assert activity["evidence"]["selection_mode"] == "cross_sectional_rank"
    assert activity["evidence"]["score_threshold"] is None
    assert activity["research_status"]["status"] == "production_fallback"
    assert activity["research_status"]["release_gate"]["state"] == "production_fallback_not_promotion"
    full = activity["inactivity_diagnosis"]["full_run"]
    assert full["qualified_on_rebalance"] == full["eligible_on_rebalance"]


def test_fallback_single_stock_does_not_compare_scores_with_null_threshold(fallback_archive):
    runs, base = fallback_archive
    release = expert_fallback.load(runs / expert_fallback.RUN_ID)
    spec = {
        "scope": "single_stock", "symbol": base["symbol"], "source_run_id": release["source_run_id"],
        "release_id": release["release_id"], "arm": release["arm"], "release": release,
        "model_id": release["model_id"], "model_version": None, "diagnostic": False,
        "start": base["start"], "end": base["start"], "cost": "normal",
    }
    output = expert_replay.run(run_id="fallback-single", spec=spec)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["latest_action"]["state"] == "buy"
    assert report["latest_action"]["pending_order"]["requested_shares"] > 0


def test_invalid_fallback_never_silently_returns_to_legacy_candidate(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    monkeypatch.setattr(expert_fallback, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", runs)
    prediction = runs / expert_fallback.RUN_ID / f"backtests/oos_1/{expert_fallback.ARM}_predictions.parquet"
    with prediction.open("ab") as content:
        content.write(b"tampered")
    current = expert_release.current_strategy(ModelRegistry(runs / "registry.sqlite", snapshot_path=None))
    assert current["status"] == "unavailable"
    assert "生产回退证据无效" in current["reason"]


def _candidate_report(fallback_archive, monkeypatch, run_id, returns, *, different_scope=False):
    runs, _ = fallback_archive
    monkeypatch.setattr(expert_fallback, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", runs)
    source = runs / expert_fallback.RUN_ID
    multiview = json.loads((source / expert_fallback.REPORT).read_text(encoding="utf-8"))
    config = json.loads((runs / "expert/config/study.json").read_text(encoding="utf-8"))
    folds = []
    for period, value in zip(expert_fallback.DEVELOPMENT, returns, strict=True):
        original = next(fold for fold in multiview["folds"] if fold["period"] == period)
        end = "2025-03-12" if different_scope and period == "oos_3" else original["evaluation_end"]
        folds.append({
            "period": period, "evaluation_start": original["evaluation_start"], "evaluation_end": end,
            "arms": {"fusion": {"execution": {"double": {
                "total_return": value,
                "verification": {"cash_verified": True, "equity_verified": True, "next_session_verified": True},
            }}}},
        })
    root = runs / run_id
    _json(root / "status.json", {"status": "completed"})
    _json(root / "config/study.json", config)
    report = root / expert_release.REPORT
    _json(report, {
        "run": {"run_id": run_id}, "decision": {"state": "research_only"},
        "input_coverage": {"pool": "full", "first_evaluation_session": folds[0]["evaluation_start"][:10],
                           "last_evaluation_session": folds[-1]["evaluation_end"][:10]},
        "final_portfolio": {"arm": "fusion"}, "folds": folds,
    })
    return report


def test_higher_comparable_test_return_automatically_replaces_production_fallback(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    registry = ModelRegistry(runs / "registry.sqlite", snapshot_path=None)
    first = _candidate_report(fallback_archive, monkeypatch, "candidate-high", [0.11, 0.20, -0.05])
    entry = expert_release.record_candidate("candidate-high", first, registry)
    assert entry.params.get("production_fallback") is True, entry.params
    current = expert_release.current_strategy(registry)
    assert current["production_source"] == "research_candidate"
    assert current["run_id"] == "candidate-high"
    assert current["selection_evidence"]["total_return"] == pytest.approx(1.11 * 1.20 * 0.95 - 1)
    assert current["selection_evidence"]["previous_run_id"] == expert_fallback.RUN_ID
    assert expert_release.chart_release(registry)["status"] == "unavailable"

    newer = _candidate_report(fallback_archive, monkeypatch, "candidate-newer", [0.12, 0.20, -0.05])
    expert_release.record_candidate("candidate-newer", newer, registry)
    current = expert_release.current_strategy(registry)
    assert current["run_id"] == "candidate-newer"
    assert current["selection_evidence"]["previous_run_id"] == "candidate-high"
    assert sum(entry.params.get("production_fallback") is True for entry in registry.entries()) == 1

    lower = _candidate_report(fallback_archive, monkeypatch, "candidate-lower", [0.10, 0.20, -0.05])
    low_entry = expert_release.record_candidate("candidate-lower", lower, registry)
    assert low_entry.params["production_comparison"]["state"] == "not_replaced"
    assert expert_release.current_strategy(registry)["run_id"] == "candidate-newer"

    other = _candidate_report(fallback_archive, monkeypatch, "candidate-other-scope", [0.20, 0.20, 0.20],
                              different_scope=True)
    other_entry = expert_release.record_candidate("candidate-other-scope", other, registry)
    assert "口径不同" in other_entry.params["production_comparison"]["reason"]
    assert expert_release.current_strategy(registry)["run_id"] == "candidate-newer"

    newer.write_text(newer.read_text(encoding="utf-8") + " ", encoding="utf-8")
    invalid = expert_release.current_strategy(registry)
    assert invalid["status"] == "unavailable" and "自动生产回退证据无效" in invalid["reason"]


def test_direct_knn_run_seals_standard_report_and_replay_config(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    monkeypatch.setattr(expert_fallback, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", runs)
    legacy = runs / expert_fallback.RUN_ID
    root = runs / "direct-knn-run"
    shutil.copytree(legacy, root)
    (root / expert_fallback.MANIFEST).unlink()
    old_report = json.loads((root / expert_fallback.REPORT).read_text(encoding="utf-8"))
    folds = old_report["folds"]
    report = {
        "run": {"run_id": root.name}, "training_engine": expert_fallback.DIRECT_ENGINE,
        "input_coverage": {"pool": "full"}, "folds": folds[:3], "historical_review": folds[3],
        "decision": old_report["decision"], "artifact_verification": old_report["artifact_verification"],
    }
    source_config = json.loads((runs / "expert/config/study.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config/study.json").read_text(encoding="utf-8")) | {
        "training_engine": expert_fallback.DIRECT_ENGINE, "model_spec": {"pool": "full"},
        "portfolio": source_config["portfolio"] | {"selection_mode": "cross_sectional_rank", "min_score": None},
        "initial_cash": source_config["initial_cash"], "expense_model": source_config["expense_model"],
        "cost_multipliers": {"normal": 1, "double": 2},
    }
    _json(root / "config/study.json", config)
    _json(root / expert_release.REPORT, report)
    expert_fallback.write_manifest(root)
    proof = expert_fallback.load(root)
    assert proof["direct_archive"] is True
    assert proof["research_report"] == expert_release.REPORT
    replay_config, _ = expert_fallback.replay_config(root)
    assert replay_config["model_spec"] == {"pool": "full"}
    assert "data_source_run_id" not in replay_config
    assert expert_fallback.activate(root, force=True)["activated"] is True


def test_explicit_knn_pointer_overrides_automatic_fusion(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    registry = ModelRegistry(runs / "registry.sqlite", snapshot_path=None)
    candidate = _candidate_report(fallback_archive, monkeypatch, "candidate-before-knn", [0.50, 0.50, 0.50])
    expert_release.record_candidate("candidate-before-knn", candidate, registry)
    assert expert_release.current_strategy(registry)["arm"] == "fusion"

    activated = expert_fallback.activate(runs / expert_fallback.RUN_ID, force=True)
    assert activated["activated"] is True
    current = expert_release.current_strategy(registry)
    assert current["run_id"] == expert_fallback.RUN_ID
    assert current["arm"] == expert_fallback.ARM
    assert current["production_source"] == "online_knn"
    assert current["selection_evidence"]["compound_return"] == pytest.approx(1.10 * 1.20 * 0.95 - 1)


def _hybrid_archive(fallback_archive, monkeypatch, run_id="hybrid-run", returns=(0.10, 0.20, -0.05), *, different_scope=False):
    runs, base = fallback_archive
    monkeypatch.setattr(expert_fallback, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", runs)
    legacy = runs / expert_fallback.RUN_ID
    root = runs / run_id
    shutil.copytree(legacy, root)
    (root / expert_fallback.MANIFEST).unlink()
    old_report = json.loads((root / expert_fallback.REPORT).read_text(encoding="utf-8"))
    folds = json.loads(json.dumps(old_report["folds"]))
    for index, fold in enumerate(folds):
        knn = fold["arms"][expert_fallback.ARM]
        hybrid = json.loads(json.dumps(knn))
        for cost in ("normal", "double"):
            hybrid["execution"][cost]["total_return"] = returns[index] if index < 3 else -0.02
        fusion = json.loads(json.dumps(hybrid))
        for cost in ("normal", "double"):
            fusion["execution"][cost]["total_return"] = 0.0
        fold["arms"] = {expert_fallback.HYBRID_ARM: hybrid, expert_fallback.ARM: knn, "fusion": fusion}
        if different_scope and fold["period"] == "oos_3":
            fold["evaluation_end"] = "2025-03-12"
        destination = root / "backtests" / fold["period"] / f"{expert_fallback.HYBRID_ARM}_predictions.parquet"
        shutil.copyfile(root / "backtests" / fold["period"] / f"{expert_fallback.ARM}_predictions.parquet", destination)
    _json(root / expert_release.REPORT, {
        "run": {"run_id": run_id}, "training_engine": expert_fallback.HYBRID_ENGINE,
        "input_coverage": {"pool": "full", "first_evaluation_session": folds[0]["evaluation_start"][:10],
                           "last_evaluation_session": folds[2]["evaluation_end"][:10]},
        "folds": folds[:3], "historical_review": folds[3], "decision": old_report["decision"],
        "artifact_verification": old_report["artifact_verification"],
    })
    source_config = json.loads((runs / "expert/config/study.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config/study.json").read_text(encoding="utf-8")) | {
        "training_engine": expert_fallback.HYBRID_ENGINE, "model_spec": {"pool": "full"},
        "portfolio": source_config["portfolio"] | {"selection_mode": "cross_sectional_rank", "min_score": None},
        "initial_cash": source_config["initial_cash"], "expense_model": source_config["expense_model"],
        "cost_multipliers": {"normal": 1, "double": 2},
    }
    _json(root / "config/study.json", config)
    expert_fallback.write_manifest(root)
    return root


def test_hybrid_seal_has_independent_pointer_and_fails_closed(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    root = _hybrid_archive(fallback_archive, monkeypatch)
    proof = expert_fallback.load(root)
    assert proof["production_source"] == "hybrid" and proof["arm"] == expert_fallback.HYBRID_ARM
    assert proof["selection_evidence"]["compound_return"] == pytest.approx(1.10 * 1.20 * 0.95 - 1)
    activated = expert_fallback.activate(root, force=True)
    assert activated["activated"] is True
    pointer = json.loads((runs / expert_fallback.HYBRID_POINTER).read_text(encoding="utf-8"))
    assert pointer["run_id"] == root.name and not (runs / "online_knn_production.json").exists()
    registry = ModelRegistry(runs / "registry.sqlite", snapshot_path=None)
    current = expert_release.current_strategy(registry)
    assert current["run_id"] == root.name and current["production_source"] == "hybrid"
    with (root / f"backtests/oos_1/{expert_fallback.HYBRID_ARM}_predictions.parquet").open("ab") as content:
        content.write(b"tampered")
    current = expert_release.current_strategy(registry)
    assert current["status"] == "unavailable" and "混合生产证据无效" in current["reason"]


def test_hybrid_replaces_only_on_strictly_higher_comparable_return(fallback_archive, monkeypatch):
    runs, _ = fallback_archive
    equal = _hybrid_archive(fallback_archive, monkeypatch, "hybrid-equal")
    rejected = expert_fallback.activate(equal)
    assert rejected["activated"] is False and rejected["comparison"]["state"] == "not_replaced"
    assert not (runs / expert_fallback.HYBRID_POINTER).exists()

    high = _hybrid_archive(fallback_archive, monkeypatch, "hybrid-high", (0.11, 0.20, -0.05))
    activated = expert_fallback.activate(high)
    assert activated["activated"] is True and activated["comparison"]["mode"] == "higher_comparable_hybrid_return"
    registry = ModelRegistry(runs / "registry.sqlite", snapshot_path=None)
    assert expert_release.current_strategy(registry)["run_id"] == high.name

    lower = _hybrid_archive(fallback_archive, monkeypatch, "hybrid-lower", (0.10, 0.20, -0.05))
    assert expert_fallback.activate(lower)["activated"] is False
    assert expert_release.current_strategy(registry)["run_id"] == high.name

    other = _hybrid_archive(fallback_archive, monkeypatch, "hybrid-other", (0.50, 0.50, 0.50), different_scope=True)
    comparison = expert_fallback.activate(other)["comparison"]
    assert comparison["state"] == "not_replaced" and "口径不同" in comparison["reason"]
    assert expert_release.current_strategy(registry)["run_id"] == high.name


def test_standard_research_fallback_uses_common_replay_without_knn_manifest(archive):
    runs, base = archive
    release = {
        "status": "production_fallback", "production_source": "research_candidate",
        "release_id": "automatic-candidate", "source_run_id": "expert", "arm": "fusion",
        "manifest_sha256": None, "fallback_manifest_sha256": None,
    }
    spec = base | {"release": release, "release_id": release["release_id"], "diagnostic": False}
    identity = expert_replay.replay_identity(spec)
    assert identity["approval"] == "production_fallback"
    output = expert_replay.run(run_id="automatic-candidate-replay", spec=spec | {"input_identity": identity})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"]["release_id"] == "automatic-candidate"
    assert report["ledger_metrics"]["trade_count"] > 0
    assert all(report["verification"].values())
