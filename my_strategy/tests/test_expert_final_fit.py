"""Deterministic final-fit closure tests with mocked native GPU fitters."""

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant import forward
from my_strategy.latent_quant.domestic_runner import FrozenInputs
from my_strategy.research_workbench import expert_final_fit as final_fit
from my_strategy.research_workbench import expert_forward, expert_gpu, expert_research
from my_strategy.research_workbench.runner import Window, write_json


class _Native:
    def __init__(self, value=0.0, features=()):
        self.value = value
        self.feature_names_ = list(features)
        self.tree_count_ = 12

    def predict(self, matrix, **kwargs):
        return matrix["x"].to_numpy(dtype=float) * 0.0001 + self.value

    def save_model(self, path, **kwargs):
        Path(path).write_text(json.dumps({"value": self.value, "features": self.feature_names_}), encoding="utf-8")

    def load_model(self, path, **kwargs):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        self.value = saved["value"]
        self.feature_names_ = saved["features"]


def _fixture(tmp_path, monkeypatch, *, point_in_time=False):
    dates = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame({
        "date": dates.repeat(2), "stock": ["A", "B"] * len(dates), "signal_eligible": True,
        "label_end": np.repeat(pd.Series(dates).shift(-2).to_numpy(), 2),
        "x": np.arange(len(dates) * 2, dtype=float) / 10,
        "st_direction": 1.0, "atr_pct": 0.02, "st_distance": 0.03,
    })
    frame["target"] = np.sin(np.arange(len(frame))) * 0.02 + 0.01
    frame.loc[frame["label_end"].isna(), "target"] = np.nan
    root = tmp_path / "run-new"
    report = {
        "run": {"run_id": root.name}, "final_portfolio": {"arm": "fusion"},
        "decision": {"state": "research_only", "historical_numerical_gate": True,
                     "point_in_time_verified": point_in_time},
        "input_coverage": {"point_in_time": {"verified": point_in_time}},
        "experts": {"domestic_gpu": {"status": "active"}, "historical_quantiles": {"status": "active"},
                    "supertrend_atr": {"status": "active"}, "overseas_gpu": {"status": "abstain"},
                    "SeqLSTM": {"status": "abstain"}},
        "reused_predictions_from": None,
    }
    config = {
        "end": str(dates[-1].date()), "seed": 42, "stock_cap": 2,
        "model_spec": {"pool": "frozen_120"}, "features": {"domestic_gpu": ["x"]},
        "sources": {"panel": {"sha256": "a" * 64, "size_bytes": 10}},
        "final_fit_code_sha256": {"expert_final_fit.py": "b" * 64},
    }
    write_json(root / "reports/expert_research.json", report)
    write_json(root / "config/study.json", config)
    calls = []

    def tune(train, valid, features, seed, audit):
        calls.append((train.copy(), valid.copy()))
        fitted = [_Native(value, features) for value in (-0.01, 0.01, 0.03)]
        return fitted, {"features": list(features), "winner": {
            "tree_counts": [12, 12, 12], "validation_q50": {"unique": 30},
        }}

    def inner(train, features, seed, devices, scheduler, *, matched_fit=False):
        assert train["label_end"].max() < valid_start
        return {"domestic_gpu": 1.0}, {"winner": {"weights": {"domestic_gpu": 1.0}}}, pd.DataFrame()

    valid_start = dates[-16]
    monkeypatch.setattr(expert_research, "_tune", tune)
    monkeypatch.setattr(expert_research, "_inner_fusion", inner)
    monkeypatch.setattr(forward, "_now", lambda: datetime(2025, 1, 2, 12, tzinfo=forward.SHANGHAI))
    import catboost

    monkeypatch.setattr(catboost, "CatBoostRegressor", _Native)
    return root, frame, report, config, calls


def test_split_maturity_future_rejection_and_ineligible_label_invariance(tmp_path, monkeypatch):
    _, frame, _, config, _ = _fixture(tmp_path, monkeypatch)
    sizes = {"train_sessions": 24, "validation_sessions": 8, "calibration_sessions": 8}
    train, valid, calibration, _, window = final_fit._split(frame, pd.DatetimeIndex(sorted(frame.date.unique())), config["end"], **sizes)
    assert train.label_end.max() < window.valid_start
    assert valid.label_end.max() < window.calibration_start
    assert calibration.label_end.max() <= pd.Timestamp(config["end"])
    assert train.date.max() < valid.date.min() <= valid.date.max() < calibration.date.min()

    ineligible_noise = frame.copy()
    ineligible_noise["signal_eligible"] = False
    ineligible_noise["target"] = np.where(ineligible_noise["label_end"].notna(), 999.0, np.nan)
    changed = pd.concat([frame, ineligible_noise], ignore_index=True)
    changed_train, changed_valid, changed_cal, _, _ = final_fit._split(
        changed, pd.DatetimeIndex(sorted(frame.date.unique())), config["end"], **sizes,
    )
    pd.testing.assert_frame_equal(train, changed_train)
    pd.testing.assert_frame_equal(valid, changed_valid)
    pd.testing.assert_frame_equal(calibration, changed_cal)

    future = frame.copy()
    future.loc[future.index[-1], "target"] = 0.5
    future.loc[future.index[-1], "label_end"] = pd.Timestamp(config["end"]) + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="未来标签"):
        final_fit._split(future, pd.DatetimeIndex(sorted(frame.date.unique())), config["end"], **sizes)
    future_input = pd.concat([frame, frame.tail(1).assign(date=pd.Timestamp(config["end"]) + pd.Timedelta(days=1))])
    with pytest.raises(ValueError, match="未来标签"):
        final_fit._split(future_input, pd.DatetimeIndex(sorted(frame.date.unique())), config["end"], **sizes)


def test_bundle_hash_closure_exact_replay_and_no_report_cycle(tmp_path, monkeypatch):
    root, frame, report, config, calls = _fixture(tmp_path, monkeypatch)
    report_path = root / "reports/expert_research.json"
    before = report_path.read_bytes()
    unified_path = final_fit.fit_final_fit(
        root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8,
    )
    assert report_path.read_bytes() == before
    assert len(calls) == 1
    train_used, valid_used = calls[0]
    assert train_used.label_end.max() < valid_used.date.min()
    assert valid_used.date.max() < pd.Timestamp(config["end"])
    unified = final_fit.verify_final_fit(root)
    assert unified_path.name == "unified_expert.json"
    assert unified["release_status"] == {"state": "unregistered", "auto_promoted": False}
    assert unified["research_status"]["point_in_time_verified"] is False
    assert unified["universe"]["pool"] == "frozen_120"
    model = json.loads((root / final_fit.MODEL).read_text(encoding="utf-8"))
    assert model["point_in_time_verified"] is False
    assert "model_version" not in model
    summary = json.loads((root / final_fit.FINAL_FIT_REPORT).read_text(encoding="utf-8"))
    assert summary["exact_replay"]["passed"] is True
    assert summary["split"]["refit_on_validation_or_calibration"] is False
    saved_train = pd.read_parquet(root / "models/forward/train.parquet")
    pd.testing.assert_frame_equal(saved_train, train_used.reset_index(drop=True))
    replay = frame.tail(10).drop(columns=["target", "label_end"])
    reproduced = expert_forward._forecast_assets(
        model, root / "models/forward", root / "models/forward/train.parquet",
        expert_forward._arms(model["arm"]), replay, forward._now(), verify_macro_availability=False,
    )
    assert np.isfinite(reproduced[["q10", "q50", "q90", "score"]]).all(axis=None)
    assert (reproduced.q10 <= reproduced.q50).all() and (reproduced.q50 <= reproduced.q90).all()

    native = root / "models/forward/domestic_gpu/q50.cbm"
    native.write_bytes(native.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="原生文件哈希"):
        final_fit.verify_final_fit(root)


def test_manifest_detects_research_report_tamper(tmp_path, monkeypatch):
    root, frame, _, _, _ = _fixture(tmp_path, monkeypatch)
    final_fit.fit_final_fit(root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8)
    report = root / "reports/expert_research.json"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="不可变研究"):
        final_fit.verify_final_fit(root)


def test_weighted_single_tree_collapse_is_rejected_without_partial_bundle(tmp_path, monkeypatch):
    root, frame, _, _, _ = _fixture(tmp_path, monkeypatch)

    def collapsed(train, valid, features, seed, audit):
        return [_Native(value, features) for value in (-0.01, 0.01, 0.03)], {"winner": {
            "tree_counts": [12, 1, 12], "validation_q50": [0.01] * len(valid),
        }}

    monkeypatch.setattr(expert_research, "_tune", collapsed)
    with pytest.raises(ValueError, match="分数塌缩"):
        final_fit.fit_final_fit(root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8)
    assert not (root / "models/forward").exists()
    assert not (root / final_fit.UNIFIED_REPORT).exists()


def test_collapsed_overseas_abstains_and_reweights_only_on_train_oof(tmp_path, monkeypatch):
    root, frame, report, config, _ = _fixture(tmp_path, monkeypatch)
    for name in expert_research.MACRO_FEATURES:
        frame[name] = 1.0
    report["experts"]["overseas_gpu"] = {"status": "active"}
    config["features"]["overseas_gpu"] = ["x", "fed_target_lower"]
    write_json(root / "reports/expert_research.json", report)
    write_json(root / "config/study.json", config)
    inner_calls = []

    def inner(train, features, seed, devices, scheduler, *, matched_fit=False):
        inner_calls.append(set(features))
        weights = {"domestic_gpu": 0.5, "overseas_gpu": 0.5} if "overseas_gpu" in features else {"domestic_gpu": 1.0}
        return weights, {"winner": {"weights": weights}}, pd.DataFrame()

    def tune(train, valid, features, seed, audit):
        collapsed = "fed_target_lower" in features
        fitted = [_Native(value, features) for value in (-0.01, 0.01, 0.03)]
        return fitted, {"features": list(features), "winner": {
            "tree_counts": [12, 1 if collapsed else 12, 12],
            "validation_q50": [0.01] * len(valid) if collapsed else {"unique": 30},
        }}

    monkeypatch.setattr(expert_research, "_inner_fusion", inner)
    monkeypatch.setattr(expert_research, "_tune", tune)
    final_fit.fit_final_fit(root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8)
    assert inner_calls == [{"domestic_gpu", "overseas_gpu"}, {"domestic_gpu"}]
    model = json.loads((root / final_fit.MODEL).read_text(encoding="utf-8"))
    assert model["weights"] == {"domestic_gpu": 1.0}
    assert set(model["features"]) == {"domestic_gpu"}
    assert not (root / "models/forward/overseas_gpu").exists()
    summary = json.loads((root / final_fit.FINAL_FIT_REPORT).read_text(encoding="utf-8"))
    assert summary["fusion"]["method"] == "validation_health_abstention_and_train_oof_reweight"
    assert summary["fusion"]["weight_scope"] == "same_train_inner_oof"
    assert summary["fusion"]["abstained"]["overseas_gpu"]["collapsed"] is True
    assert summary["native_arms"]["overseas_gpu"]["median_health"]["collapsed"] is True
    assert final_fit.verify_final_fit(root)["model_version"] == summary["model_version"]


def test_false_point_in_time_bundle_cannot_pass_forward_asset_lock(tmp_path, monkeypatch):
    root, frame, report, config, _ = _fixture(tmp_path, monkeypatch, point_in_time=False)
    final_fit.fit_final_fit(root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8)
    with pytest.raises(ValueError, match="通过核验"):
        expert_forward._model_assets(root, report, config)


def test_verified_point_in_time_bundle_uses_existing_forward_asset_contract(tmp_path, monkeypatch):
    root, frame, report, config, _ = _fixture(tmp_path, monkeypatch, point_in_time=True)
    final_fit.fit_final_fit(root, frame, train_sessions=24, validation_sessions=8, calibration_sessions=8)
    model, hashes = expert_forward._model_assets(root, report, config)
    assert model["point_in_time_verified"] is True
    assert hashes[final_fit.MODEL] == forward._sha(root / final_fit.MODEL)


def test_final_fit_code_identity_is_separate_from_historical_prediction_code(monkeypatch):
    monkeypatch.setattr(expert_research, "_sources", lambda: {})
    monkeypatch.setattr(expert_research, "_sha", lambda path: path.name)
    historical = expert_research.research_inputs()["code_sha256"]
    producer = expert_research.research_inputs(include_final_fit=True)["code_sha256"]
    assert "expert_final_fit.py" not in historical and "expert_forward.py" not in historical
    assert {"expert_final_fit.py", "expert_forward.py"} <= producer.keys()
    assert all(producer[key] == value for key, value in historical.items())


def test_final_fit_failure_can_supply_only_verified_historical_predictions(tmp_path, monkeypatch):
    source = tmp_path / "run-old"
    files = {"backtests/oos_1/fusion_predictions.parquet": {"sha256": "a" * 64}}
    report = {"prediction_key": "key", "prediction_evidence_sha256": expert_research.stable_hash(files), "folds": []}
    write_json(source / "reports/expert_research.json", report)
    write_json(source / "config/study.json", {"prediction_key": "key"})
    write_json(source / "status.json", {"status": "failed", "phase": "failed"})
    write_json(source / expert_research.PREDICTION_EVIDENCE, {"prediction_key": "key", "files": files})
    monkeypatch.setattr(expert_research, "artifact_run_dir", lambda run_id, create=False: source)
    monkeypatch.setattr(expert_research, "_prediction_evidence", lambda *args: files)
    reused, reused_report, _, _ = expert_research._reuse_source("run-old", "key")
    assert reused == source and reused_report == report
    write_json(source / "status.json", {"status": "running", "phase": "oos_2"})
    with pytest.raises(ValueError, match="完整封存预测"):
        expert_research._reuse_source("run-old", "key")
    write_json(source / "status.json", {"status": "failed", "phase": "failed"})
    monkeypatch.setattr(expert_research, "_prediction_evidence", lambda *args: {})
    with pytest.raises(ValueError, match="发生变化"):
        expert_research._reuse_source("run-old", "key")


def test_reused_predictions_still_close_one_new_final_fit(tmp_path, monkeypatch):
    previous = tmp_path / "run-old"
    (previous / "models/forward").mkdir(parents=True)
    (previous / "models/forward/old.cbm").write_bytes(b"old archive")
    dates = pd.bdate_range("2024-01-02", periods=12)
    frame = pd.DataFrame({"date": dates, "stock": "A", "target": 0.01})
    coverage = {"point_in_time": {"verified": False}, "overseas": {"verified_sessions": 0}, "source_identity": {}}
    data = FrozenInputs(
        frame, dates, pd.Series(0.0, index=dates), pd.DataFrame(), pd.DataFrame(), coverage,
    )
    reference = {"sha256": "a" * 64, "size_bytes": 1}
    references = {"frozen_inputs/panel/panel.parquet": reference}
    prior_config = {
        "features": {"domestic_gpu": ["x"]}, "catboost_version": "fixture", "sources": {"panel": reference},
    }
    prior_report = {"validation": {}, "input_coverage": coverage,
                    "folds": [{"period": f"oos_{number}"} for number in range(3)]}
    prior_evidence = {"prediction_key": "key", "files": {}}
    monkeypatch.setattr(expert_research, "artifact_run_dir", lambda run_id, create=False: tmp_path / run_id)
    monkeypatch.setattr(expert_research, "prediction_identity", lambda spec: {
        "spec": {"pool": "frozen_120", "preset": "custom", "seed": 42}, "research": {"code_sha256": {}},
    })
    monkeypatch.setattr(expert_research, "research_inputs", lambda **kwargs: {"code_sha256": {"final.py": "hash"}})
    monkeypatch.setattr(expert_research, "_reuse_source", lambda run_id, key: (
        previous, prior_report, prior_config, prior_evidence,
    ))

    class Store:
        def read_manifest(self, run_id):
            return {"files": references}

        def write_manifest(self, run_id, references, metadata):
            return tmp_path / "artifact-manifest.json"

    monkeypatch.setattr(expert_research, "ExpertArtifactStore", Store)
    monkeypatch.setattr(expert_research, "_data_by_reference", lambda *args: data)
    windows = [Window(f"oos_{number}", dates[0], dates[3], dates[6], dates[9 + number:10 + number]) for number in range(3)]
    monkeypatch.setattr(expert_research, "plan_expert_windows", lambda *args: windows)

    def copy_predictions(source, root, evidence):
        assert source == previous
        assert not any(path.startswith("models/forward") for path in evidence["files"])

    monkeypatch.setattr(expert_research, "_copy_predictions", copy_predictions)
    monkeypatch.setattr(expert_research, "_prediction_evidence", lambda *args: {})

    class Context:
        run_id = "run-new"

        def subdir(self, *parts):
            path = tmp_path.joinpath(self.run_id, *parts)
            path.mkdir(parents=True, exist_ok=True)
            return path

        def to_dict(self):
            return {"run_id": self.run_id}

    monkeypatch.setattr(expert_research, "create_run_context", lambda **kwargs: Context())

    def reprice(data, window, context, archived, *, policy, cost_model, seed):
        fold = {"period": window.period, "models": {}, "fusion": {}, "arms": {}}
        return fold, {cost: {} for cost in expert_research.COSTS}

    monkeypatch.setattr(expert_research, "_reprice_fold", reprice)
    monkeypatch.setattr(expert_research, "_paired", lambda *args: {})
    monkeypatch.setattr(expert_research, "_paired_ablations", lambda *args: {})
    monkeypatch.setattr(expert_research, "_decision", lambda *args: {
        "state": "research_only", "historical_numerical_gate": False, "point_in_time_verified": False, "reasons": [],
    })
    monkeypatch.setattr(expert_research, "_final_portfolio", lambda *args: {"arm": "domestic_gpu"})
    monkeypatch.setattr(expert_research, "_summary_table", lambda folds: pd.DataFrame({"fold": [fold["period"] for fold in folds]}))
    calls = []

    def close(root, supplied):
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "running" and status["phase"] == "final_fit"
        assert (root / "reports/expert_research.json").is_file()
        calls.append(root)
        (root / "models/forward").mkdir(parents=True)
        (root / "models/forward/model.json").write_text("new fit", encoding="utf-8")
        write_json(root / final_fit.FINAL_FIT_REPORT, {"run_id": root.name})
        write_json(root / final_fit.UNIFIED_REPORT, {"run_id": root.name})
        return root / final_fit.UNIFIED_REPORT

    monkeypatch.setattr(final_fit, "fit_final_fit", close)
    result = expert_research.run(run_id="run-new", reuse_run_id="run-old")
    assert result.is_file() and len(calls) == 1
    assert (tmp_path / "run-new/models/forward/model.json").read_text(encoding="utf-8") == "new fit"
    assert not (tmp_path / "run-new/models/forward/old.cbm").exists()
    assert (previous / "models/forward/old.cbm").read_bytes() == b"old archive"
    status = json.loads((tmp_path / "run-new/status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed" and status["phase"] == "complete"


@pytest.mark.parametrize("full", [False, True])
def test_native_tuning_excludes_collapsed_median_before_pinball(monkeypatch, full):
    train = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=20), "x": np.arange(20), "target": 0.0})
    valid = train.copy()
    small_grid = [(4, 0.03), (4, 0.07)]
    full_grid = [(4, 0.03, 3.0), (4, 0.07, 3.0)]

    def fit(_train, _valid, _features, params, _seed, **kwargs):
        trees = 1 if params[1] == 0.03 else 8
        return [SimpleNamespace(tree_count_=trees) for _ in range(3)]

    def predict(_fitted, rows, _features, _audit):
        collapsed = _fitted[1].tree_count_ == 1
        median = np.zeros(len(rows)) if collapsed else np.linspace(0.01, 0.02, len(rows))
        return np.column_stack([median - 0.001, median, median + 0.001])

    monkeypatch.setattr(expert_research, "_fit_gpu", fit)
    monkeypatch.setattr(expert_research, "_predict", predict)
    monkeypatch.setattr(expert_research, "GPU_GRID", small_grid)
    monkeypatch.setattr(expert_gpu, "GRID", full_grid)
    audit = expert_research._inference_audit()
    if full:
        class Scheduler:
            def fit(self, *args, **kwargs):
                fitted = fit(*args, **kwargs)
                return fitted, [{"trees": model.tree_count_} for model in fitted]

        fitted, details = expert_research._tune_full(
            train, valid, ("x",), 42, audit, Scheduler(), repeat_winner=False,
        )
    else:
        fitted, details = expert_research._tune(train, valid, ("x",), 42, audit)
    assert fitted[1].tree_count_ == 8
    assert details["winner"]["learning_rate"] == 0.07
    assert details["candidates"][0]["validation_pinball"] < details["winner"]["validation_pinball"]
    health = expert_research.expert_score_health._learned_median_health(
        details["winner"]["tree_counts"][1], details["winner"]["validation_q50"],
    )
    assert not health["collapsed"]
