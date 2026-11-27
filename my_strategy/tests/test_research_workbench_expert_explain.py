"""Checks for bounded read-only raw q50 expert attribution."""

from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench import expert_explain as subject
from my_strategy.storage.expert_artifacts import ExpertArtifactStore


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_common_sample_stable_and_bounded_per_date():
    days = pd.bdate_range("2025-01-02", periods=150)
    frame = pd.DataFrame({"date": days.repeat(3), "stock": ["600000.SH", "000001.SZ", "600001.SH"] * len(days)})
    sample = subject._sample(frame)
    shuffled = subject._sample(frame.sample(frac=1, random_state=44))
    pd.testing.assert_frame_equal(sample, shuffled)
    assert len(sample) == sample["date"].nunique() == subject.MAX_SESSIONS
    assert sample["date"].iloc[[0, -1]].tolist() == [days[0], days[-1]]


def test_model_hash_and_native_additivity_are_mandatory(tmp_path, monkeypatch):
    catboost = pytest.importorskip("catboost")
    relative = "models/oos_1/domestic_gpu/q50.cbm"
    model = tmp_path / relative
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic sealed model fixture")
    record = {"model_files": {"q50": relative}, "model_file_sha256": {"q50": subject._sha(model)}, "features": ["a", "b"]}

    class Regressor:
        feature_names_ = ["a", "b"]
        drift = 0.0

        def load_model(self, path):
            assert path == str(model)

        def get_param(self, key):
            assert key == "loss_function"
            return "Quantile:alpha=0.5"

        def get_feature_importance(self, values, **kwargs):
            assert kwargs["type"] == "ShapValues"
            return values[["a", "b"]].assign(base=0.04).to_numpy()

        def predict(self, values, **kwargs):
            assert kwargs["task_type"] == "CPU"
            return values[["a", "b"]].assign(base=0.04).sum(axis=1).to_numpy() + self.drift

    monkeypatch.setattr(catboost, "CatBoostRegressor", Regressor)
    monkeypatch.setattr(catboost, "Pool", lambda frame: frame)
    frame = pd.DataFrame({"a": [0.125, 0.25], "b": [0.5, -0.25], "target": [1000, 2000]})
    report = subject._explain_model(tmp_path, "oos_1", "domestic_gpu", record, frame)
    assert report["max_additivity_error"] <= 1e-10
    assert [item["feature"] for item in report["top5"]] == ["b", "a"]
    np.testing.assert_allclose(report["top5"][0]["mean_abs_shap"], 0.375)
    Regressor.drift = 1e-4
    with pytest.raises(ValueError, match="加和验证"):
        subject._explain_model(tmp_path, "oos_1", "domestic_gpu", record, frame)
    model.write_bytes(b"different data")
    with pytest.raises(ValueError, match="哈希不符"):
        subject._explain_model(tmp_path, "oos_1", "domestic_gpu", record, frame)


def test_outside_frozen_roster_is_explicit_and_has_no_side_effects(tmp_path, monkeypatch):
    run_id = "expert-fixture"
    root = tmp_path / "run"
    roster = tmp_path / "roster.parquet"
    pd.DataFrame({"stock": ["002650.SZ"]}).to_parquet(roster)
    store = ExpertArtifactStore(tmp_path / "content")
    reference = store.put_file(roster)
    store.write_manifest(run_id, {"frozen_inputs/roster/roster.parquet": reference})
    _json(root / "status.json", {"run_id": run_id, "status": "completed"})
    _json(root / "reports/expert_research.json", {"folds": []})
    _json(root / "config/study.json", {"sources": {"roster": asdict(reference)}})
    monkeypatch.setattr(subject, "artifact_run_dir", lambda value, create: root)
    monkeypatch.setattr(subject, "ExpertArtifactStore", lambda: store)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subject.explain(run_id, symbol="600000.SH")
    assert result["status"] == "not_covered" and result["folds"] == []
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_fold_rejects_unmatured_validation_labels():
    fold = {"evaluation_start": "2025-04-01", "evaluation_end": "2025-06-01", "models": {
        "train": {"last_label_end": "2025-01-20"}, "validation": {"start": "2025-02-01", "last_label_end": "2025-03-01"},
        "calibration": {"start": "2025-03-01", "last_label_end": "2025-03-21"},
    }}
    with pytest.raises(ValueError, match="时点边界"):
        subject._validate_fold(fold)
