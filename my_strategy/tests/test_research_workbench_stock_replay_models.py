"""Frozen-rule, GPU fallback and same-date model replay regression checks."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.execution import Strategy, StudyCostModel
from my_strategy.research_workbench import stock_replay_models as subject
from my_strategy.research_workbench.runner import Window


def _write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_rule_replay_uses_only_frozen_train_calibration_for_a_new_stock(tmp_path):
    dates = pd.bdate_range("2024-01-04", periods=70)
    frame = pd.DataFrame({
        "date": dates, "stock": "600001.SH", "signal_eligible": True,
        "target": np.linspace(-0.10, 0.15, len(dates)), "label_end": pd.Series(dates).shift(-6),
        "st_direction": np.where(np.arange(len(dates)) % 2, 1, -1),
        "atr_pct": 0.02, "st_distance": 0.015,
    })
    window = Window("case", dates[0], dates[34], dates[45], dates[61:])
    tm, _, cm = window.masks(frame)
    train, cal = frame.loc[tm], frame.loc[cm]
    rows = frame.loc[frame["date"].isin(window.evaluate)].assign(stock="600002.SH")
    observations = pd.concat([cal, rows])
    pooled = train["target"].quantile(list(subject.models.QUANTILE_LEVELS)).to_numpy()
    raw_hist, _ = subject.models._historical_quantiles(train, observations, pooled)
    raw_trend, _ = subject.models._trend_quantiles(train, observations, pooled)
    details = {}
    for key, raw in (("historical_quantiles", raw_hist), ("supertrend_atr", raw_trend)):
        _, details[key] = subject.models._calibrate(raw, cal["target"].to_numpy())
    db = tmp_path / "data_snapshots" / "panel.parquet"
    db.parent.mkdir()
    frame.to_parquet(db)
    evidence = {key: {"calibration": meta} for key, meta in details.items()}
    expected = subject._rule_predictions(tmp_path, window, rows.drop(columns="target"), evidence)
    assert np.repeat(expected["historical_quantiles"].to_numpy()[[0]], len(rows), axis=0).tolist() == expected["historical_quantiles"].to_numpy().tolist()
    changed = rows.assign(target=5000.0, label_end=dates[-1])
    for name, values in subject._rule_predictions(tmp_path, window, changed, evidence).items():
        pd.testing.assert_frame_equal(values, expected[name])


def test_overseas_replay_keeps_frozen_date_columns_and_uses_multivariate_cuda_once(tmp_path, monkeypatch):
    catboost = pytest.importorskip("catboost")
    days = pd.bdate_range("2025-01-02", periods=4)
    frame = pd.DataFrame({"date": days, "stock": "600000.SH", "signal_eligible": True, "base": 2.0, "spx": 99.0})
    scope = tmp_path / "data_snapshots"
    scope.mkdir()
    pd.DataFrame({"date": days, "spx": [0.01, 0.02, 0.03, 0.04]}).to_parquet(scope / "overseas.parquet")
    _write(tmp_path / "config" / "study.json", {
        "portfolio": asdict(Strategy()), "cost_model": asdict(StudyCostModel()), "initial_cash": subject.INITIAL_CASH,
        "cost_multipliers": subject.COSTS, "overseas_features": ["spx"],
    })
    folder = tmp_path / "models" / "oos_1"
    _write(folder / "pca.json", {"output_features": ["overseas_pc1"]})
    pd.DataFrame({"date": days, "overseas_pc1": [0.2, 0.3, 0.4, 0.5]}).to_parquet(folder / "pca_by_date.parquet")
    arms = {"domestic": ["base"], "overseas_raw": ["base", "spx"], "overseas_pca": ["base", "spx", "overseas_pc1"]}
    evidence = {
        key: {"model_file": f"{key}.cbm", "features": cols, "calibration": {"offsets": [0.0, 0.0, 0.0]}}
        for key, cols in arms.items()
    }
    for model, meta in evidence.items():
        (folder / meta["model_file"]).touch()
        _write(folder / f"{model}.json", meta)
    report = {"folds": [{
        "period": "oos_1", "arms": {name: {} for name in arms}, "models": {"arms": evidence},
        "evaluation_start": str(days[1]), "evaluation_end": str(days[-1]),
    }]}
    attempts = []

    class Regressor:
        def load_model(self, filename):
            self.name = Path(filename).stem

        def predict(self, data, thread_count, task_type):
            attempts.append((self.name, task_type))
            if task_type == "GPU":
                raise catboost.CatBoostError("Model is not one-dimensional")
            return np.broadcast_to([-0.1, 0.04, 0.20], (len(data), 3))

    def predict_cuda(model, values):
        assert list(values.columns) == arms[model.name]
        if "spx" in values:
            np.testing.assert_allclose(values["spx"], [0.02, 0.03, 0.04])
        attempts.append((model.name, "PyTorch CUDA"))
        return np.broadcast_to([-0.1, 0.04, 0.20], (len(values), 3))

    alternative = ModuleType("stock_replay_cuda")
    alternative.FrozenCudaUnavailable = RuntimeError
    alternative.predict_frozen_on_cuda = predict_cuda
    monkeypatch.setattr(catboost, "CatBoostRegressor", Regressor)
    monkeypatch.setitem(sys.modules, "my_strategy.research_workbench.stock_replay_cuda", alternative)
    result, _, devices = subject.replay(tmp_path, "overseas", report, "oos_1", frame, days)
    assert set(result) == set(arms) and all(len(df) == 3 for df in result.values())
    assert list(devices["effective_devices"].values()) == ["PyTorch CUDA"] * 3
    assert devices["fallback_reason"] is None and "not one-dimensional" in devices["native_gpu_limit"]
    assert sum(device == "GPU" for _, device in attempts) == 1
    assert devices["max_absolute_differences"] == dict.fromkeys(arms, 0.0)
