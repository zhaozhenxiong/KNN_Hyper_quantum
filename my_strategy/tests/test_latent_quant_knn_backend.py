import logging

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant import knn_backend
from my_strategy.latent_quant.signal_experts import forecast_experts

from test_latent_quant_signal_experts import _panel, _split


def test_distance_ties_use_original_sample_position_on_cpu_and_cuda():
    samples = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
    targets = np.array([0.0, 1.0, 2.0, 100.0, 200.0])
    queries = np.zeros((2, 3))
    cpu = knn_backend.KNNBackend("cpu", query_chunk=1).quantiles(samples, targets, queries, neighbors=3)
    np.testing.assert_allclose(cpu, np.tile([0.2, 1.0, 1.8], (2, 1)))
    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("CUDA unavailable")
    gpu = knn_backend.KNNBackend("cuda:1", query_chunk=1).quantiles(samples, targets, queries, neighbors=3)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-12)


def test_auto_fallback_is_audited_and_explicit_cuda_does_not_hide_errors(monkeypatch, caplog):
    def unavailable():
        raise ImportError("test CUDA unavailable")

    monkeypatch.setattr(knn_backend, "_load_torch", unavailable)
    with caplog.at_level(logging.WARNING, logger=knn_backend.__name__):
        backend = knn_backend.KNNBackend("auto")
    assert backend.device == "cpu"
    assert "test CUDA unavailable" in backend.evidence["fallback_reason"]
    assert "整折明确回退 CPU" in caplog.text
    with pytest.raises(RuntimeError, match="显式请求 CUDA"):
        knn_backend.KNNBackend("cuda")


def test_real_panel_cpu_cuda_quantiles_and_causality_match():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    train, rows = _split(_panel(stocks=12, days=180))
    cpu = forecast_experts(train, rows, ("online_knn_volume_trend",), knn_device="cpu")["online_knn_volume_trend"]
    gpu = forecast_experts(train, rows, ("online_knn_volume_trend",), knn_device="cuda:1")["online_knn_volume_trend"]
    assert gpu.evidence["backend"]["backend"] == "torch_cuda_float64"
    assert gpu.evidence["backend"]["query_rows"] == gpu.evidence["knn_rows"] > 0
    np.testing.assert_allclose(gpu.quantiles, cpu.quantiles, rtol=1e-11, atol=1e-12)

    dates = pd.DatetimeIndex(rows["date"].unique())
    cutoff = dates[len(dates) // 2]
    changed = rows.copy()
    future = changed["date"].ge(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 3
    changed.loc[future, "volume"] *= 10
    changed.loc[future, "target"] = 100.0
    stressed = forecast_experts(train, changed, ("online_knn_volume_trend",), knn_device="cuda:1")["online_knn_volume_trend"]
    before = rows["date"].lt(cutoff).to_numpy()
    np.testing.assert_allclose(stressed.quantiles[before], gpu.quantiles[before], rtol=1e-11, atol=1e-12)
