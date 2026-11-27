from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from my_strategy.latent_quant.execution import StudyCostModel
from my_strategy.latent_quant.runner import COSTS, PORTFOLIO
from my_strategy.latent_quant.signal_experts import ExpertForecast
from my_strategy.research_workbench import expert_hybrid_study as hybrid
from my_strategy.research_workbench.runner import Window


def _quantiles(median):
    values = np.asarray(median, dtype=float)
    return np.column_stack([values - 0.01, values, values + 0.01])


def test_rank_blend_is_exact_and_missing_multimodal_uses_knn():
    index = pd.RangeIndex(5)
    dates = pd.to_datetime(["2025-01-02"] * 4 + ["2025-01-03"])
    knn = pd.DataFrame(_quantiles([0.01, 0.03, 0.02, 0.04, 0.05]), index=index, columns=["q10", "q50", "q90"])
    knn["date"] = dates
    multi = pd.DataFrame(_quantiles([100, 200, 150, 300, 400]), index=index, columns=["q10", "q50", "q90"])
    multi.loc[2, ["q10", "q50", "q90"]] = np.nan
    result = hybrid._blend_predictions(knn, multi)
    expected_knn = pd.Series([0.25, 0.75, 0.50, 1.0, 1.0], index=index)
    expected_multi = pd.Series([1 / 3, 2 / 3, np.nan, 1.0, 1.0], index=index)
    pd.testing.assert_series_equal(result["knn_rank"], expected_knn.rename("knn_rank"))
    pd.testing.assert_series_equal(result["multimodal_rank"], expected_multi.rename("multimodal_rank"))
    np.testing.assert_allclose(result.loc[[0, 1, 3, 4], "hybrid_score"],
                               0.6 * expected_knn.loc[[0, 1, 3, 4]] + 0.4 * expected_multi.loc[[0, 1, 3, 4]])
    assert result.loc[2, "hybrid_score"] == 0.5
    assert result.loc[2, "effective_knn_weight"] == 1.0
    assert result.loc[2, "effective_multimodal_weight"] == 0.0
    assert result.loc[2, "fallback_reason"] == "multimodal_missing_or_invalid"
    assert not {"q10", "q50", "q90"}.intersection(result.columns)


def test_invalid_knn_has_no_score_and_unordered_multimodal_is_not_used():
    index = pd.RangeIndex(2)
    dates = pd.to_datetime(["2025-01-02"] * 2)
    knn = pd.DataFrame(_quantiles([0.01, 0.02]), index=index, columns=["q10", "q50", "q90"])
    knn["date"] = dates
    multi = knn[["q10", "q50", "q90"]].copy()
    knn.loc[0, "q50"] = np.nan
    multi.loc[1, ["q10", "q50", "q90"]] = [0.03, 0.02, 0.01]
    result = hybrid._blend_predictions(knn, multi)
    assert np.isnan(result.loc[0, "hybrid_score"])
    assert result.loc[0, "effective_knn_weight"] == 0.0
    assert result.loc[0, "effective_multimodal_weight"] == 0.0
    assert result.loc[1, "fallback_reason"] == "multimodal_missing_or_invalid"
    assert result.loc[1, "effective_knn_weight"] == 1.0


def test_fold_blends_evaluation_rows_without_slicing_calibration_twice(tmp_path, monkeypatch):
    dates = pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-06", "2025-01-07", "2025-01-09", "2025-01-10",
    ])
    frame = pd.DataFrame({
        "date": dates, "stock": "000001.SZ", "signal_eligible": True,
        "label_end": dates + pd.Timedelta(days=1), "target": np.linspace(-0.02, 0.02, len(dates)),
        "trade_date": dates + pd.Timedelta(days=1),
    })
    window = Window("oos_1", dates[0], dates[3], dates[5], pd.DatetimeIndex(dates[-2:]))
    quantiles = _quantiles([-0.01, 0.01, 0.03])

    def native_fold(data, window, context, features, **kwargs):
        assert kwargs["include_raw_quantiles"] is True
        folder = context.subdir("backtests", window.period)
        evaluation = data.frame.loc[data.frame["date"].isin(window.evaluate)]
        native = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
        native = native.join(pd.DataFrame(_quantiles([0.02, 0.04]), index=native.index, columns=["q10", "q50", "q90"]))
        native["score"] = native["q50"]
        native["raw_q10"], native["raw_q50"], native["raw_q90"] = native["q10"], native["q50"], native["q90"]
        from my_strategy.research_workbench.runner import _write_table

        _write_table(folder / "fusion_predictions.parquet", native)
        execution = {cost: {"total_return": 0.01, "verification": {"cash": True}} for cost in COSTS}
        fold = {
            "period": window.period, "arms": {"fusion": {"execution": execution}},
            "models": {"arms": {}}, "fusion": {"winner": {"weights": {"domestic_gpu": 1.0}}},
            "model_health": {"passed": True},
        }
        return fold, {cost: {"fusion": pd.Series(dtype=float)} for cost in COSTS}

    def forecast(train, rows, *, names, knn_device, knn_query_chunk):
        assert names == (hybrid.KNN_ARM,)
        assert knn_device == "cpu" and knn_query_chunk == 7
        assert len(rows) == 3
        return {hybrid.KNN_ARM: ExpertForecast(
            quantiles, pd.Series(np.zeros(len(rows)), index=rows.index),
            pd.Series(np.ones(len(rows)), index=rows.index),
            {"causal": True, "backend": {"actual_device": "cpu", "backend": "numpy_cpu_float64"}},
        )}

    def execute(data, prediction, sessions, context, folder, arm, policy, cost_model):
        assert len(prediction) == len(sessions) == 2
        execution = {
            cost: {"total_return": 0.02, "verification": {"cash": True, "equity": True, "next_session": True}}
            for cost in COSTS
        }
        return execution, {cost: pd.Series(dtype=float) for cost in COSTS}

    class Context:
        run_id = "hybrid-shape-test"

        def subdir(self, *parts):
            path = tmp_path.joinpath(*parts)
            path.mkdir(parents=True, exist_ok=True)
            return path

    monkeypatch.setattr(hybrid.research, "_fold", native_fold)
    monkeypatch.setattr(hybrid, "forecast_experts", forecast)
    monkeypatch.setattr(hybrid.research, "_execution", execute)
    policy = replace(PORTFOLIO, selection_mode="cross_sectional_rank", min_score=None)
    fold, _ = hybrid._fold(
        SimpleNamespace(frame=frame), window, Context(), {}, policy=policy, cost_model=StudyCostModel(),
        seed=42, scheduler=None, device="cpu", chunk=7,
    )
    assert set(fold["arms"]) == {"fusion", hybrid.KNN_ARM, hybrid.ARM}
    saved = pd.read_parquet(tmp_path / "backtests/oos_1" / f"{hybrid.ARM}_predictions.parquet")
    assert len(saved) == 2
    assert saved["hybrid_score"].notna().all()
    assert saved["effective_knn_weight"].eq(0.6).all()
    assert saved["effective_multimodal_weight"].eq(0.4).all()
    assert not {"q10", "q50", "q90"}.intersection(saved.columns)
