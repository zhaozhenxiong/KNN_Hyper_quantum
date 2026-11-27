"""Synthetic checks for chronological quantile fitting and replay."""

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.models import (
    ARMS,
    _historical_quantiles,
    _trend_quantiles,
    predict_fold,
    replay_catboost,
)
from my_strategy.research_workbench.runner import Window


def _synthetic_panel() -> tuple[pd.DataFrame, Window]:
    dates = pd.bdate_range("2025-01-02", periods=90)
    frame = pd.DataFrame({
        "date": dates.repeat(4),
        "stock": np.tile(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"], len(dates)),
        "label_end": np.repeat(pd.Series(dates).shift(-6).to_numpy(), 4),
        "signal_eligible": True,
    })
    rng = np.random.default_rng(77)
    frame["momentum"] = rng.normal(size=len(frame))
    frame["st_direction"] = np.where(frame["momentum"].gt(0), 1, -1)
    frame["atr_pct"] = rng.uniform(0.01, 0.05, len(frame))
    frame["st_distance"] = frame["momentum"] * 0.01
    frame["target"] = (0.004 * frame["momentum"] + rng.normal(0, 0.006, len(frame))).where(frame["label_end"].notna())
    frame.loc[dates[73] == frame["date"], "target"] = np.nan
    frame.loc[dates[74] == frame["date"], "signal_eligible"] = False
    return frame, Window("evaluation", dates[0], dates[40], dates[56], dates[72:])


def test_quantiles_fall_back_to_pooled_history_for_sparse_or_unknown_groups():
    train = pd.DataFrame({
        "stock": ["600000.SH"] * 20 + ["600001.SH"],
        "target": np.arange(21, dtype=float),
        "st_direction": np.nan,
        "atr_pct": np.nan,
    })
    rows = pd.DataFrame({
        "stock": ["600000.SH", "600001.SH", "600002.SH"],
        "st_direction": [-1.0, np.nan, 1.0],
        "atr_pct": [0.01, np.nan, 0.03],
    })
    pooled = np.array([2.0, 10.0, 18.0])
    empirical, evidence = _historical_quantiles(train, rows, pooled)
    assert evidence["stock_groups"] == 1
    np.testing.assert_allclose(empirical[0], [1.9, 9.5, 17.1])
    np.testing.assert_allclose(empirical[1:], np.tile(pooled, (2, 1)))
    trend, evidence = _trend_quantiles(train, rows, pooled)
    assert evidence["direction_atr_groups"] == 0
    np.testing.assert_allclose(trend, np.tile(pooled, (3, 1)))


def test_predict_fold_orders_quantiles_replays_models_and_ignores_future_perturbations(tmp_path):
    pytest.importorskip("catboost")
    frame, window = _synthetic_panel()
    base = ("momentum",)
    extra = ("st_direction", "atr_pct", "st_distance")
    predictions, evidence = predict_fold(frame, window, base, extra, seed=7, model_dir=tmp_path, iterations=8)

    expected = frame.index[frame["date"].isin(window.evaluate) & frame["signal_eligible"]]
    assert tuple(predictions) == ARMS
    for table in predictions.values():
        pd.testing.assert_index_equal(table.index, expected)
        assert np.isfinite(table).all(axis=None)
        assert table["q10"].le(table["q50"]).all()
        assert table["q50"].le(table["q90"]).all()
        assert frame.loc[table.index, "target"].isna().any()
    for name, cutoff in zip(("train", "validation", "calibration"),
                            (window.valid_start, window.calibration_start, window.evaluate[0]), strict=True):
        assert pd.Timestamp(evidence[name]["last_label_end"]) < cutoff
    assert evidence["arms"]["catboost_base"]["params"] == evidence["arms"]["catboost_supertrend_atr"]["params"]
    assert evidence["arms"]["catboost_base"]["features"] == list(base)
    assert evidence["arms"]["catboost_supertrend_atr"]["features"] == [*base, *extra]
    for arm in ARMS[2:]:
        metadata = evidence["arms"][arm]
        assert metadata["calibration"]["final_crossing_rate"] == 0.0
        replay = replay_catboost(frame.loc[expected], tmp_path / metadata["model_file"])
        pd.testing.assert_frame_equal(predictions[arm], replay)
    trend = predictions["supertrend_atr"]
    assert (np.sign(trend["score"]) == frame.loc[trend.index, "st_direction"]).all()

    perturbed = frame.copy()
    signal_cutoff = window.evaluate[5]
    future = perturbed["date"].gt(signal_cutoff)
    perturbed.loc[future, [*base, *extra, "target"]] *= 100
    for cutoff in (window.valid_start, window.calibration_start, window.evaluate[0]):
        crosses_boundary = perturbed["date"].lt(cutoff) & perturbed["label_end"].ge(cutoff)
        perturbed.loc[crosses_boundary, "target"] = 10_000
    changed, _ = predict_fold(perturbed, window, base, extra, seed=7, model_dir=None, iterations=8)
    early = expected[frame.loc[expected, "date"].le(signal_cutoff)]
    for arm in ARMS:
        pd.testing.assert_frame_equal(predictions[arm].loc[early], changed[arm].loc[early])
    later = expected[frame.loc[expected, "date"].gt(signal_cutoff)]
    assert not changed["supertrend_atr"].loc[later, "score"].equals(trend.loc[later, "score"])
