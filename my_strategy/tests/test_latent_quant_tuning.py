"""Synthetic lookahead and replay checks for the overseas tuning stage."""

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant import tuning
from my_strategy.latent_quant.models import replay_catboost
from my_strategy.research_workbench.runner import Window


CONTINUOUS = ("spx", "nikkei", "kospi")


def _panel() -> tuple[pd.DataFrame, Window]:
    rng = np.random.default_rng(437)
    dates = pd.bdate_range("2024-07-01", periods=80)
    overseas = pd.DataFrame(rng.normal(size=(len(dates), len(CONTINUOUS))), columns=CONTINUOUS)
    overseas["date"] = dates
    overseas["label_end"] = pd.Series(dates).shift(-6)
    overseas.loc[7, "spx"] = np.nan
    overseas.loc[70, list(CONTINUOUS)] = np.nan
    overseas["fed_event"] = np.arange(len(dates)) % 10 == 0
    repeated = 1 + np.arange(len(dates)) % 4
    frame = overseas.loc[overseas.index.repeat(repeated)].reset_index(drop=True)
    frame["stock"] = frame.groupby("date").cumcount().map(lambda item: f"600{item:03d}.SH")
    frame["momentum"] = rng.normal(size=len(frame))
    frame["signal_eligible"] = frame["date"].ne(dates[68])
    frame["target"] = (
        0.03 * frame["momentum"] + 0.02 * frame["spx"].fillna(0) + rng.normal(0., 0.01, len(frame))
    ).where(frame["label_end"].notna() & frame["date"].ne(dates[67]))
    return frame, Window("synthetic", dates[0], dates[34], dates[50], dates[66:])


def test_pca_uses_one_mature_training_row_per_date_and_can_replay():
    frame, window = _panel()
    fit, metadata = tuning.add_fold_pca(frame, window, CONTINUOUS)
    train = window.masks(frame)[0]
    days = frame.loc[train].drop_duplicates("date")
    assert metadata["train_dates"] == len(days)
    np.testing.assert_allclose(metadata["centers"], days[list(CONTINUOUS)].mean())
    np.testing.assert_allclose(metadata["scales"], days[list(CONTINUOUS)].std(ddof=0))
    assert metadata["observed_train_dates"]["spx"] == len(days) - 1
    assert np.asarray(metadata["explained_variance_ratio"]).sum() < 1.0
    pd.testing.assert_frame_equal(fit, tuning.apply_fold_pca(frame, metadata))
    assert fit.loc[frame[list(CONTINUOUS)].isna().all(axis=1), metadata["output_features"]].isna().all(axis=None)
    assert fit["fed_event"].equals(frame["fed_event"])

    updated = frame.copy()
    outside = updated["date"].ge(window.valid_start) | updated["label_end"].ge(window.valid_start)
    updated.loc[outside, list(CONTINUOUS)] *= 100.0
    _, updated_metadata = tuning.add_fold_pca(updated, window, CONTINUOUS)
    assert updated_metadata == metadata

    conflict = frame.copy()
    index = conflict.index[conflict["date"].eq(window.evaluate[0])][0]
    conflict.loc[index, "spx"] += 1
    with pytest.raises(ValueError, match="single shared observation"):
        tuning.apply_fold_pca(conflict, metadata)


def test_grid_uses_date_weighted_validation_and_preserves_past_signals(tmp_path, monkeypatch):
    pytest.importorskip("catboost")
    monkeypatch.setattr(tuning, "ITERATIONS", 24)
    frame, window = _panel()
    frame, pca = tuning.add_fold_pca(frame, window, CONTINUOUS)
    feature_sets = {
        "domestic": ("momentum",),
        "overseas_raw": ("momentum", *CONTINUOUS, "fed_event"),
        "overseas_pca": ("momentum", *CONTINUOUS, "fed_event", *pca["output_features"]),
    }
    predictions, evidence = tuning.tune_fold(frame, window, feature_sets, seed=33, model_dir=tmp_path)
    ready = frame["signal_eligible"] & frame["date"].isin(window.evaluate)
    assert tuple(predictions) == tuple(feature_sets)
    for name, end in zip(
        ("train", "validation", "calibration"), (window.valid_start, window.calibration_start, window.evaluate[0]), strict=True,
    ):
        assert pd.Timestamp(evidence[name]["last_label_end"]) < end
    for name, output in predictions.items():
        arm = evidence["arms"][name]
        pd.testing.assert_index_equal(output.index, frame.index[ready])
        assert frame.loc[output.index, "target"].isna().any()
        assert output["q10"].le(output["q50"]).all() and output["q50"].le(output["q90"]).all()
        assert len(arm["candidates"]) == 4
        assert {(x["params"]["depth"], x["params"]["learning_rate"]) for x in arm["candidates"]} == set(tuning.GRID)
        assert arm["winner"] == min(arm["candidates"], key=lambda item: item["validation_pinball"])
        for trial in arm["candidates"]:
            assert trial["tree_count"] == trial["best_iteration"] + 1
            assert trial["validation_pinball"] == pytest.approx(trial["validation_recomputed_pinball"], abs=1e-6)
        pd.testing.assert_frame_equal(output, replay_catboost(frame.loc[ready], tmp_path / arm["model_file"]))

    changed = frame.copy()
    cutoff = window.evaluate[4]
    future = changed["date"].gt(cutoff)
    numeric = ["momentum", *CONTINUOUS, *pca["output_features"], "target"]
    changed.loc[future, numeric] += 500
    crossing = changed["date"].lt(window.evaluate[0]) & changed["label_end"].ge(window.evaluate[0])
    changed.loc[crossing, "target"] = 10_000
    perturbed, amended = tuning.tune_fold(changed, window, feature_sets, seed=33, model_dir=None)
    earlier = frame.index[ready & ~future]
    for name, output in predictions.items():
        pd.testing.assert_frame_equal(output.loc[earlier], perturbed[name].loc[earlier])
        assert evidence["arms"][name]["winner"] == amended["arms"][name]["winner"]
