import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench import expert_calibration as calibration
from my_strategy.research_workbench import expert_online_knn_calibration as online


def test_settled_accepts_target_metadata_without_relaxing_sample_proof():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    frame = pd.DataFrame({
        "date": dates,
        "label_end": pd.to_datetime(["2025-01-08", "2025-01-09"]),
        "target": [0.01, -0.02],
        "signal_eligible": [True, True],
    })
    proof = {
        "rows": 2, "dates": 2, "start": "2025-01-02", "end": "2025-01-03",
        "last_label_end": "2025-01-09", "target": {"rows": 2},
    }
    pd.testing.assert_frame_equal(calibration._settled(frame, proof), frame)
    with pytest.raises(ValueError, match="成熟标签样本"):
        calibration._settled(frame, proof | {"rows": 1})


def test_verify_irreversible_crossings_with_native_predictions():
    raw = np.array([[-0.04, 0.01, 0.05], [-0.02, 0.02, 0.06]])
    offsets = np.array([0.0, -0.02, 0.01])
    archived = calibration._apply(raw, offsets)
    values, proof = calibration._invert(archived, offsets, 0)
    np.testing.assert_allclose(values, raw)
    assert proof == "inverse_no_crossing"

    crossed = np.array([0.10, -0.02, 0.01])
    archived = calibration._apply(raw, crossed)
    with pytest.raises(ValueError, match="原生推理"):
        calibration._invert(archived, crossed, 1)
    values, proof = calibration._invert(archived, crossed, 1, replay=lambda: raw)
    np.testing.assert_array_equal(values, raw)
    assert proof == "verified_native_replay"
    with pytest.raises(ValueError, match="无法复算"):
        calibration._invert(archived, crossed, 1, replay=lambda: raw + 1)


def test_date_equal_calibration_is_unchanged_by_same_day_stock_duplication():
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    residuals = np.array([[1.0] * 3, [2.0] * 3, [8.0] * 3])
    expected = np.array([1, 2, 8])
    np.testing.assert_array_equal(calibration._date_equal(residuals, dates), expected)
    indices = [0] * 25 + [1, 2]
    np.testing.assert_array_equal(calibration._date_equal(residuals[indices], dates.iloc[indices]), expected)


def test_diagnostic_keeps_latest_signals_without_future_outcomes():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-03"])
    rows = pd.DataFrame({"date": dates, "stock": ["A", "A", "B"], "target": [0.01, np.nan, np.nan]})
    forecasts = np.array([[-0.01, 0.01, 0.02]] * 3)
    daily, proof = calibration._daily(rows, forecasts, "raw")
    assert daily["positive_scores"].tolist() == [1, 2]
    assert daily["matured_rows"].tolist() == [1, 0]
    assert proof["rows"] == 1 and proof["pinball_q50"] == 0
    assert daily["top10_mean_matured_excess"].isna().all()


def test_knn_residual_calibrator_is_conditional_and_label_isolated(monkeypatch):
    monkeypatch.setattr(calibration, "KNN_NEIGHBORS", 3)
    monkeypatch.setattr(calibration, "KNN_PER_DATE", 4)
    monkeypatch.setattr(calibration, "KNN_CHUNK", 2)
    dates = pd.date_range("2025-01-02", periods=4, freq="B").repeat(8)
    q50 = np.tile(np.linspace(-0.004, 0.004, 8), 4)
    cal_raw = np.column_stack([q50 - 0.02, q50, q50 + 0.02])
    correction = np.where(q50 > 0, 0.01, -0.01)
    cal = pd.DataFrame({
        "date": dates, "label_end": dates + pd.Timedelta(days=1),
        "target": q50 + correction,
    }, index=np.arange(100, 132))
    evaluation_dates = pd.date_range("2025-01-09", periods=2, freq="B").repeat(8)
    evaluation_q50 = np.tile(q50[:8], 2)
    evaluation_raw = np.column_stack([evaluation_q50 - 0.02, evaluation_q50, evaluation_q50 + 0.02])
    evaluation = pd.DataFrame({"date": evaluation_dates, "target": 999.0}, index=np.arange(200, 216))

    sample = calibration._knn_calibration_sample(cal, cal_raw)
    assert len(sample) == 16
    calibrated, evidence = calibration._knn_residual_quantiles(cal, cal_raw, evaluation, evaluation_raw)
    np.testing.assert_allclose(calibrated[:, 1], evaluation_q50 + correction[:16], atol=2e-12)
    assert np.isfinite(calibrated).all() and (np.diff(calibrated, axis=1) >= 0).all()
    assert evidence["sample_rows"] == 16 and evidence["neighbors"] == 3
    evaluation["target"] = -999.0
    repeated, _ = calibration._knn_residual_quantiles(cal, cal_raw, evaluation, evaluation_raw)
    np.testing.assert_array_equal(calibrated, repeated)


def test_knn_residual_calibrator_rejects_unmatured_calibration_labels():
    cal = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02"]), "label_end": pd.to_datetime(["2025-01-03"]), "target": [0.01],
    })
    rows = pd.DataFrame({"date": pd.to_datetime(["2025-01-03"])})
    with pytest.raises(ValueError, match="越过评估起点"):
        calibration._knn_residual_quantiles(cal, np.array([[0.0, 0.01, 0.02]]), rows, np.array([[0.0, 0.01, 0.02]]))


def test_online_knn_uses_only_previously_matured_labels_and_ignores_future_changes(monkeypatch):
    monkeypatch.setattr(calibration, "KNN_NEIGHBORS", 3)
    monkeypatch.setattr(calibration, "KNN_PER_DATE", 4)
    monkeypatch.setattr(calibration, "KNN_CHUNK", 2)
    cal_dates = pd.date_range("2025-01-02", periods=3, freq="B").repeat(4)
    eval_dates = pd.date_range("2025-01-08", periods=6, freq="B").repeat(4)
    q50 = np.tile(np.linspace(-0.004, 0.004, 4), 9)
    raw = np.column_stack([q50 - 0.02, q50, q50 + 0.02])
    dates = cal_dates.append(eval_dates)
    rows = pd.DataFrame({
        "date": dates, "label_end": dates + pd.Timedelta(days=1),
        "target": q50 + np.repeat(np.linspace(0.01, -0.02, 9), 4),
    })
    cal, evaluation = rows.iloc[:12].copy(), rows.iloc[12:].copy().reset_index(drop=True)
    values, evidence = online.online_knn_residual_quantiles(cal, raw[:12], evaluation, raw[12:], lookback_sessions=3)
    assert np.isfinite(values).all() and (np.diff(values, axis=1) >= 0).all()
    assert len(evidence["daily"]) == 6
    assert all(item["last_label_end"] < item["date"] for item in evidence["daily"])
    assert all(item["history_sessions"] <= 3 for item in evidence["daily"])

    cutoff = pd.Timestamp(evidence["daily"][3]["date"])
    changed = evaluation.copy()
    future = changed["date"].ge(cutoff)
    changed.loc[future, "target"] = 999.0
    changed.loc[future, "label_end"] = pd.Timestamp("2026-01-01")
    repeated, _ = online.online_knn_residual_quantiles(cal, raw[:12], changed, raw[12:], lookback_sessions=3)
    np.testing.assert_array_equal(values[evaluation["date"].lt(cutoff)], repeated[evaluation["date"].lt(cutoff)])
