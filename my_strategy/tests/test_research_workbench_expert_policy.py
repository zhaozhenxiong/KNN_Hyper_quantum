import pandas as pd
import pytest

from my_strategy.research_workbench import expert_gpu, expert_policy_study


def test_policy_selection_purges_labels_and_keeps_complete_validation_dates():
    dates = pd.bdate_range("2024-09-02", periods=80)
    rows = pd.DataFrame({"date": dates, "label_end": dates + pd.offsets.BDay(6)})
    fit, validation = expert_policy_study._validation_split(rows)
    assert fit["label_end"].max() < validation["date"].min()
    assert len(fit) == 34 and len(validation) == 40
    assert validation["date"].iloc[0] == dates[40]


def test_policy_scores_join_by_stock_date_and_keep_unavailable_rows():
    date = pd.Timestamp("2025-03-03")
    frame = pd.DataFrame({"date": [date, date, date], "stock": ["A", "B", "C"]}, index=[9, 2, 5])
    scored = pd.DataFrame({"date": [date, date], "stock": ["B", "A"], "score": [0.1, -0.1]})
    series = expert_policy_study._scores(frame, scored)
    assert series.index.tolist() == [9, 2, 5]
    assert series.at[9] == -0.1 and series.at[2] == 0.1 and pd.isna(series.at[5])


def test_gpu_reserves_memory_already_in_use_and_keeps_both_devices(monkeypatch):
    from catboost import utils

    monkeypatch.setattr(utils, "get_gpu_device_count", lambda: 2)
    monkeypatch.setattr(expert_gpu, "_gpu_memory", lambda: [
        {"device": 0, "total_mib": 16_000, "used_mib": 3_000},
        {"device": 1, "total_mib": 16_000, "used_mib": 0},
    ])
    with expert_gpu.DualGPU() as scheduler:
        assert scheduler.ram_parts[0] == pytest.approx(0.90 - (3_000 + 768) / 16_000)
        assert scheduler.ram_parts[1] == pytest.approx(0.80)
