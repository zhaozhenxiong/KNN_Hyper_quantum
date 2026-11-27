from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.research_workbench.backtest import Strategy, simulate, target_weights
from my_strategy.research_workbench.data import _load_benchmark, _reference_amounts
from my_strategy.research_workbench.runner import _fold, _rank_by_classification, _studies, _study_cohorts, _windows


def test_broker_uses_only_prior_session_score_and_liquidity():
    dates = pd.bdate_range("2026-04-01", periods=4)
    frame = pd.DataFrame(
        {
            "date": dates,
            "stock": ["600000.SH"] * 4,
            "board": ["sh_main"] * 4,
            "industry": ["known"] * 4,
            "has_trade_price": [True] * 4,
            "trade_open": [10.0, 10.45, 10.4, 10.55],
            "trade_close": [10.2, 10.6, 10.6, 10.55],
            "amount": [20_000_000, 0, 20_000_000, 20_000_000],
        }
    )
    benchmark = pd.Series([0, .01, .01, .01], index=dates)
    scores = pd.Series([1.0, 1.0, 1.0, 1.0])
    result = simulate(
        frame, scores, dates, benchmark,
        Strategy(top_n=1, rebalance_days=1, score_threshold=0, max_weight=0.1),
        run_id="chronology-fixture",
    )
    first = result["trades"][0]
    assert first["signal_date"] == "2026-04-01"
    assert first["date"] == "2026-04-02"
    assert first["price"] == pytest.approx(10.45 * 1.0005)
    assert first["fee"] > 0
    assert all(row["cash"] >= 0 for row in result["equity"])
    assert all(row["date"] != "2026-04-01" for row in result["trades"])

    frame.at[0, "amount"] = 0
    rejected = simulate(frame, scores, dates, benchmark, Strategy(top_n=1, rebalance_days=1), run_id="no-lookahead")
    assert not any(row["date"] == "2026-04-02" for row in rejected["trades"])
    assert {"date": "2026-04-02", "stock": "600000.SH", "side": "BUY", "reason": "no_previous_session_liquidity"} in rejected["rejected"]


def test_risk_caps_apply_to_known_industry_and_each_position():
    frame = pd.DataFrame(
        {"stock": ["A", "B", "C", "D"], "score": [9, 8, 7, 6], "industry": ["one", "one", "one", "two"]}
    )
    target = target_weights(frame, Strategy(top_n=4, score_threshold=0, max_weight=0.10, max_industry_weight=0.15))
    assert target == pytest.approx({"A": .10, "B": .05, "D": .10})


def test_window_masks_clear_observed_labels_around_each_boundary():
    dates = pd.bdate_range("2024-01-02", periods=150)
    frame = pd.DataFrame({"date": dates, "label_end": pd.Series(dates).shift(-6), "target": 1.0})
    windows = _windows(dates)
    assert [window.period for window in windows] == ["oos_1", "oos_2", "holdout"]
    for window in windows:
        train, valid, calibration = window.masks(frame)
        for mask, cutoff in zip((train, valid, calibration), (window.valid_start, window.calibration_start, window.evaluate[0]), strict=True):
            assert mask.any()
            assert frame.loc[mask, "label_end"].lt(cutoff).all()
            assert not (mask & frame["date"].isin(dates[dates < cutoff][-6:])).any()


def test_explicit_holdout_is_after_old_study_and_keeps_selection_windows_fixed():
    dates = pd.bdate_range("2025-10-01", "2026-10-30")
    windows = _windows(dates, "2026-08-01")
    shortened = _windows(dates[:-10], "2026-08-01")
    assert windows[-1].evaluate[0] == pd.Timestamp("2026-08-03")
    for full, shorter in zip(windows, shortened, strict=True):
        assert full.train_start == shorter.train_start
        assert full.valid_start == shorter.valid_start
        assert full.calibration_start == shorter.calibration_start
        if full.period != "holdout":
            pd.testing.assert_index_equal(full.evaluate, shorter.evaluate)
    with pytest.raises(ValueError, match="晚于上一轮"):
        _windows(dates, "2026-07-28")


def test_seqlstm_studies_share_purged_sample_and_report_classification_evidence(monkeypatch):
    dates = pd.bdate_range("2025-10-01", "2026-10-30")
    windows = _windows(dates, "2026-08-03")
    data = pd.concat([
        pd.DataFrame({
            "date": dates, "stock": f"{stock:06}.SH", "board": "sh_main", "exchange": "SH", "momentum": stock,
            "industry": "known" if stock < 5 else "unknown", "target": stock / 10,
            "label_end": pd.Series(dates).shift(-6), "signal_eligible": True,
        })
        for stock in range(8)
    ], ignore_index=True)
    config = {
        "mode": "seqlstm", "seed": 42, "lookback": 15, "max_train_samples": 65, "classification_path": "pit.csv",
        "epochs": 1, "device": "cpu",
    }
    cohort = _study_cohorts(data, windows, config)
    first = cohort[windows[0].period]
    studies = _studies(config, data.loc[first.train])
    assert {study.model for study in studies} == {"lightgbm", "seqlstm"}
    assert studies[0].static_columns == studies[-1].static_columns == ("exchange", "board", "industry")
    assert [study.static_columns for study in studies[1:3]] == [(), ("exchange", "board")]
    assert first.train.sum() == 65 < first.uncapped_train_rows
    for window in windows:
        sample = cohort[window.period]
        for mask, boundary in zip((sample.train, sample.valid, sample.calibration), (window.valid_start, window.calibration_start, window.evaluate[0]), strict=True):
            assert data.loc[mask, "label_end"].lt(boundary).all()
    requests = []

    def observe_fit(frame, *, train_mask, valid_mask, predict_mask, **kwargs):
        assert frame["date"].max() == windows[0].evaluate[-1]
        assert frame.loc[~(train_mask | valid_mask), "target"].isna().all()
        requests.append((frame.index[train_mask], frame.index[valid_mask], frame.index[predict_mask]))
        return SimpleNamespace(predictions=frame.loc[predict_mask, "momentum"], metadata={})

    monkeypatch.setattr("my_strategy.research_workbench.runner.fit_predict", observe_fit)
    for study in studies:
        _fold(SimpleNamespace(frame=data, feature_columns=["momentum"]), windows[0], study, config, cohort=first)
    for request in requests[1:]:
        for actual, expected in zip(request, requests[0], strict=True):
            pd.testing.assert_index_equal(actual, expected)
    study_rows = data.loc[data["date"].isin(windows[-1].evaluate)]
    evidence = _rank_by_classification(study_rows, study_rows["target"])
    assert evidence["board"][0]["rank_ic"] == pytest.approx(1)
    assert evidence["industry"][0]["rank_ic"] == pytest.approx(1)
    assert evidence["industry"][1]["rank_ic"] is None


def test_repairs_only_the_verified_legacy_tencent_benchmark_schema(tmp_path: Path):
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "date,open,close,source\n2026-04-01,110,100,tencent_cn_index\n2026-04-02,111,120,tencent_cn_index_ohlc_v2\n",
        encoding="utf-8",
    )
    result = _load_benchmark(path)
    assert list(result["open"]) == [100, 111]
    assert list(result["close"]) == [110, 120]
    assert result.attrs["legacy_tencent_repaired_rows"] == 1



def test_reference_amount_requires_same_date_matching_raw_ohlc(tmp_path: Path):
    dates = pd.bdate_range("2026-04-01", periods=3)
    bars = pd.DataFrame(
        {"date": dates, "stock": "600000.SH", "trade_open": 10.0, "trade_high": 10.5,
         "trade_low": 9.8, "trade_close": 10.25, "amount": [0.0, 0.0, 1500.0]}
    )
    reference = pd.DataFrame(
        {"date": dates, "stock": "600000.SH", "open": [10.0, 10.1, 10.0], "high": 10.5,
         "low": 9.8, "close": 10.25, "amount": [500, 700, 900]}
    )
    path = tmp_path / "raw.csv"
    reference.to_csv(path, index=False)
    result = _reference_amounts(bars, path)
    assert list(result["amount"]) == [500.0, 0.0, 1500.0]
    assert list(result["execution_reference_matched"]) == [True, False, True]
    assert list(result["amount_from_reference"]) == [True, False, False]
