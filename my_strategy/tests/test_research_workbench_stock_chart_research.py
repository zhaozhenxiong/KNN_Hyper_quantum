"""Contract tests for derived stock research layers and frozen forecasts."""

import copy

import pytest

from my_strategy.research_workbench import stock_chart_research as research


def bars(count=90):
    rows = []
    for index in range(count):
        close = 10 + index * .03 + (index % 7 - 3) * .12
        rows.append({"time": str(__import__("pandas").Timestamp("2026-01-02") + __import__("pandas").Timedelta(days=index))[:10],
                     "open": close - .04, "high": close + .22, "low": close - .25, "close": close,
                     "volume": 1000 + index * 17, "status": "tradable"})
    return rows


def payload(**changes):
    result = {"symbol": "000001.SZ", "bars": bars(), "context": {}, "warnings": []}
    return result | changes


def prediction(chart):
    last = next(row for row in reversed(chart["bars"]) if row["close"] is not None)
    samples = []
    for sample in range(8):
        path = []
        for horizon in range(5):
            close = last["close"] * (1 + .002 * (horizon + 1) + sample * .0001)
            path.append([close - .01, close + .03, close - .04, close, 1000 + sample, 10000 + sample])
        samples.append(path)
    forecast = []
    for horizon in range(5):
        closes = [path[horizon][3] for path in samples]
        forecast.append({"date": str(__import__("pandas").Timestamp(last["time"]) + __import__("pandas").Timedelta(days=horizon + 1))[:10],
                         "close": float(__import__("numpy").quantile(closes, .5)),
                         "close_p10": float(__import__("numpy").quantile(closes, .1)),
                         "close_p90": float(__import__("numpy").quantile(closes, .9)), "volume": 1003.5})
    return {"schema": "khquant_stock_prediction_v1", "status": "completed", "symbol": "000001.SZ", "as_of": last["time"],
            "input": {"symbol": "000001.SZ", "history_end": last["time"], "last_close": last["close"], "horizon": 5},
            "identity": {"future_dates": [row["date"] for row in forecast], "parameters": {"horizon": 5, "samples": 8}},
            "forecast": forecast, "samples": samples,
            "quality": {"prediction_usable": True, "valid_mask": [True] * 8, "requested_samples": 8,
                        "valid_samples": 8, "invalid_samples": 0, "valid_fraction": 1.0,
                        "minimum_valid_samples": 8, "minimum_valid_fraction": .8}}


def test_three_calendar_month_default_and_derived_layers(study=None):
    result = research.build_research(payload())
    assert result["status"] == "available"
    assert result["meta"]["months"] == 3
    assert result["bars"] == result["layers"]["frequencies"]["日线"]["bars"]
    assert result["layers"]["frequencies"]["周线"]["bars"][-1]["is_closed"] is False
    daily = result["layers"]["frequencies"]["日线"]
    assert daily["indicators"]["rows"][-1]["ma20"] is not None
    assert daily["indicators"]["rows"][20]["volume_mean20_prior"] > 0
    assert daily["indicators"]["chip_profiles"]
    assert result["prediction"]["status"] == "not_executed"
    assert result["prediction"]["forecast"] == []


def test_duplicate_invalid_and_unclosed_history_never_becomes_chart_fact():
    duplicated = payload()
    duplicated["bars"][10] = copy.deepcopy(duplicated["bars"][9])
    with pytest.raises(ValueError, match="严格递增"):
        research.build_research(duplicated)
    invalid = payload()
    invalid["bars"][0]["high"] = 1
    with pytest.raises(ValueError, match="OHLC"):
        research.build_research(invalid)
    unclosed = payload()
    unclosed["bars"][-1]["is_closed"] = False
    with pytest.raises(ValueError, match="真实闭合"):
        research.build_research(unclosed)


def test_nontrading_slot_is_not_an_observed_candle():
    chart = payload()
    halted = chart["bars"][30]["time"]
    chart["bars"][30] = {"time": halted, "open": None, "high": None, "low": None,
                         "close": None, "volume": None, "status": "suspended"}
    result = research.build_research(chart)
    assert result["meta"]["closed_bars"] == 89
    assert result["meta"]["calendar_slots"] == 90
    assert result["bars"][30]["status"] == "suspended"
    assert all(row["time"] != halted for row in result["layers"]["frequencies"]["日线"]["indicators"]["rows"])


def test_pass_fail_unavailable_and_not_executed_remain_distinct():
    chart = payload()
    frozen = prediction(chart)
    passed = research.build_research(chart, frozen)["prediction"]
    assert passed["status"] == "pass" and len(passed["forecast"]) == 5
    assert passed["validation"]["future_sessions_fabricated"] is False
    failed_payload = copy.deepcopy(frozen)
    failed_payload["quality"]["prediction_usable"] = False
    failed = research.build_research(chart, failed_payload)["prediction"]
    assert failed["status"] == "fail"
    assert len(failed["forecast"]) == 5 and len(failed["samples"]) == 8
    unavailable = research.build_research(chart, {"symbol": "000001.SZ", "status": "unavailable", "reason": "模型缺失"})["prediction"]
    assert unavailable["status"] == "unavailable" and unavailable["forecast"] == []
    not_executed = research.build_research(chart)["prediction"]
    assert not_executed["status"] == "not_executed" and not_executed["forecast"] == []


def test_prediction_cannot_invent_dates_change_cutoff_or_contradict_samples():
    chart = payload()
    wrong_date = prediction(chart)
    wrong_date["identity"]["future_dates"] = []
    with pytest.raises(ValueError, match="交易日历"):
        research.build_research(chart, wrong_date)
    wrong_close = prediction(chart)
    wrong_close["input"]["last_close"] += 1
    with pytest.raises(ValueError, match="末收盘"):
        research.build_research(chart, wrong_close)
    wrong_quantile = prediction(chart)
    wrong_quantile["forecast"][0]["close_p90"] += 1
    with pytest.raises(ValueError, match="分位数"):
        research.build_research(chart, wrong_quantile)
    wrong_fraction = prediction(chart)
    wrong_fraction["quality"]["valid_fraction"] = .5
    with pytest.raises(ValueError, match="有效率"):
        research.build_research(chart, wrong_fraction)
    wrong_mask = prediction(chart)
    wrong_mask["samples"][0][0][1] = 0
    with pytest.raises(ValueError, match="样本有效性"):
        research.build_research(chart, wrong_mask)


def test_attach_preserves_original_ledger_and_marks_bad_research_unavailable():
    chart = payload(strategies=[{"events": [{"stage": "fill", "price": 12.34}]}])
    original = copy.deepcopy(chart)
    result = research.attach(chart)
    assert result["strategies"] == original["strategies"]
    assert result["bars"] == original["bars"]
    assert result["research"]["status"] == "available"
    broken = payload(bars=[])
    result = research.attach(broken)
    assert result["research"]["status"] == "unavailable"
    assert result["research"]["prediction"]["status"] == "unavailable"


def test_invalid_prediction_keeps_derived_history_visible():
    chart = payload()
    broken = prediction(chart)
    broken["identity"]["future_dates"] = []
    chart["research_prediction"] = broken
    result = research.attach(chart)
    assert result["research"]["status"] == "available"
    assert result["research"]["prediction"]["status"] == "unavailable"
    assert len(result["research"]["layers"]["frequencies"]["日线"]["bars"]) == 90
    assert result["research"]["layers"]["frequencies"]["日线"]["pens"]
    assert any("预测" in warning for warning in result["warnings"])
