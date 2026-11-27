"""Check domestic portfolio choice and historical evaluation isolation."""

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant import domestic_runner as domestic, tuning
from my_strategy.research_workbench.runner import Window


def _data() -> tuple[domestic.FrozenInputs, Window]:
    days = pd.bdate_range("2025-01-02", periods=84)
    stocks = ["600001.SH", "600002.SH", "600003.SH"]
    frame = pd.DataFrame({
        "date": days.repeat(len(stocks)), "stock": stocks * len(days), "signal_eligible": True,
        "trade_date": np.repeat(pd.Series(days).shift(-1).to_numpy(), len(stocks)),
        "label_end": np.repeat(pd.Series(days).shift(-6).to_numpy(), len(stocks)),
        "trade_open": 20 + np.repeat(np.arange(len(days)) * 0.08, len(stocks)),
        "trade_close": 20 + np.repeat(np.arange(len(days)) * 0.08, len(stocks)),
        "board": "sh_main", "industry": "unknown", "amount": 70_000_000.0,
    })
    rng = np.random.default_rng(87)
    for name in domestic.BASE_FEATURES:
        frame[name] = rng.normal(size=len(frame))
    frame["target"] = (0.02 + 0.01 * frame[domestic.BASE_FEATURES[0]]).where(frame["label_end"].notna())
    roster = pd.DataFrame({"delist_date": pd.NaT}, index=pd.Index(stocks, name="stock"))
    states = pd.DataFrame(columns=["date", "stock", "market_status", "source"])
    data = domestic.FrozenInputs(frame, days, pd.Series(0.0, index=days), states, roster, {})
    return data, Window("oos_test", days[0], days[36], days[50], days[66:])


def test_portfolio_selects_only_pretest_and_passed_risk_grid(monkeypatch):
    data, window = _data()
    sessions = domestic._selection_sessions(data.sessions, window)
    indices = data.frame.index[data.frame["date"].isin(sessions)]
    scores = pd.Series(0.02, index=indices)
    called = []

    def simulate(data, scores, actual_sessions, strategy, multiplier, run_id):
        assert actual_sessions.equals(sessions) and actual_sessions[-1] < window.evaluate[0]
        assert multiplier == 2
        key = (strategy.top_n, strategy.rebalance_days)
        called.append(key)
        return {"metrics": {
            "total_return": 0.30 if key == (5, 10) else 0.20 if key == (8, 5) else 0.05,
            "max_drawdown": -0.2 if key == (5, 10) else -0.08,
            "mean_daily_turnover": 0.25 if key == (10, 10) else 0.10,
        }}

    monkeypatch.setattr(domestic, "_simulate", simulate)
    winner, evidence = domestic._portfolio(data, window, scores, "synthetic")
    assert tuple(called) == domestic.PORTFOLIO_GRID
    assert (winner.top_n, winner.rebalance_days) == (8, 5)
    assert sum(not trial["risk_gate_pass"] for trial in evidence["candidates"]) == 2
    outside = data.frame.index[data.frame["date"].eq(window.evaluate[0])]
    with pytest.raises(ValueError, match="独立窗口"):
        domestic._portfolio(data, window, pd.Series(0.02, index=outside), "rejected")


def test_one_domestic_fold_writes_shared_predictions_and_distinct_ledgers(monkeypatch, tmp_path):
    pytest.importorskip("catboost")
    monkeypatch.setattr(tuning, "ITERATIONS", 16)
    data, window = _data()

    class Context:
        run_id = "synthetic"

        def subdir(self, *parts):
            folder = tmp_path.joinpath(*parts)
            folder.mkdir(parents=True, exist_ok=True)
            return folder

    report, returns = domestic._fold(data, window, Context(), 42)
    assert set(report["arms"]) == set(domestic.ARMS)
    assert report["arms"]["domestic_fixed"]["prediction"] == report["arms"]["domestic_tuned"]["prediction"]
    assert report["portfolio"]["selection_end"] < window.evaluate[0]
    for cost in domestic.COSTS:
        for arm in domestic.ARMS:
            assert returns[cost][arm].index.equals(window.evaluate)
            assert report["arms"][arm]["execution"][cost]["verification"]["next_session_verified"]
            ledger = tmp_path / "backtests" / window.period / f"{arm}_{cost}_orders.parquet"
            assert ledger.exists()
    predicted = pd.read_parquet(tmp_path / "backtests" / window.period / "domestic_predictions.parquet")
    assert predicted["date"].nunique() == len(window.evaluate)
    assert predicted.loc[predicted["scored"], "label_end"].le(window.evaluate[-1]).all()
