"""Maturity and sealed-order checks for the prospective evaluation entrypoint."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant import forward_accept
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, simulate
from my_strategy.latent_quant.models import ARMS


def _sealed_example():
    sessions = pd.bdate_range("2026-11-23", periods=12)
    panel = pd.DataFrame({
        "date": sessions.repeat(2), "stock": ["600000.SH", "600001.SH"] * len(sessions),
        "open": 10 + np.arange(len(sessions)).repeat(2) * 0.15,
        "trade_open": 20 + np.arange(len(sessions)).repeat(2) * 0.3,
        "trade_close": 20 + np.arange(len(sessions)).repeat(2) * 0.3,
        "has_trade_price": True, "volume": 10_000, "amount": 50_000_000.0,
        "board": "sh_main", "industry": "unknown", "signal_eligible": True,
    })
    market = pd.DataFrame({"date": [sessions[6]], "stock": ["600000.SH"], "market_status": ["suspended"], "source": ["synthetic"]})
    panel.loc[panel["date"].eq(sessions[6]) & panel["stock"].eq("600000.SH"), "signal_eligible"] = False
    benchmark = pd.DataFrame({"date": sessions, "open": 100.0, "close": 101.0})
    benchmark_returns = benchmark.set_index("date")["close"].pct_change()
    roster = pd.DataFrame({"delist_date": [pd.NaT] * 2}, index=pd.Index(panel["stock"].unique(), name="stock"))
    scores = pd.Series(0.01, index=panel.index).where(panel["signal_eligible"])
    prediction = panel.loc[panel["signal_eligible"], ["date", "stock"]].assign(score=0.01, q10=-0.04, q50=0.01, q90=0.08)
    predictions = pd.concat([prediction.assign(arm=arm) for arm in ARMS], ignore_index=True)
    cost_multipliers = {"normal": 1.0, "double": 2.0}
    config = {
        "initial_cash": forward_accept.INITIAL_CASH, "portfolio": asdict(Strategy()),
        "expense_model": asdict(StudyCostModel()), "cost_multipliers": cost_multipliers, "seed": 42,
    }
    orders = []
    for cost, multiplier in cost_multipliers.items():
        result = simulate(panel, scores, sessions, benchmark_returns, run_id="demo", cost_multiplier=multiplier,
                          market_status=market, roster=roster)
        orders.append(result["orders"].loc[result["orders"]["requested_shares"].gt(0), forward_accept._ORDER].assign(cost=cost))
    base_orders = pd.concat(orders, ignore_index=True)
    orders = pd.concat([base_orders.assign(arm=arm) for arm in ARMS], ignore_index=True)
    return dict(panel=panel, benchmark=benchmark, market_status=market, predictions=predictions, orders=orders), sessions, config, roster


def test_on_calendar_label_maturity_and_sealed_orders():
    tables, sessions, config, roster = _sealed_example()
    report, artifacts = forward_accept._evaluate(tables, sessions, config, roster, "test-forward-acceptance")
    predictions = artifacts["catboost_base_predictions.parquet"].set_index(["date", "stock"])
    assert np.isnan(predictions.loc[(sessions[0], "600000.SH"), "target"])
    assert np.isnan(predictions.loc[(sessions[5], "600000.SH"), "target"])
    row = predictions.loc[(sessions[0], "600001.SH")]
    assert row["trade_date"] == sessions[1] and row["label_end"] == sessions[6]
    assert row["target"] == pytest.approx(10.9 / 10.15 - 1)
    assert predictions.loc[pd.IndexSlice[sessions[-6]:, :], "target"].isna().all()
    for arm in ARMS:
        assert report["arms"][arm]["prediction"]["all_signal_rows"] == 23
        assert report["arms"][arm]["prediction"]["rows"] == 10
        for cost in config["cost_multipliers"]:
            assert report["arms"][arm]["execution"][cost]["sealed_orders_verified"]
    assert report["contrasts"]["catboost_base_vs_historical"]["normal"]["net_return_difference"] == 0

    tables["orders"].loc[0, "requested_shares"] += 100
    with pytest.raises(AssertionError):
        forward_accept._evaluate(tables, sessions, config, roster, "tampered")


def test_incomplete_gate_refuses_without_resolving_new_artifacts(monkeypatch):
    monkeypatch.setattr(forward_accept.forward, "status", lambda **kw: {
        "ready_for_scoring": False, "valid_daily_captures": 59, "target_sessions": 60, "status": "collecting",
    })
    monkeypatch.setattr(forward_accept, "artifact_run_dir", lambda *args, **kw: pytest.fail("must stop before resolving output"))
    with pytest.raises(ValueError, match="59/60"):
        forward_accept.accept(run_id="does-not-create-anything")
