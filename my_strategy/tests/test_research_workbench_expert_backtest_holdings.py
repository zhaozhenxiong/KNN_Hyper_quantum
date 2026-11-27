"""Holding returns reconcile to the authorized replay without re-charging execution."""

import json

import pandas as pd
import pytest

from my_strategy.research_workbench import expert_api, expert_backtest_holdings as api, expert_replay, latent_api


@pytest.fixture
def archived_holdings(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_api, "RUNS", tmp_path)
    run = "test-holdings"
    folder = tmp_path / run
    table = folder / "backtests/custom"
    table.mkdir(parents=True)
    (folder / "reports").mkdir()
    (folder / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    days = [f"2025-01-{day:02}" for day in (2, 3, 6, 7, 8, 9)]
    report = {
        "mode": "expert_watchlist", "run": {"run_id": run, "task": expert_replay.TASK},
        "source": {"cost": "normal", "approval": "research_only", "source_run_id": "source-research",
                   "model_version": "test-model", "requested_model_sessions": days},
        "coverage": {"requested_start": "2025-01-02", "requested_end": "2025-01-09"},
        "accounts": {"full_pool": {"downloads": {"positions": "positions", "trades": "trades", "daily": "daily"}},
                     "watchlist": {"downloads": {}}},
    }
    (folder / f"reports/{expert_replay.REPORT}").write_text(json.dumps(report), encoding="utf-8")
    daily = pd.DataFrame({"date": days, "market_value": [0., 0., 0., 0., 0., 4500.],
                          "listed_market_value": [0., 0., 0., 0., 0., 4500.],
                          "stranded_delisted_value": [0., 0., 0., 0., 0., 0.]})
    positions = pd.DataFrame({"date": ["2025-01-09"], "stock": ["000001.SZ"], "board": ["main"],
                              "industry": ["测试行业"], "shares": [300], "mark_price": [15.],
                              "mark_date": ["2025-01-09"], "market_value": [4500.], "market_status": ["listed"],
                              "position_state": ["listed"], "detached_on": [None], "mark_is_stale": [False]})
    trades = pd.DataFrame([
        ("2025-01-02", -1010., 10.10, 100, .10),
        ("2025-01-03", -2220., 3230. / 300, 300, .20),
        ("2025-01-06", 3590., 0., 0, 0.),
        ("2025-01-07", -2410., 12.05, 200, .20),
        ("2025-01-08", 1290., 12.05, 100, .10),
        ("2025-01-09", -2810., 4015. / 300, 300, .30),
    ], columns=["date", "cash_flow", "holding_cost_after", "position_after", "target_weight"])
    trades["symbol"] = "000001.SZ"
    for kind, frame in (("daily", daily), ("positions", positions), ("trades", trades)):
        frame.to_parquet(table / f"full_pool_normal_{kind}.parquet", index=False)
    return run, folder


def _positions(folder):
    return pd.read_parquet(folder / "backtests/custom/full_pool_normal_positions.parquet")


def _daily(folder):
    return pd.read_parquet(folder / "backtests/custom/full_pool_normal_daily.parquet")


def test_realized_and_unrealized_returns_include_partial_sales_fees_and_reentry(archived_holdings):
    run, _ = archived_holdings
    result = api.holdings(run)
    assert result["as_of"] == "2025-01-09"
    assert result["context"] == {"run_id": run, "source_run_id": "source-research", "model_version": "test-model",
                                "model_id": None, "approval": "research_only"}
    row = result["holdings"][0]
    assert row["cost_per_share"] == pytest.approx(4015. / 300)
    assert row["cost_basis"] == pytest.approx(4015.)
    assert row["market_value"] == 4500.
    assert row["unrealized_pnl"] == pytest.approx(485.)
    assert row["unrealized_return"] == pytest.approx(485. / 4015.)
    assert row["realized_pnl"] == pytest.approx(445.)
    assert row["total_pnl"] == pytest.approx(930.)
    assert row["total_pnl"] == pytest.approx(-1010. - 2220. + 3590. - 2410. + 1290. - 2810. + 4500.)
    assert row["zero_recovery_pnl"] is None
    assert row["last_trade_date"] == "2025-01-09" and row["last_target_weight"] == .30
    assert result["summary"] == {"positions": 1, "listed_positions": 1, "stranded_delisted_positions": 0,
                                "cost_basis": pytest.approx(4015.), "market_value": 4500.,
                                "unrealized_pnl": pytest.approx(485.), "unrealized_return": pytest.approx(485. / 4015.),
                                "realized_pnl": pytest.approx(445.), "total_pnl": pytest.approx(930.),
                                "zero_recovery_pnl": pytest.approx(930.), "status": "verified"}
    assert result["reconciliation"]["passed"] is True
    assert all(value == 0 for value in result["reconciliation"]["differences"].values())
    assert [item["code"] for item in result["advice"]] == ["historical_actions"]


def test_stranded_holding_has_stale_mark_and_zero_recovery_scenario(archived_holdings):
    run, folder = archived_holdings
    positions = _positions(folder)
    positions.loc[0, ["mark_price", "market_value", "market_status", "position_state", "detached_on", "mark_is_stale"]] = [
        5., 1500., "delisted", "stranded_delisted", "2025-01-08", True,
    ]
    positions.to_parquet(folder / "backtests/custom/full_pool_normal_positions.parquet", index=False)
    daily = _daily(folder)
    daily.loc[daily.index[-1], ["market_value", "listed_market_value", "stranded_delisted_value"]] = [1500., 0., 1500.]
    daily.to_parquet(folder / "backtests/custom/full_pool_normal_daily.parquet", index=False)

    result = api.holdings(run)
    row = result["holdings"][0]
    assert row["stranded_delisted"] is True and row["mark_is_stale"] is True
    assert row["mark_date"] == "2025-01-09" and row["detached_on"] == "2025-01-08"
    assert row["unrealized_pnl"] == pytest.approx(-2515.)
    assert row["realized_pnl"] == pytest.approx(445.)
    assert row["total_pnl"] == pytest.approx(-2070.)
    assert row["zero_recovery_pnl"] == pytest.approx(-3570.)
    assert result["summary"]["zero_recovery_pnl"] == pytest.approx(-3570.)
    assert result["reconciliation"]["passed"] is True
    assert [item["code"] for item in result["advice"]] == ["stranded_valuation", "no_listed_positions"]


def test_empty_positions_are_not_confused_with_missing_ledger(archived_holdings):
    run, folder = archived_holdings
    positions = _positions(folder).iloc[0:0]
    positions.to_parquet(folder / "backtests/custom/full_pool_normal_positions.parquet", index=False)
    daily = _daily(folder)
    daily.loc[daily.index[-1], ["market_value", "listed_market_value", "stranded_delisted_value"]] = 0.
    daily.to_parquet(folder / "backtests/custom/full_pool_normal_daily.parquet", index=False)

    result = api.holdings(run)
    assert result["holdings"] == []
    assert result["summary"]["positions"] == 0 and result["summary"]["status"] == "verified"
    assert result["summary"]["market_value"] == 0 and result["reconciliation"]["passed"] is True
    assert [item["code"] for item in result["advice"]] == ["no_listed_positions"]
    missing = api.holdings(run, "account=watchlist")
    assert missing["summary"]["status"] == "unavailable"
    assert missing["advice"][0]["code"] == "missing_ledgers"


@pytest.mark.parametrize(("field", "value"), [("holding_cost_after", None), ("position_after", 301)])
def test_missing_cost_or_share_mismatch_degrades_instead_of_filling_zero(archived_holdings, field, value):
    run, folder = archived_holdings
    path = folder / "backtests/custom/full_pool_normal_trades.parquet"
    trades = pd.read_parquet(path)
    trades.loc[trades.index[-1], field] = value
    trades.to_parquet(path, index=False)

    result = api.holdings(run)
    row = result["holdings"][0]
    assert row["status"] == "unavailable" and row["reason"]
    assert row["cost_basis"] is None and row["realized_pnl"] is None and row["total_pnl"] is None
    assert result["summary"]["status"] == "unavailable"
    assert result["summary"]["cost_basis"] is None and result["summary"]["total_pnl"] is None
    assert result["reconciliation"]["passed"] is True
    assert result["advice"][-1]["code"] == "verification_failed"


def test_market_reconciliation_failure_marks_summary_unavailable(archived_holdings):
    run, folder = archived_holdings
    daily = _daily(folder)
    daily.loc[daily.index[-1], "market_value"] = 4501.
    daily.to_parquet(folder / "backtests/custom/full_pool_normal_daily.parquet", index=False)

    result = api.holdings(run)
    assert result["holdings"][0]["status"] == "verified"
    assert result["summary"]["status"] == "unavailable"
    assert result["reconciliation"]["passed"] is False
    assert result["reconciliation"]["differences"]["market_value"] == -1
    assert result["advice"][-1]["code"] == "verification_failed"


@pytest.mark.parametrize("query", ["account=benchmark", "account=full_pool&account=watchlist", "after_last=1"])
def test_invalid_holdings_filters_are_rejected_before_run_lookup(query):
    with pytest.raises(latent_api.RouteError) as error:
        api.holdings("not-a-run", query)
    assert error.value.status == 400


def test_http_holdings_route_uses_the_authorized_account(archived_holdings):
    run, _ = archived_holdings

    class Handler:
        def _json(self, status, data):
            self.result = (status, data)

    handler = Handler()
    assert expert_api.handle_get(handler, f"/api/expert/replays/{run}/holdings", "account=full_pool")
    status, result = handler.result
    assert status == 200 and result["summary"]["total_pnl"] == pytest.approx(930.)


def test_current_replay_holding_returns_reconcile_to_real_ledgers():
    run = "20261124-212836-758317-20261124-expert-replay-19f8ef2"
    if not (latent_api.RUNS / run / "status.json").is_file():
        pytest.skip("requires the local completed replay")
    result = api.holdings(run)
    assert result["as_of"] == "2026-11-20"
    assert result["summary"]["status"] == "verified"
    assert result["reconciliation"]["passed"] is True
    assert all(value == 0 for value in result["reconciliation"]["differences"].values())
    assert result["summary"]["positions"] == 3
    assert result["summary"]["listed_positions"] == 0
    assert result["summary"]["stranded_delisted_positions"] == 3
    assert result["summary"]["cost_basis"] == pytest.approx(306275.423832575)
    assert result["summary"]["market_value"] == 105650.
    assert result["summary"]["unrealized_pnl"] == pytest.approx(-200625.423832575)
    assert result["summary"]["unrealized_return"] == pytest.approx(-.6550490448174077)
    assert result["summary"]["realized_pnl"] == pytest.approx(-574.3309843050083)
    assert result["summary"]["total_pnl"] == pytest.approx(-201199.75481687995)
    assert result["summary"]["zero_recovery_pnl"] == pytest.approx(-306849.75481688)
    rows = {row["stock"]: row for row in result["holdings"]}
    assert set(rows) == {"600070.SH", "600811.SH", "603388.SH"}
    assert rows["600070.SH"]["shares"] == 104300 and rows["600070.SH"]["market_value"] == 43806.
    assert rows["600070.SH"]["realized_pnl"] == 0 and rows["600070.SH"]["mark_date"] == "2025-04-10"
    assert rows["600811.SH"]["shares"] == 86400 and rows["600811.SH"]["market_value"] == 31104.
    assert rows["600811.SH"]["realized_pnl"] == 0 and rows["600811.SH"]["mark_date"] == "2025-04-14"
    assert rows["603388.SH"]["shares"] == 53000 and rows["603388.SH"]["market_value"] == pytest.approx(30740.)
    assert rows["603388.SH"]["realized_pnl"] == pytest.approx(-574.3309843050083)
    assert rows["603388.SH"]["total_pnl"] == pytest.approx(-70615.913592575)
    assert rows["603388.SH"]["mark_date"] == "2025-11-10"
    assert all(row["stranded_delisted"] for row in rows.values())
    assert [item["code"] for item in result["advice"]] == ["stranded_valuation", "no_listed_positions"]
