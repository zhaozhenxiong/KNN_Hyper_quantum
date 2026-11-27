import pandas as pd
import pytest

from my_strategy.latent_quant.execution import Strategy, StudyCostModel, simulate
from my_strategy.latent_quant.market_evidence import execution_evidence_identity, load_market_evidence


def _quotes(days: pd.DatetimeIndex, stocks: list[str]) -> pd.DataFrame:
    frame = pd.MultiIndex.from_product([days, stocks], names=["date", "stock"]).to_frame(index=False)
    return frame.assign(
        trade_open=10.0, trade_close=10.0, has_trade_price=True, amount=20_000_000.0, volume=2_000_000.0,
        board="sh_main", industry="technology", signal_eligible=True,
    )


def _simulate(frame, days, scores, **kwargs):
    return simulate(
        frame, pd.Series(scores, index=frame.index), days, pd.Series(0.0, index=days),
        run_id="synthetic", **kwargs,
    )


def test_prior_close_fixes_order_shares_and_fold_boundary():
    dates = pd.bdate_range("2026-04-01", periods=5)
    days = dates[1:4]
    frame = _quotes(dates, ["600000.SH"])
    scores = [1, 1, 1, 1, 1]
    base = _simulate(frame, days, scores)
    frame.loc[frame["date"].eq(days[1]), "trade_open"] = 10.50
    changed = _simulate(frame, days, scores)

    base_order = base["orders"].iloc[0]
    changed_order = changed["orders"].iloc[0]
    assert len(changed["orders"]) == 1
    assert changed_order["signal_date"] == str(days[0].date())
    assert changed_order["date"] == str(days[1].date())
    assert base_order["requested_shares"] == changed_order["requested_shares"] == 9500
    assert changed_order["filled_shares"] == 9500
    assert changed_order["execution_price"] == pytest.approx(10.50 * 1.0005)
    pd.testing.assert_series_equal(base["daily"].iloc[0], changed["daily"].iloc[0])


def test_account_is_reconstructable_from_positions_and_trades_under_stress():
    days = pd.bdate_range("2026-04-01", periods=4)
    frame = _quotes(days, ["600000.SH", "600001.SH"])
    frame["trade_close"] = [10, 10, 10.05, 10.10, 10.12, 10.30, 10.22, 10.52]
    scores = [1, 2, 1, 2, 1, 2, 1, 2]
    base = _simulate(frame, days, scores)
    stress = _simulate(frame, days, scores, cost_multiplier=2)
    assert len(base["trades"]) == len(stress["trades"]) == 2
    assert base["trades"]["fee"].sum() < stress["trades"]["fee"].sum()
    assert base["metrics"]["total_return"] > stress["metrics"]["total_return"]
    for result in (base, stress):
        trades = result["trades"]
        assert trades["transfer_fee_paid"].to_numpy() == pytest.approx(
            (trades["price"] * trades["shares"] * 0.00001).to_numpy(),
        )
        day_ledger = result["daily"].set_index("date")
        flows = result["trades"].groupby("date")["cash_flow"].sum().reindex(day_ledger.index, fill_value=0)
        by_position = result["positions"].assign(
            recalc=lambda table: table["shares"] * table["mark_price"],
        )
        values = by_position.groupby("date")["recalc"].sum().reindex(day_ledger.index, fill_value=0)
        cash = 1_000_000 + flows.cumsum()
        assert cash.to_numpy() == pytest.approx(day_ledger["cash"].to_numpy())
        assert (cash + values).to_numpy() == pytest.approx(day_ledger["equity"].to_numpy())
        assert (cash.iloc[-1] + values.iloc[-1]) / 1_000_000 - 1 == pytest.approx(result["metrics"]["total_return"])
        assert (1 + day_ledger["net_return"]).prod() - 1 == pytest.approx(result["metrics"]["total_return"])


def test_rejection_provenance_and_unknown_open_proxy_are_distinguished():
    days = pd.bdate_range("2026-04-01", periods=2)
    stocks = ["600001.SH", "600002.SH", "600003.SH", "600004.SH", "600005.SH", "600006.SH"]
    frame = _quotes(days, stocks)
    frame.loc[frame["date"].eq(days[0]) & frame["stock"].eq(stocks[4]), "amount"] = 0
    frame.loc[frame["date"].eq(days[1]) & frame["stock"].eq(stocks[1]), "trade_open"] = float("nan")
    frame.loc[frame["date"].eq(days[1]) & frame["stock"].eq(stocks[2]), "trade_open"] = 11.0
    frame.loc[frame["date"].eq(days[1]) & frame["stock"].eq(stocks[5]), "volume"] = 0
    statuses = pd.DataFrame(
        {"date": [days[1]], "stock": [stocks[0]], "market_status": ["suspended"], "source": ["historical_status"]},
    )
    result = _simulate(
        frame, days, [1] * len(frame), market_status=statuses,
        strategy=Strategy(top_n=6, max_industry_weight=0.60, max_board_weight=0.60),
    )
    orders = result["orders"].set_index("stock")
    assert orders["reason"].to_dict() == dict(zip(
        stocks, ["suspended", "no_open", "approximate_price_limit", "", "no_previous_session_liquidity", "zero_volume_bar"],
        strict=True,
    ))
    assert orders.loc[stocks[0], "market_status_source"] == "historical_status"
    assert orders.loc[stocks[3], "market_status"] == "unknown"
    assert result["metrics"]["fills_with_unknown_market_status"] == 1
    assert result["metrics"]["reject_rate"] == pytest.approx(4 / 5)
    assert result["metrics"]["pretrade_rejected_count"] == 1
    candidates = result["selection_candidates"].set_index("stock")
    assert candidates.loc[stocks[4], "order_count"].eq(0).all()
    assert result["sector_candidate_daily"]["order_count"].sum() == result["metrics"]["order_count"]


def test_sells_follow_nonpositive_signal_and_delisted_holdings_keep_last_mark():
    days = pd.bdate_range("2026-04-01", periods=4)
    stock = "600000.SH"
    frame = _quotes(days, [stock])
    base = _simulate(frame, days, [1, 0, -1, -1], strategy=Strategy(rebalance_days=1))
    stress = _simulate(frame, days, [1, 0, -1, -1], strategy=Strategy(rebalance_days=1), cost_multiplier=2)
    assert base["trades"]["action"].tolist() == ["BUY", "SELL"]
    assert base["trades"].iloc[1]["fee"] / (
        base["trades"].iloc[1]["shares"] * base["trades"].iloc[1]["price"]
    ) == pytest.approx(0.00081)
    stress_sale = stress["trades"].iloc[1]
    assert stress_sale["fee"] / (stress_sale["shares"] * stress_sale["price"]) == pytest.approx(0.00111)
    frame = frame.loc[frame["date"].lt(days[2])]
    roster = pd.DataFrame({"delist_date": [days[2]]}, index=pd.Index([stock], name="stock"))
    delisted = _simulate(frame, days, [1, 0], strategy=Strategy(rebalance_days=1), roster=roster)
    assert delisted["orders"]["reason"].tolist() == ["", "delisted"]
    last = delisted["positions"].iloc[-1]
    assert last["mark_date"] == str(days[1].date())
    assert last["market_status"] == "delisted"
    assert delisted["metrics"]["delisted_positions_at_end"] == 1


def test_cash_buffer_limits_fill_after_open_without_changing_requested_shares():
    days = pd.bdate_range("2026-04-01", periods=2)
    frame = _quotes(days, ["600000.SH"])
    frame.loc[1, "trade_open"] = 10.90
    result = _simulate(
        frame, days, [1, 1], initial_cash=100_000,
        strategy=Strategy(top_n=1, max_weight=1, max_industry_weight=1, max_board_weight=1),
    )
    order = result["orders"].iloc[0]
    assert order["requested_shares"] == 9500
    assert order["filled_shares"] == 8700
    assert order["status"] == "partial"
    assert order["reason"] == "cash_buffer"
    assert result["metrics"]["reject_rate"] == 0
    assert result["daily"].iloc[-1]["cash"] >= 5000


def test_cross_sectional_rank_is_explicit_and_absolute_zero_orders_are_auditable():
    days = pd.bdate_range("2026-04-01", periods=2)
    frame = _quotes(days, ["600000.SH", "600001.SH"])
    scores = [-0.03, -0.01, -0.03, -0.01]
    absolute = _simulate(frame, days, scores, strategy=Strategy(rebalance_days=1))
    assert absolute["orders"].empty and absolute["trades"].empty
    first = absolute["selection_daily"].iloc[0]
    assert first["selection_reason"] == "NO_QUALIFIED_CANDIDATES"
    assert first["eligible_rows"] == 2 and first["finite_scores"] == 2
    assert first["qualified_scores"] == 0 and first["target_count"] == 0 and first["intent_count"] == 0
    assert absolute["metrics"]["planned_rebalance_count"] == 1
    assert absolute["metrics"]["no_candidate_rebalances"] == 1

    ranked = _simulate(frame, days, scores, strategy=Strategy(
        rebalance_days=1, top_n=1, selection_mode="cross_sectional_rank", min_score=None,
    ))
    assert len(ranked["trades"]) == 1
    assert ranked["orders"].iloc[0]["stock"] == "600001.SH"
    ranked_first = ranked["selection_daily"].iloc[0]
    assert ranked_first["selection_mode"] == "cross_sectional_rank"
    assert pd.isna(ranked_first["score_threshold"])
    assert ranked_first["qualified_scores"] == 2
    assert ranked_first["target_count"] == 1 and ranked_first["intent_count"] == 1

    with pytest.raises(ValueError, match="min_score=None"):
        Strategy(selection_mode="cross_sectional_rank", min_score=0)
    with pytest.raises(ValueError, match="finite minimum score"):
        Strategy(selection_mode="absolute_q50", min_score=None)


def test_sector_candidate_ledger_separates_global_top_n_from_targets_and_execution():
    days = pd.bdate_range("2026-04-01", periods=3)
    stocks = ["600000.SH", "600001.SH", "600002.SH"]
    frame = _quotes(days, stocks)
    frame["industry"] = ["banks", "technology", "technology"] * len(days)
    frame.loc[frame["stock"].eq("600002.SH"), "industry"] = "unknown"
    scores = [0.01, 0.03, 0.02] * len(days)
    result = _simulate(frame, days, scores, strategy=Strategy(
        rebalance_days=2, top_n=1, selection_mode="cross_sectional_rank", min_score=None,
    ))
    ledger = result["selection_candidates"]
    first = ledger.loc[ledger["date"].eq(str(days[0].date()))].set_index("stock")
    assert first["qualified"].all()
    assert first["global_rank"].astype(int).to_dict() == {"600000.SH": 3, "600001.SH": 1, "600002.SH": 2}
    assert first["purchase_candidate"].to_dict() == {"600000.SH": False, "600001.SH": True, "600002.SH": False}
    assert first.loc["600001.SH", "target_weight"] > 0
    assert first.loc["600001.SH", "intent_side"] == "BUY"
    assert first.loc["600001.SH", "order_count"] == first.loc["600001.SH", "fill_count"] == 1
    later = ledger.loc[ledger["date"].eq(str(days[1].date()))]
    assert not later["purchase_candidate"].any()
    assert later["selection_reason"].eq("NOT_REBALANCE_DAY").all()
    sector = result["sector_candidate_daily"].set_index(["date", "industry"])
    technology = sector.loc[(str(days[0].date()), "technology")]
    assert technology["qualified_count"] == 1
    assert technology["purchase_candidate_count"] == technology["target_count"] == 1
    assert sector.loc[(str(days[0].date()), "unknown"), "qualified_count"] == 1
    totals = result["sector_candidate_daily"].groupby("date")[["qualified_count", "purchase_candidate_count", "target_count", "order_count", "fill_count"]].sum()
    assert totals.loc[str(days[0].date()), "purchase_candidate_count"] == 1
    assert totals["order_count"].sum() == len(result["orders"])
    assert totals["fill_count"].sum() == len(result["trades"])


def test_absolute_mode_has_no_purchase_candidates_when_no_score_qualifies():
    days = pd.bdate_range("2026-04-01", periods=2)
    frame = _quotes(days, ["600000.SH", "600001.SH"])
    result = _simulate(frame, days, [-0.03, -0.01, -0.03, -0.01], strategy=Strategy(rebalance_days=1))
    ledger = result["selection_candidates"]
    assert len(ledger.loc[ledger["date"].eq(str(days[0].date()))]) == 2
    assert not ledger["qualified"].any()
    assert not ledger["purchase_candidate"].any()
    assert result["sector_candidate_daily"]["purchase_candidate_count"].sum() == 0


def test_defensive_exposure_is_separate_from_score_selection_and_can_exit():
    days = pd.bdate_range("2026-04-01", periods=4)
    frame = _quotes(days, ["600000.SH", "600001.SH"])
    scores = [-0.03, -0.01] * len(days)
    strategy = Strategy(rebalance_days=1, top_n=1, selection_mode="cross_sectional_rank", min_score=None)
    exposure = pd.Series([1.0, 0.0, 0.5, 1.0], index=days)
    result = _simulate(frame, days, scores, strategy=strategy, target_exposure=exposure)
    assert result["selection_daily"]["target_exposure"].tolist() == [1.0, 0.0, 0.5, 1.0]
    assert result["selection_daily"]["selection_reason"].iloc[1] == "DEFENSIVE_CASH"
    assert result["selection_daily"]["target_count"].tolist() == [1, 0, 1, 0]
    assert result["trades"]["action"].tolist() == ["BUY", "SELL", "BUY"]
    assert result["metrics"]["defensive_cash_rebalances"] == 1
    absolute = _simulate(frame, days, scores, target_exposure=exposure)
    assert absolute["orders"].empty
    assert absolute["selection_daily"]["selection_reason"].iloc[0] == "NO_QUALIFIED_CANDIDATES"
    with pytest.raises(ValueError, match="between zero and one"):
        _simulate(frame, days, scores, strategy=strategy, target_exposure=exposure.drop(days[0]))



    days = pd.bdate_range("2026-04-01", periods=3)
    frame = _quotes(days, ["600000.SH", "600001.SH"])
    frame.loc[4, "limit_down"] = True
    strategy = Strategy(top_n=1, rebalance_days=1)
    result = _simulate(frame, days, [1, 0, 0, 1, 0, 1], strategy=strategy)
    assert result["orders"]["reason"].tolist() == ["", "price_limit", "position_limit"]
    assert result["metrics"]["positions_at_end"] == 1
    inactive = _simulate(frame, days, [0] * len(frame))
    assert inactive["trades"].empty and "cash_flow" in inactive["trades"]
    assert inactive["orders"].empty and "requested_shares" in inactive["orders"]
    assert inactive["positions"].empty and "market_value" in inactive["positions"]


def test_transfer_fee_applies_with_minimum_commission_and_reduces_affordable_shares():
    cost = StudyCostModel()
    assert cost.fees("BUY", 1000) == pytest.approx(5.01)
    assert cost.fees("SELL", 1000) == pytest.approx(5.51)
    assert cost.affordable_buy_shares(1005.00, 10.0) == 99
    assert cost.affordable_buy_shares(1005.02, 10.0) == 100


def test_custom_cost_ledger_breaks_out_stamp_tax_for_partial_position_sales():
    days = pd.bdate_range("2026-04-01", periods=4)
    frame = _quotes(days, ["600000.SH"])
    frame.loc[frame["date"].ge(days[1]), "trade_close"] = 20.0
    frame.loc[frame["date"].ge(days[2]), "trade_open"] = 20.0
    result = _simulate(frame, days, [1, 1, 0, 0], strategy=Strategy(rebalance_days=1),
                       base_cost_model=StudyCostModel(commission=0.0002))
    trades = result["trades"]
    assert trades["action"].tolist() == ["BUY", "REDUCE", "SELL"]
    assert trades["stamp_tax_paid"].iloc[1] == pytest.approx(trades["price"].iloc[1] * trades["shares"].iloc[1] * 0.0005)
    assert trades["fee"].to_numpy() == pytest.approx(
        (trades["commission_paid"] + trades["stamp_tax_paid"] + trades["transfer_fee_paid"]).to_numpy(),
    )


def test_actual_603388_adverse_open_uses_dated_five_percent_limit():
    days = pd.bdate_range("2025-11-03", periods=3)
    frame = _quotes(days, ["603388.SH"]).assign(trade_open=[0.74, 0.70, 0.67], trade_close=[0.74, 0.70, 0.67],
                                               prev_trade_close=[0.78, 0.74, 0.70])
    result = _simulate(frame, days, [1, 0, 0], strategy=Strategy(rebalance_days=1), cost_multiplier=2)
    sale = result["orders"].iloc[-1]
    assert sale["side"] == "SELL" and sale["open"] == sale["limit_lower"] == 0.67
    assert sale["limit_pct_used"] == 0.05 and sale["preclose_source"] == "previous_raw_close_proxy"
    assert sale["risk_warning_status"] == "confirmed"
    assert sale["risk_warning_effective_from"] == "2025-05-06"
    assert sale["limit_basis"] == "verified_risk_warning_limit"
    assert sale["reason"] == "price_limit" and sale["filled_shares"] == 0
    assert result["trades"]["action"].tolist() == ["BUY"]


def test_unknown_five_percent_open_is_audited_without_claiming_st_state():
    days = pd.bdate_range("2025-11-03", periods=2)
    frame = _quotes(days, ["600000.SH"]).assign(trade_open=[10.0, 10.5])
    result = _simulate(frame, days, [1, 1])
    order = result["orders"].iloc[-1]
    assert order["limit_basis"] == "approximate_board_limit" and order["limit_pct_used"] == 0.10
    assert order["risk_warning_status"] == "unknown" and order["possible_risk_warning_boundary"]
    assert result["metrics"]["fills_at_possible_risk_warning_boundary"] == 1


def test_modeled_slippage_clamps_to_band_and_reconciles_separately_from_fees():
    days = pd.bdate_range("2025-04-07", periods=2)
    frame = _quotes(days, ["600070.SH"]).assign(trade_open=[1.0, 1.04], trade_close=1.0, prev_trade_close=1.0)
    result = _simulate(frame, days, [1, 1], cost_multiplier=2, base_cost_model=StudyCostModel(slippage=0.03))
    fill = result["trades"].iloc[-1]
    assert fill["open"] == 1.04 and fill["price"] == fill["limit_upper"] == 1.05
    assert fill["unclamped_execution_price"] == pytest.approx(1.04 * 1.06)
    assert fill["slippage_clamped"] and result["metrics"]["slippage_clamp_count"] == 1
    assert fill["modeled_slippage_paid"] == pytest.approx(0.01 * fill["shares"])
    assert fill["fee"] == pytest.approx(fill["commission_paid"] + fill["stamp_tax_paid"] + fill["transfer_fee_paid"])


def test_600070_halt_and_delist_only_detach_holdings_once():
    days = pd.DatetimeIndex(["2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-14",
                            "2025-04-29", "2025-04-30", "2025-05-06", "2025-05-07"])
    stock = "600070.SH"
    frame = _quotes(days, [stock, "600000.SH"])
    held = frame["stock"].eq(stock)
    frame.loc[held, ["trade_open", "trade_close"]] = [0.46, 0.46]
    frame.loc[held & frame["date"].eq(days[1]), ["trade_open", "trade_close"]] = [0.44, 0.44]
    frame.loc[held & frame["date"].eq(days[2]), ["trade_open", "trade_close"]] = [0.42, 0.42]
    frame.loc[held & frame["date"].ge(days[3]), ["trade_open", "trade_close"]] = [70, 70]
    scores = (held & frame["date"].eq(days[0])) | (~held & frame["date"].ge(days[6]))
    roster = pd.DataFrame({"delist_date": [pd.Timestamp("2025-04-30"), pd.NaT]}, index=[stock, "600000.SH"])
    result = _simulate(frame, days, scores.astype(int), strategy=Strategy(top_n=1, rebalance_days=1), roster=roster)
    positions = result["positions"].loc[lambda rows: rows["stock"].eq(stock)]
    halt = positions.loc[positions["date"].eq("2025-04-11")].iloc[0]
    assert halt["market_status"] == "suspended" and halt["mark_price"] == 0.42
    detached = positions.loc[positions["position_state"].eq("stranded_delisted")]
    assert detached["detached_on"].unique().tolist() == ["2025-04-30"]
    assert detached["mark_price"].eq(0.42).all() and detached["listed_market_value"].eq(0).all()
    orders = result["orders"]
    assert not (orders["stock"].eq(stock) & orders["signal_date"].ge("2025-04-30")).any()
    assert result["trades"]["symbol"].tolist() == [stock, "600000.SH"]
    assert result["metrics"]["delist_transition_count"] == 1
    last = result["daily"].iloc[-1]
    assert last["equity"] - last["zero_recovery_equity"] == pytest.approx(last["stranded_delisted_value"])


def test_evidence_limits_are_timestamped_and_roll_on_rule_effective_date():
    from datetime import date

    evidence = load_market_evidence()
    assert evidence.event("600070.SH", "risk_warning", date(2024, 5, 17)) is None
    assert evidence.event("600070.SH", "risk_warning", date(2024, 5, 20)) is not None
    assert evidence.rule("sh_main", date(2026, 7, 3)).ratio == 0.05
    assert evidence.rule("sh_main", date(2026, 7, 6)).ratio == 0.10
    assert all(len(digest) == 64 for digest in execution_evidence_identity().values())
