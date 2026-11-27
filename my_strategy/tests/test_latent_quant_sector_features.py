import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.sector_features import (
    build_native_sector_daily,
    build_sector_daily,
    strategy_confirmation,
    summarize_selection_candidates,
)


def _row(
    day,
    stock,
    industry,
    close,
    *,
    open_=None,
    high=None,
    low=None,
    amount=100.0,
    turnover=1.0,
    suspended=False,
    market_status="",
):
    return {
        "date": day,
        "stock": stock,
        "industry": industry,
        "open": close if open_ is None else open_,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "amount": amount,
        "turnover": turnover,
        "suspended": suspended,
        "market_status": market_status,
    }


def _event_frame():
    days = pd.bdate_range("2026-01-05", periods=4)
    rows = [
        _row(days[0], "A", "tech", 10),
        _row(days[0], "B", "tech", 20),
        _row(days[0], "C", "finance", 5),
        _row(days[0], "U", None, 7),
        _row(days[1], "A", "tech", 11, open_=11, high=12, low=10, amount=100, turnover=1),
        _row(days[1], "B", "tech", 20, open_=18, high=22, low=17, amount=0, turnover=3),
        _row(days[1], "C", "finance", 5, amount=200, turnover=2),
        _row(days[1], "U", "unknown", 7, amount=50, turnover=np.nan),
        # B changes industry using its own pre-change price history. C is delisted.
        _row(days[2], "A", "tech", 12.1, open_=12, high=13, low=11, amount=0),
        _row(days[2], "B", "finance", 22, amount=30),
        _row(days[2], "C", "finance", 5, amount=999, market_status="delisted"),
        _row(days[2], "N", "tech", 8, amount=20),  # IPO: no previous close.
        _row(days[2], "M", "tech", 4, amount=0),
        _row(days[2], "U", "unknown", 7, amount=10),
        _row(days[3], "A", "tech", 13, suspended=True, market_status="suspended"),
        _row(days[3], "B", "finance", 23),
        _row(days[3], "C", "finance", 5, market_status="delisted"),
        _row(days[3], "N", "tech", 8.8),
        _row(days[3], "M", "tech", 4.2, open_=np.nan, high=4.5, low=3.9),
        _row(days[3], "U", "unknown", 7),
    ]
    return days, pd.DataFrame(rows)


def _trend_frame(periods=70):
    days = pd.bdate_range("2025-09-01", periods=periods)
    rows = []
    for index, day in enumerate(days):
        recent_amount = 200.0 if index >= periods - 5 else 100.0
        for stock, industry, start, growth in (
            ("A", "up", 10.0, 1.01),
            ("B", "up", 20.0, 1.01),
            ("C", "down", 100.0, 0.99),
            ("D", "down", 50.0, 0.99),
        ):
            close = start * growth**index
            rows.append(_row(
                day, stock, industry, close,
                high=close * 1.01, low=close * 0.99, amount=recent_amount,
            ))
    return days, pd.DataFrame(rows)


def _candidate_ledger(days):
    rows = []
    for day_index, day in enumerate(days):
        reason = "" if day_index == 0 else "NOT_REBALANCE_DAY"
        specifications = (
            ("A", "tech", 3.0, 1, True, 0.10, "BUY", 2, 1),
            ("B", "tech", 2.0, 2, True, 0.00, "", 0, 0),
            ("C", "finance", 1.0, 3, False, 0.20, "BUY", 1, 1),
            ("U", None, 0.5, 4, True, 0.00, "SELL", 1, 0),
        )
        for stock, industry, score, rank, qualified, target, side, orders, fills in specifications:
            rows.append({
                "date": day,
                "stock": stock,
                "industry": industry,
                "score": score,
                "global_rank": rank,
                "qualified": qualified,
                "purchase_candidate": day_index == 1 or stock != "B",
                "current_shares": 0,
                "target_weight": target,
                "intent_side": side,
                "selection_reason": reason,
                "order_count": orders,
                "fill_count": fills,
            })
    return pd.DataFrame(rows)


def test_synthetic_ohlc_lifecycle_status_and_unknown_accounting():
    days, frame = _event_frame()
    benchmark = pd.Series(100.0, index=days)
    result = build_sector_daily(frame, benchmark, sessions=days)
    assert not result.duplicated(["date", "industry"]).any()

    first = result.loc[result["date"].eq(days[0])]
    assert first[["open", "high", "low", "close"]].isna().all().all()
    assert first["synthetic_price_reason"].eq("NO_VALID_CONSTITUENTS").all()

    tech1 = result.set_index(["date", "industry"]).loc[(days[1], "tech")]
    # A ratios: 1.10/1.20/1.00/1.10; B: 0.90/1.10/0.85/1.00.
    assert tech1["open"] == pytest.approx(1.00)
    assert tech1["high"] == pytest.approx(1.15)
    assert tech1["low"] == pytest.approx(0.925)
    assert tech1["close"] == pytest.approx(1.05)
    assert tech1["price_participant_count"] == 2
    assert tech1["sector_amount"] == 100  # B's zero amount is retained.
    assert tech1["market_amount_share"] == pytest.approx(100 / 350)
    assert tech1["turnover_median"] == 2
    assert tech1["turnover_coverage"] == 1

    tech2 = result.set_index(["date", "industry"]).loc[(days[2], "tech")]
    assert tech2["constituent_count"] == 3
    assert tech2["price_participant_count"] == 1
    assert tech2["missing_price_count"] == 2  # IPO/new rows lack prior closes.
    assert tech2["close"] == pytest.approx(1.05 * 1.10)
    assert tech2["sector_amount"] == 20
    assert tech2["market_amount_share"] == pytest.approx(20 / 60)
    finance2 = result.set_index(["date", "industry"]).loc[(days[2], "finance")]
    assert finance2["constituent_count"] == 1  # Delisted C is absent, not zero-filled.
    assert finance2["close"] == pytest.approx(1.10)

    tech3 = result.set_index(["date", "industry"]).loc[(days[3], "tech")]
    assert tech3["constituent_count"] == 3
    assert tech3["price_participant_count"] == 1
    assert tech3["suspended_count"] == 1
    assert tech3["missing_price_count"] == 1  # M has a missing current open.
    assert tech3["close"] == pytest.approx(1.05 * 1.10 * 1.10)

    unknown = result.loc[result["industry"].eq("unknown")]
    assert len(unknown) == len(days)
    assert unknown["unclassified_count"].eq(1).all()
    valid = result.dropna(subset=["open", "high", "low", "close"])
    assert (valid["low"] <= valid[["open", "close"]].min(axis=1)).all()
    assert (valid["high"] >= valid[["open", "close"]].max(axis=1)).all()


def test_amount_activity_breadth_advances_and_extremes_match_hand_calculation():
    days, frame = _trend_frame()
    result = build_sector_daily(frame, pd.Series(100.0, index=days), sessions=days)
    final = result.loc[result["date"].eq(days[-1])].set_index("industry")

    for industry in ("up", "down"):
        row = final.loc[industry]
        assert row["sector_amount"] == 400
        assert row["market_amount_share"] == 0.5
        assert row["amount_change_1d"] == pytest.approx(0.0)
        assert row["amount_change_5d"] == pytest.approx(1.0)
        assert row["amount_ma5"] == 400
        assert row["amount_ma20"] == 250  # (15 * 200 + 5 * 400) / 20
        assert row["activity_ratio"] == pytest.approx(1.6)
    assert final.loc["up", "breadth_above_ma20"] == 1.0
    assert final.loc["up", "advance_ratio"] == 1.0
    assert final.loc["up", "new_high_20d_count"] == 2
    assert final.loc["up", "new_low_20d_count"] == 0
    assert final.loc["down", "breadth_above_ma20"] == 0.0
    assert final.loc["down", "advance_ratio"] == 0.0
    assert final.loc["down", "new_high_20d_count"] == 0
    assert final.loc["down", "new_low_20d_count"] == 2


def test_positive_negative_and_insufficient_resonance_and_activity_gate():
    days, frame = _trend_frame()
    benchmark = pd.Series(100.0, index=days)
    result = build_sector_daily(frame, benchmark, sessions=days)
    final = result.loc[result["date"].eq(days[-1])].set_index("industry")

    up = final.loc["up"]
    assert (up["price_trend"], up["breadth_trend"], up["relative_strength_trend"]) == (1, 1, 1)
    assert up["market_trend_score"] == 3
    assert up["strong_up_resonance"]
    assert up["resonance_state"] == "strong_up"
    down = final.loc["down"]
    assert (down["price_trend"], down["breadth_trend"], down["relative_strength_trend"]) == (-1, -1, -1)
    assert down["market_trend_score"] == -3
    assert down["strong_down_resonance"]
    assert down["resonance_state"] == "strong_down"
    assert result.loc[result["date"].eq(days[10]), "resonance_state"].eq("insufficient").all()

    quiet = frame.copy()
    quiet_up = quiet["industry"].eq("up") & quiet["date"].gt(days[-6])
    quiet.loc[quiet_up, "amount"] = 10.0
    quiet_result = build_sector_daily(quiet, benchmark, sessions=days)
    quiet_up_final = quiet_result.set_index(["date", "industry"]).loc[(days[-1], "up")]
    assert quiet_up_final["market_trend_score"] == 3
    assert quiet_up_final["activity_ratio"] < 1
    assert not quiet_up_final["strong_up_resonance"]
    assert quiet_up_final["resonance_state"] == "up"


def test_candidate_counts_non_rebalance_and_global_reconciliation():
    days = pd.bdate_range("2026-02-02", periods=2)
    ledger = _candidate_ledger(days)
    result = summarize_selection_candidates(ledger, sessions=days, rebalance_dates=[days[0]])

    assert set(result["industry"]) == {"tech", "finance", "unknown"}
    first = result.loc[result["date"].eq(days[0])].set_index("industry")
    assert first.loc["tech", "qualified_count"] == 2
    assert first.loc["tech", "purchase_candidate_count"] == 1
    assert first.loc["tech", "target_count"] == 1
    assert first.loc["tech", "buy_intent_count"] == 1
    assert first.loc["tech", "order_count"] == 2  # One stock row can carry two orders.
    assert first.loc["tech", "fill_count"] == 1
    assert first.loc["unknown", "purchase_candidate_count"] == 1
    assert first["market_purchase_candidate_count"].eq(3).all()
    assert first["market_order_count"].eq(4).all()
    assert first["market_fill_count"].eq(2).all()

    second = result.loc[result["date"].eq(days[1])]
    assert second["purchase_candidate_count"].eq(0).all()
    assert second["selection_reason"].eq("NOT_REBALANCE_DAY").all()
    assert second["qualified_count"].sum() == 3
    assert second["order_count"].sum() == 4
    count_columns = [
        "candidate_row_count", "qualified_count", "purchase_candidate_count", "target_count",
        "buy_intent_count", "order_count", "fill_count",
    ]
    for day, rows in result.groupby("date"):
        for column in count_columns:
            assert rows[column].sum() == rows[f"market_{column}"].iloc[0]
        assert rows["candidate_row_count"].sum() == len(ledger.loc[ledger["date"].eq(day)])


def test_native_ths_index_uses_real_bars_but_never_fabricates_historical_breadth():
    days = pd.bdate_range("2025-09-01", periods=70)
    close = pd.Series(10 * 1.01 ** np.arange(len(days)), index=days)
    daily = pd.DataFrame({
        "date": days, "industry": "881270.TI", "industry_name": "元件",
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
        "amount": np.where(np.arange(len(days)) >= len(days) - 5, 200.0, 100.0),
    })
    benchmark = pd.Series(100.0, index=days)
    result = build_native_sector_daily(daily, benchmark, sessions=days)
    final = result.iloc[-1]
    assert final["close"] == pytest.approx(close.iloc[-1])
    assert final["industry_name"] == "元件"
    assert final["price_trend"] == 1 and final["relative_strength_trend"] == 1
    assert final["breadth_trend"] == 0 and final["market_trend_score"] == 2
    assert pd.isna(final["breadth_above_ma20"]) and pd.isna(final["constituent_count"])
    assert final["activity_ratio"] > 1
    assert not final["strong_up_resonance"]
    assert final["resonance_state"] == "insufficient"
    assert result["synthetic_price_reason"].eq("").all()


def test_strategy_confirmation_uses_only_trailing_history_and_not_market_score():
    days = pd.bdate_range("2026-01-05", periods=31)
    daily = pd.DataFrame({
        "date": days,
        "industry": "tech",
        "purchase_candidate_count": [4 if index in (25, 30) else 0 for index in range(len(days))],
        "is_rebalance": [index % 5 == 0 for index in range(len(days))],
        "market_trend_score": 2,
    })
    result = strategy_confirmation(daily)
    assert result.loc[20, "strategy_confirmation"] == "insufficient"
    assert result.loc[20, "confirmation_non_rebalance_history_count"] == 16
    assert result.loc[25, "confirmation_non_rebalance_history_count"] == 20
    assert result.loc[25, "candidate_mean_5d"] == pytest.approx(0.8)
    assert result.loc[25, "candidate_median_60d"] == 0  # Current candidate count is excluded.
    assert result.loc[25, "strategy_confirmation"] == "confirmed"
    assert result.loc[26, "strategy_confirmation"] == "unconfirmed"
    assert result.loc[30, "strategy_confirmation"] == "confirmed"
    assert result["market_trend_score"].eq(2).all()


def test_future_input_changes_cannot_rewrite_past_outputs():
    days, frame = _trend_frame(30)
    cutoff = days[19]
    benchmark = pd.Series(100.0, index=days)
    baseline = build_sector_daily(frame, benchmark, sessions=days)

    changed = frame.copy()
    future = changed["date"].gt(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 3
    changed.loc[future, "amount"] = 0
    changed.loc[future, "industry"] = "future_only"
    changed.loc[future, "suspended"] = True
    changed_benchmark = benchmark.copy()
    changed_benchmark.loc[changed_benchmark.index > cutoff] = 999
    changed_result = build_sector_daily(changed, changed_benchmark, sessions=days)

    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"].le(cutoff)].reset_index(drop=True),
        changed_result.loc[changed_result["date"].le(cutoff)].reset_index(drop=True),
    )

    ledger_days = days[:2]
    ledger = _candidate_ledger(ledger_days)
    candidates_base = summarize_selection_candidates(ledger, rebalance_dates=[ledger_days[0]])
    candidates_changed = ledger.copy()
    later = candidates_changed["date"].eq(ledger_days[1])
    candidates_changed.loc[later, "qualified"] = False
    candidates_changed.loc[later, "order_count"] = 99
    candidates_new = summarize_selection_candidates(candidates_changed, rebalance_dates=[ledger_days[0]])
    pd.testing.assert_frame_equal(
        candidates_base.loc[candidates_base["date"].le(ledger_days[0])].reset_index(drop=True),
        candidates_new.loc[candidates_new["date"].le(ledger_days[0])].reset_index(drop=True),
    )
