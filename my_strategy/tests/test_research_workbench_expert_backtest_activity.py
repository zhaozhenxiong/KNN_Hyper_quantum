"""Archived activity remains account-scoped and paginated."""

import json

import pandas as pd
import pytest

from my_strategy.research_workbench import expert_api, expert_backtest_activity as api, expert_replay, latent_api


@pytest.fixture
def archived_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_api, "RUNS", tmp_path)
    run = "test-portfolio"
    folder = tmp_path / run
    table = folder / "backtests/custom"
    table.mkdir(parents=True)
    (folder / "reports").mkdir()
    (folder / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    kinds = ("daily", "orders", "trades")
    report = {
        "mode": "expert_watchlist", "run": {"run_id": run, "task": expert_replay.TASK},
        "source": {"cost": "double", "approval": "candidate", "requested_model_sessions": ["2025-10-09", "2025-10-10"]},
        "coverage": {"requested_start": "2025-10-01", "requested_end": "2025-11-30"},
        "accounts": {"full_pool": {"downloads": {kind: kind for kind in kinds}}, "watchlist": {"downloads": {}}},
    }
    (folder / f"reports/{expert_replay.REPORT}").write_text(json.dumps(report), encoding="utf-8")
    daily = pd.DataFrame({"date": ["2025-09-30", "2025-10-09", "2025-10-10"], "cash": [200., 195., 190.],
                          "stale_positions": [1, 0, 1], "delisted_positions": [0, 0, 1]})
    order_rows = []
    for i in range(60):
        order_rows.append({"date": "2025-10-10", "signal_date": "2025-10-09", "stock": f"{i:06}.SZ", "side": "BUY" if i < 55 else "SELL",
                           "status": "filled" if i < 59 else "pretrade_rejected", "reason": None, "market_status": "delisted",
                           "requested_shares": 100, "filled_shares": 100 if i < 59 else 0, "execution_price": 1.5})
    order_rows.append({**order_rows[0], "date": "2025-09-30"})
    orders = pd.DataFrame(order_rows)
    trades = orders.loc[orders.filled_shares.gt(0), ["date", "stock", "side"]].rename(columns={"stock": "symbol", "side": "action"}).assign(fee=5.0)
    for kind, frame in zip(kinds, (daily, orders, trades), strict=True):
        frame.to_parquet(table / f"full_pool_double_{kind}.parquet", index=False)
    return run, folder


@pytest.fixture
def empty_archived_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_api, "RUNS", tmp_path)
    run = "test-empty-portfolio"
    source_run = "test-source-research"
    sessions = ["2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09"]
    folder = tmp_path / run
    table = folder / "backtests/custom"
    table.mkdir(parents=True)
    (folder / "reports").mkdir()
    (folder / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    source = tmp_path / source_run
    (source / "config").mkdir(parents=True)
    (source / "backtests/historical_replay").mkdir(parents=True)
    (source / "reports").mkdir()
    config = source / "config/study.json"
    config.write_text(json.dumps({"portfolio": {"rebalance_days": 2, "min_score": 0.0}}), encoding="utf-8")
    decision = {
        "state": "research_only", "historical_numerical_gate": False,
        "point_in_time_verified": False, "qualified_forward_results": False,
        "reasons": ["尚无独立封存的 60 日前瞻留出。"],
    }
    source_report = source / f"reports/{expert_replay.SOURCE_REPORT}"
    source_report.write_text(json.dumps({"decision": decision}), encoding="utf-8")
    predictions = pd.DataFrame({
        "date": [day for day in sessions for _ in range(2)],
        "stock": ["000001.SZ", "000002.SZ"] * len(sessions),
        "score": [-0.05, -0.04, -0.03, -0.02, -0.04, -0.03, -0.02, -0.01, -0.03, -0.02],
    })
    prediction_file = source / "backtests/historical_replay/fusion_predictions.parquet"
    predictions.to_parquet(prediction_file, index=False)
    proof = {"config_sha256": expert_replay._sha(config), "predictions_sha256": expert_replay._sha(prediction_file)}
    report = {
        "mode": "expert_watchlist", "run": {"run_id": run, "task": expert_replay.TASK},
        "source": {
            "cost": "normal", "approval": "candidate", "source_run_id": source_run, "arm": "fusion",
            "requested_model_sessions": sessions, "source_report_sha256": expert_replay._sha(source_report),
            "source_files": {f"{source_run}/historical_replay": proof},
            "model_versions": [{"run_id": source_run, "fold": "historical_replay",
                                "valid_from": sessions[0], "valid_through": sessions[-1]}],
        },
        "coverage": {"requested_start": sessions[0], "requested_end": sessions[-1]},
        "accounts": {"full_pool": {"downloads": {kind: kind for kind in ("daily", "orders", "trades")}}},
    }
    (folder / f"reports/{expert_replay.REPORT}").write_text(json.dumps(report), encoding="utf-8")
    daily = pd.DataFrame({"date": sessions, "cash": [100.] * len(sessions),
                          "stale_positions": [0] * len(sessions), "delisted_positions": [0] * len(sessions)})
    orders = pd.DataFrame(columns=["date", "signal_date", "stock", "side", "status", "reason", "market_status",
                                   "requested_shares", "filled_shares", "execution_price"])
    trades = pd.DataFrame(columns=["date", "symbol", "action", "fee"])
    for kind, frame in zip(("daily", "orders", "trades"), (daily, orders, trades), strict=True):
        frame.to_parquet(table / f"full_pool_normal_{kind}.parquet", index=False)
    return run, folder


def test_order_pages_and_filters_preserve_count_totals_and_source_rows(archived_portfolio):
    run, _ = archived_portfolio
    first = api.activity(run)
    assert first["summary"] == {"sessions": 2, "trades": 59, "orders": 60, "last_trade_date": "2025-10-10",
                                "last_order_date": "2025-10-10", "unfilled_after_last": 0,
                                "unfilled_on_last_trade_date": 1}
    assert first["filtered"] == {"page": 1, "page_size": 50, "pages": 2, "total": 60}
    assert len(first["entries"]) == 50 and first["entries"][0]["fee"] == 5.0
    assert first["entries"][0]["source_row"] == 1
    partial = api.activity(run, "symbol=000059&status=pretrade_rejected&month=2025-10&page=9")
    assert partial["filtered"] == {"page": 1, "page_size": 50, "pages": 1, "total": 1}
    assert partial["entries"][0]["price"] is None and partial["entries"][0]["fee"] is None
    assert partial["entries"][0]["filled_shares"] == 0
    assert partial["months"][0]["delisted_positions_at_end"] == 1
    assert first["months"][1]["trades"] is None and first["months"][1]["sessions"] == 0
    assert first["evidence"]["basis"] == "unavailable"
    watch = api.activity(run, "account=watchlist")
    assert watch["summary"]["sessions"] == 0 and not watch["entries"]


@pytest.mark.parametrize("query", ["account=benchmark", "symbol=../stock", "month=2025-1", "side=SHORT", "page=0", "page=1&page=2", "after_last=yes", "artifact=/tmp"])
def test_invalid_filters_are_rejected_before_reading_data(query):
    with pytest.raises(latent_api.RouteError) as error:
        api.activity("not-a-run", query)
    assert error.value.status == 400


def test_activity_requires_completed_run_and_canonical_id(archived_portfolio):
    run, folder = archived_portfolio
    with pytest.raises(latent_api.RouteError):
        api.activity("../../test-portfolio")
    with pytest.raises(latent_api.RouteError) as unknown:
        api.activity("absent-run")
    assert unknown.value.status == 404
    (folder / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(latent_api.RouteError) as error:
        api.activity(run)
    assert error.value.status == 404


def test_http_activity_passes_the_query_to_authorized_ledger(archived_portfolio):
    run, _ = archived_portfolio

    class Handler:
        def _json(self, status, data):
            self.result = (status, data)

    handler = Handler()
    assert expert_api.handle_get(handler, f"/api/expert/replays/{run}/activity", "status=filled&page=2")
    status, report = handler.result
    assert status == 200
    assert report["filtered"] == {"page": 2, "page_size": 50, "pages": 2, "total": 59}


def test_activity_side_filters_before_pagination_and_combines_with_status(archived_portfolio):
    run, _ = archived_portfolio
    buys = api.activity(run, "side=BUY&page=2")
    assert buys["filtered"]["total"] == 55
    assert len(buys["entries"]) == 5 and {row["side"] for row in buys["entries"]} == {"BUY"}
    sells = api.activity(run, "side=SELL&status=filled&month=2025-10")
    assert sells["filtered"]["total"] == 4
    assert {row["side"] for row in sells["entries"]} == {"SELL"}
    assert sells["summary"] == api.activity(run, "side=all")["summary"]


def test_after_last_filter_excludes_same_day_and_preserves_global_summary(archived_portfolio):
    run, folder = archived_portfolio
    report_path = folder / f"reports/{expert_replay.REPORT}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"]["requested_model_sessions"].append("2025-10-13")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    table = folder / "backtests/custom"
    daily = pd.read_parquet(table / "full_pool_double_daily.parquet")
    daily = pd.concat([daily, pd.DataFrame({"date": ["2025-10-13"], "cash": [190.], "stale_positions": [1],
                                            "delisted_positions": [1]})], ignore_index=True)
    daily.to_parquet(table / "full_pool_double_daily.parquet", index=False)
    orders = pd.read_parquet(table / "full_pool_double_orders.parquet")
    later = orders.iloc[[0, 0]].copy()
    later["date"] = "2025-10-13"
    later["stock"] = ["000060.SZ", "000061.SZ"]
    later["side"] = "SELL"
    later["status"] = ["rejected", "pretrade_rejected"]
    later["reason"] = ["price_limit", "suspended"]
    later["filled_shares"] = 0
    pd.concat([orders, later], ignore_index=True).to_parquet(table / "full_pool_double_orders.parquet", index=False)

    result = api.activity(run, "after_last=1")
    assert result["filtered"]["total"] == 2
    assert {row["date"] for row in result["entries"]} == {"2025-10-13"}
    assert result["summary"] == {"sessions": 3, "trades": 59, "orders": 62, "last_trade_date": "2025-10-10",
                                "last_order_date": "2025-10-13", "unfilled_after_last": 2,
                                "unfilled_on_last_trade_date": 1}
    assert result["inactivity_diagnosis"]["zero_fill_reasons"] == [
        {"status": "pretrade_rejected", "reason": "suspended", "side": "SELL", "count": 1,
         "first_date": "2025-10-13", "last_date": "2025-10-13"},
        {"status": "rejected", "reason": "price_limit", "side": "SELL", "count": 1,
         "first_date": "2025-10-13", "last_date": "2025-10-13"},
    ]
    assert api.activity(run, "after_last=1&month=2025-11")["filtered"]["total"] == 0
    assert not api.activity(run, "after_last=1&side=BUY")["entries"]

def test_zero_trade_run_returns_full_interval_pinned_score_diagnosis(empty_archived_portfolio):
    run, _ = empty_archived_portfolio
    result = api.activity(run)
    assert result["summary"] == {"sessions": 5, "trades": 0, "orders": 0, "last_trade_date": None,
                                "last_order_date": None, "unfilled_after_last": None,
                                "unfilled_on_last_trade_date": None}
    full = result["inactivity_diagnosis"]["full_run"]
    assert full == {
        "sessions": 5, "score_days": 5, "rebalances": 3,
        "eligible_stock_days": 10, "eligible_on_rebalance": 6,
        "qualified_stock_days": 0, "qualified_on_rebalance": 0, "qualified_off_rebalance": 0,
        "negative_on_rebalance": 6, "zero_on_rebalance": 0, "all_negative_on_rebalance": True,
        "score_min_on_rebalance": -0.05, "score_max_on_rebalance": -0.02,
        "orders": 0, "buy_orders": 0, "sell_orders": 0, "zero_fill_orders": 0,
        "rejected_orders": 0, "pretrade_rejected_orders": 0,
        "reason_code": "NO_QUALIFIED_CANDIDATES",
    }
    assert result["evidence"]["basis"] == "pinned_archive"
    assert result["research_status"] == {
        "status": "research_only", "approval": "candidate", "source_run_id": "test-source-research",
        "release_gate": {
            "basis": "pinned_archive", "passed": False, "state": "research_only",
            "historical_numerical_gate": False, "point_in_time_verified": False,
            "qualified_forward_results": False, "reasons": ["尚无独立封存的 60 日前瞻留出。"],
        },
    }
    june = result["months"][0]
    assert june["rebalance_days"] == 3 and june["eligible_stock_days"] == 10
    assert june["positive_on_rebalance"] == 0 and june["positive_off_rebalance"] == 0


def test_zero_trade_run_without_verifiable_scores_does_not_infer_a_score_reason(empty_archived_portfolio):
    run, folder = empty_archived_portfolio
    report_path = folder / f"reports/{expert_replay.REPORT}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"]["source_files"] = {}
    report["source"]["source_report_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = api.activity(run)
    full = result["inactivity_diagnosis"]["full_run"]
    assert result["evidence"]["basis"] == "unavailable"
    assert full["sessions"] == 5 and full["orders"] == 0
    assert full["rejected_orders"] == 0 and full["pretrade_rejected_orders"] == 0
    assert full["rebalances"] is None and full["qualified_on_rebalance"] is None
    assert full["all_negative_on_rebalance"] is None and full["reason_code"] is None
    assert result["inactivity_diagnosis"]["zero_fill_reasons"] == []
    assert result["research_status"]["release_gate"]["basis"] == "unavailable"
    assert result["research_status"]["release_gate"]["passed"] is None


@pytest.mark.parametrize("replacement, reason_code", [
    (0.0, "NO_QUALIFIED_CANDIDATES"),
    (0.01, "QUALIFIED_CANDIDATES_WITHOUT_ORDERS"),
])
def test_zero_trade_score_reason_respects_strict_threshold_and_evidence_limits(
        empty_archived_portfolio, replacement, reason_code):
    run, folder = empty_archived_portfolio
    report = json.loads((folder / f"reports/{expert_replay.REPORT}").read_text(encoding="utf-8"))
    source = latent_api.RUNS / report["source"]["source_run_id"]
    path = source / "backtests/historical_replay/fusion_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "score"] = replacement
    predictions.to_parquet(path, index=False)
    report["source"]["source_files"][f"{source.name}/historical_replay"]["predictions_sha256"] = expert_replay._sha(path)
    (folder / f"reports/{expert_replay.REPORT}").write_text(json.dumps(report), encoding="utf-8")

    full = api.activity(run)["inactivity_diagnosis"]["full_run"]
    assert full["reason_code"] == reason_code
    assert full["all_negative_on_rebalance"] is False
    assert full["score_max_on_rebalance"] == replacement
    assert full["qualified_on_rebalance"] == (1 if replacement > 0 else 0)


def test_never_filled_orders_use_full_run_ledger_reasons(empty_archived_portfolio):
    run, folder = empty_archived_portfolio
    path = folder / "backtests/custom/full_pool_normal_orders.parquet"
    orders = pd.DataFrame([
        {"date": "2026-06-04", "signal_date": "2026-06-03", "stock": "000001.SZ", "side": "BUY",
         "status": "rejected", "reason": "limit_up", "market_status": "active", "requested_shares": 100,
         "filled_shares": 0, "execution_price": None},
        {"date": "2026-06-08", "signal_date": "2026-06-05", "stock": "000002.SZ", "side": "BUY",
         "status": "pretrade_rejected", "reason": "suspended", "market_status": "suspended", "requested_shares": 0,
         "filled_shares": 0, "execution_price": None},
    ])
    orders.to_parquet(path, index=False)

    result = api.activity(run)
    full = result["inactivity_diagnosis"]["full_run"]
    assert result["summary"]["last_trade_date"] is None and result["summary"]["orders"] == 2
    assert full["orders"] == 2 and full["zero_fill_orders"] == 2
    assert full["rejected_orders"] == 1 and full["pretrade_rejected_orders"] == 1
    assert full["reason_code"] == "UNFILLED_ORDERS"
    assert result["inactivity_diagnosis"]["zero_fill_reasons"] == [
        {"status": "pretrade_rejected", "reason": "suspended", "side": "BUY", "count": 1,
         "first_date": "2026-06-08", "last_date": "2026-06-08"},
        {"status": "rejected", "reason": "limit_up", "side": "BUY", "count": 1,
         "first_date": "2026-06-04", "last_date": "2026-06-04"},
    ]


def test_never_traded_uses_whole_run_score_and_order_evidence():
    sessions = pd.bdate_range("2025-10-09", periods=6).strftime("%Y-%m-%d").tolist()
    orders = pd.DataFrame(columns=["date", "side", "filled_shares", "status", "reason"])
    scores = {day: {"eligible": 100, "qualified": 0, "negative": 100, "zero": 0, "min": -0.02, "max": -0.01}
              for day in sessions}
    scheduled = {sessions[0], sessions[-1]}
    diagnosis = api._inactivity(sessions, orders, scores, scheduled, {"basis": "pinned_archive"}, None, set(sessions))
    assert diagnosis["scope"] == "whole_run" and diagnosis["never_traded"]
    assert diagnosis["sessions_after_last"] == 6
    assert diagnosis["rebalances_after_last"] == 1
    assert diagnosis["score_days_after_last"] == 6
    assert diagnosis["positive_on_rebalance_after_last"] == 0
    assert diagnosis["orders_after_last"] == diagnosis["zero_fill_after_last"] == 0
    assert diagnosis["selection_cause"] == "NO_QUALIFIED_CANDIDATES"

    scores[sessions[0]] = {"eligible": 100, "qualified": 2, "negative": 98, "zero": 0, "min": -0.02, "max": 0.01}
    assert api._inactivity(sessions, orders, scores, scheduled, {"basis": "pinned_archive"}, None, set(sessions))["selection_cause"] == "NO_ORDER_INTENTS_OR_UNVERIFIED_SELECTION"
    unavailable = api._inactivity(sessions, orders, {}, None, {"basis": "unavailable"}, None, set(sessions))
    assert unavailable["selection_cause"] is None and unavailable["rebalances_after_last"] is None

    rejected = pd.DataFrame({
        "date": [sessions[1]], "side": ["BUY"], "filled_shares": [0],
        "status": ["rejected"], "reason": ["price_limit"],
    })
    diagnosis = api._inactivity(sessions, rejected, scores, scheduled, {"basis": "pinned_archive"}, None, set(sessions))
    assert diagnosis["selection_cause"] == "ORDERS_NOT_FILLED"
    assert diagnosis["orders_after_last"] == diagnosis["zero_fill_after_last"] == 1
    assert diagnosis["zero_fill_reasons"][0]["reason"] == "price_limit"



    run = "20261123-151605-971646-20261120-expert-replay-9920c41"
    if not (latent_api.RUNS / run / "status.json").is_file():
        pytest.skip("requires the local completed replay")
    october = api.activity(run, "month=2025-10&status=filled")
    assert october["summary"]["trades"] == 455
    assert october["summary"]["last_trade_date"] == "2025-11-05"
    assert october["summary"]["unfilled_after_last"] == 50
    assert october["filtered"]["total"] == 11
    assert october["evidence"]["basis"] == "pinned_archive"
    november = next(item for item in october["months"] if item["month"] == "2025-11")
    assert november["positive_on_rebalance"] == 0 and november["positive_off_rebalance"] == 32
    assert 0 < november["valid_raw_stock_days"] <= november["active_stock_days"]


def test_original_flat_replay_explains_the_whole_zero_order_interval():
    run = "20261125-210750-636512-20261120-expert-replay-ffacc58"
    if not (latent_api.RUNS / run / "status.json").is_file():
        pytest.skip("requires the original local zero-order replay")
    result = api.activity(run)
    assert result["summary"]["sessions"] == 116
    assert result["summary"]["orders"] == result["summary"]["trades"] == 0
    assert result["summary"]["last_trade_date"] is None
    assert result["summary"]["unfilled_after_last"] is None
    diagnosis = result["inactivity_diagnosis"]
    assert diagnosis["scope"] == "whole_run" and diagnosis["never_traded"]
    assert diagnosis["basis"] == "pinned_archive"
    assert diagnosis["sessions_after_last"] == diagnosis["score_days_after_last"] == 116
    assert diagnosis["rebalances_after_last"] == 23
    assert diagnosis["positive_on_rebalance_after_last"] == 0
    assert diagnosis["positive_off_rebalance_after_last"] == 0
    assert diagnosis["selection_cause"] == "NO_QUALIFIED_CANDIDATES"
    assert result["evidence"]["approval"] in {"candidate", "research_only"}



    run = "20261124-212836-758317-20261124-expert-replay-19f8ef2"
    if not (latent_api.RUNS / run / "status.json").is_file():
        pytest.skip("requires the local completed replay")
    result = api.activity(run)
    assert result["summary"] == {"sessions": 420, "trades": 469, "orders": 524,
                                "last_trade_date": "2025-10-15", "last_order_date": "2025-12-03",
                                "unfilled_after_last": 7, "unfilled_on_last_trade_date": 1}
    diagnosis = result["inactivity_diagnosis"]
    assert diagnosis["basis"] == "pinned_archive"
    assert diagnosis["sessions_after_last"] == 268
    assert diagnosis["score_days_after_last"] == 268
    assert diagnosis["rebalances_after_last"] == 53
    assert diagnosis["positive_on_rebalance_after_last"] == 0
    assert diagnosis["positive_off_rebalance_after_last"] == 64
    assert diagnosis["buy_orders_after_last"] == 0
    assert diagnosis["sell_orders_after_last"] == 7
    assert diagnosis["zero_fill_reasons"] == [
        {"status": "pretrade_rejected", "reason": "suspended", "side": "SELL", "count": 4,
         "first_date": "2025-11-12", "last_date": "2025-12-03"},
        {"status": "rejected", "reason": "price_limit", "side": "SELL", "count": 3,
         "first_date": "2025-10-22", "last_date": "2025-11-05"},
    ]
    filtered = api.activity(run, "after_last=1")
    assert filtered["filtered"]["total"] == 7
    assert len(filtered["entries"]) == 7
    assert {row["symbol"] for row in filtered["entries"]} == {"603388.SH"}
    assert {row["filled_shares"] for row in filtered["entries"]} == {0}
    assert all(row["date"] > "2025-10-15" for row in filtered["entries"])


def test_zero_trade_replay_preserves_empty_ledger_and_explains_full_interval():
    run = "20261125-210750-636512-20261120-expert-replay-ffacc58"
    if not (latent_api.RUNS / run / "status.json").is_file():
        pytest.skip("requires the local completed zero-trade replay")
    result = api.activity(run)
    assert result["summary"] == {"sessions": 116, "trades": 0, "orders": 0, "last_trade_date": None,
                                "last_order_date": None, "unfilled_after_last": None,
                                "unfilled_on_last_trade_date": None}
    full = result["inactivity_diagnosis"]["full_run"]
    assert full["rebalances"] == 24 and full["score_days"] == 116
    assert full["eligible_on_rebalance"] > 0 and full["qualified_on_rebalance"] == 0
    assert full["all_negative_on_rebalance"] is True and full["score_max_on_rebalance"] < 0
    assert full["orders"] == full["rejected_orders"] == full["pretrade_rejected_orders"] == 0
    assert full["reason_code"] == "NO_QUALIFIED_CANDIDATES"
    assert result["research_status"]["status"] == "research_only"
    assert result["research_status"]["release_gate"]["basis"] == "pinned_archive"
    assert result["research_status"]["release_gate"]["passed"] is False
