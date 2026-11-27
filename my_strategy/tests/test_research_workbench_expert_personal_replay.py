import json

import pandas as pd
import pytest

from my_strategy.research_workbench import expert_replay, expert_personal_replay
from my_strategy.tests.test_research_workbench_expert_replay import archive


def test_watchlist_returns_common_normalized_results_and_independent_costs(archive):
    runs, base = archive
    scope = {**base, "scope": "watchlist", "symbols": ["600000.SH", "600000.SH"],
             "model_id": "test-model", "initial_cash": 765000, "commission_bps": "12"}
    identity = expert_replay.replay_identity(scope)
    assert identity == expert_replay.replay_identity({**scope, "symbols": ["600000.SH"]})
    output = expert_replay.run(run_id="watchlist-view", spec={**scope, "input_identity": identity})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"]["model_id"] == "test-model"
    assert report["source"]["initial_cash"] == 765000
    assert report["source"]["execution_costs"]["commission"] == 0.0012
    assert set(report["accounts"]) == {"full_pool", "watchlist"}
    assert all(all(data["verification"].values()) for data in report["accounts"].values())
    series = {series["id"]: series["points"] for series in report["performance"]["series"]}
    assert set(series) == {"full_pool", "watchlist", "benchmark"}
    assert all(row[0]["nav"] == 1 for row in series.values())
    assert [[point["date"] for point in rows] for rows in series.values()] == [
        ["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"]
    ] * 3
    path = runs / "watchlist-view/backtests/custom/watchlist_normal_trades.parquet"
    trades = pd.read_parquet(path)
    assert set(trades["symbol"]) == {"600000.SH"}
    assert trades["commission_rate"].eq(0.0012).all()
    assert report["accounts"]["full_pool"]["downloads"]["trades"].endswith("custom--full_pool_normal_trades.parquet")
    with pytest.raises(KeyError):
        expert_personal_replay.authorized_download(report, "../reports/expert_research.json")

    no_filter = expert_replay.run(run_id="no-watchlist", spec={**scope, "symbols": []})
    report = json.loads(no_filter.read_text(encoding="utf-8"))
    assert list(report["accounts"]) == ["full_pool"]
    assert [entry["id"] for entry in report["performance"]["series"]] == ["full_pool", "benchmark"]
    full = "backtests/custom/full_pool_normal_trades.parquet"
    assert (runs / "watchlist-view" / full).samefile(runs / "no-watchlist" / full)


def test_independent_latest_order_is_pending_and_separate_from_fills(archive):
    runs, base = archive
    scope = {**base, "scope": "single_stock", "symbol": "000001.SZ", "end": "2025-03-05"}
    output = expert_replay.run(run_id="independent-view", spec=scope)
    report = json.loads(output.read_text(encoding="utf-8"))
    latest = report["latest_action"]
    assert latest["state"] == "buy" and latest["current_shares"] == 0 and latest["rebalance_due"]
    assert latest["signal_date"] == "2025-03-05" and latest["next_trade_date"] == "2025-03-06"
    assert latest["original_pool"]["rank"] == 1
    assert latest["pending_order"]["status"] == "pending"
    assert latest["pending_order"]["date"] == "2025-03-06"
    for strategy in report["chart"]["strategies"]:
        assert {row["stage"] for row in strategy["events"]} == {"signal"}
        assert all(row["status"] == "pending" and row["time"] == "2025-03-05" for row in strategy["events"])
    for account in ("single_stock", "full_pool"):
        suffix = f"{account}_pending.parquet"
        assert expert_personal_replay.authorized_download(report, f"custom--{suffix}") == f"backtests/custom/{suffix}"
        path = runs / "independent-view/backtests/custom" / suffix
        assert pd.read_parquet(path)["status"].tolist() == ["pending"]
    assert pd.read_parquet(runs / "independent-view/backtests/custom/single_stock_normal_trades.parquet").empty

    panel_path = runs / "base/data_snapshots/panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel.loc[panel["date"].gt(pd.Timestamp("2025-03-05")), "trade_open"] *= 4
    panel.to_parquet(panel_path, index=False)
    coverage_path = runs / "expert/data_snapshots/coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["frozen_input_sha256"]["panel.parquet"] = expert_replay._sha(panel_path)
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    after = expert_replay.run(run_id="future-input-changed", spec=scope)
    checked = json.loads(after.read_text(encoding="utf-8"))
    assert checked["latest_action"] == latest
