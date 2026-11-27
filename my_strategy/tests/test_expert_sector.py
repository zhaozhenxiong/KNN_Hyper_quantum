import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.domestic_runner import FrozenInputs
from my_strategy.latent_quant.execution import Strategy, simulate
from my_strategy.research_workbench import expert_api, expert_research, expert_sector, latent_api


def _archive(tmp_path, monkeypatch):
    root = tmp_path / "sector-run"
    root.mkdir()
    days = pd.bdate_range("2026-01-02", periods=80)
    frame = pd.MultiIndex.from_product([days, ["600000.SH", "600001.SH"]], names=["date", "stock"]).to_frame(index=False)
    technology = frame["stock"].eq("600001.SH")
    frame["industry"] = technology.map({True: "technology", False: "banks"})
    frame["board"] = "sh_main"
    frame["close"] = 10 + frame.groupby("stock").cumcount() * technology.map({True: 0.08, False: 0.02})
    frame["open"] = frame["close"] * 0.998
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.01
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.99
    frame["trade_open"] = frame["open"]
    frame["trade_close"] = frame["close"]
    frame["has_trade_price"] = True
    frame["amount"] = 20_000_000 + frame.groupby("stock").cumcount() * 100_000
    frame["volume"] = 2_000_000
    frame["turnover"] = 0.02
    frame["suspended"] = False
    frame["signal_eligible"] = True
    benchmark = pd.Series(1.001 ** pd.RangeIndex(len(days)), index=days)
    sessions = days[-10:]
    outcome = simulate(
        frame, pd.Series(technology.map({True: 0.03, False: 0.01}), index=frame.index), sessions,
        benchmark.pct_change(fill_method=None), run_id="sector-run",
        strategy=Strategy(rebalance_days=2, top_n=1, selection_mode="cross_sectional_rank", min_score=None),
    )
    report = expert_sector.write_sector_artifacts(
        root, frame, benchmark, outcome["selection_candidates"], outcome["sector_candidate_daily"],
    )
    ledger_dir = root / "backtests/custom"
    ledger_dir.mkdir(parents=True)
    for name in ("selection_candidates", "sector_candidate_daily"):
        outcome[name].to_parquet(ledger_dir / f"full_pool_normal_{name}.parquet", index=False)
    assert report["status"] == "available"
    assert report["classification"] == "point_in_time_capco"
    (root / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    replay = {
        "mode": "expert_watchlist", "run": {"run_id": root.name, "task": "expert-stock-portfolio-replay"}, "source": {"cost": "normal"},
        "accounts": {"full_pool": {"downloads": {"daily": "yes"}}},
    }
    (root / "reports/expert_stock_replay.json").write_text(json.dumps(replay), encoding="utf-8")
    monkeypatch.setattr(latent_api, "_run_folder", lambda run_id: root if run_id == root.name else tmp_path / run_id)
    return root, outcome, sessions


def test_sector_overview_chart_and_candidate_drilldown_reconcile(tmp_path, monkeypatch):
    root, outcome, sessions = _archive(tmp_path, monkeypatch)
    selected = str(sessions[0].date())
    overview = expert_sector.overview(root.name, f"date={selected}")
    assert overview["status"] == "available"
    assert overview["totals"]["purchase_candidate_count"] == 1
    assert overview["totals"]["fill_count"] == len(outcome["trades"].loc[outcome["trades"]["signal_date"].eq(selected)])
    technology = next(row for row in overview["sectors"] if row["industry"] == "technology")
    assert technology["purchase_candidate_count"] == technology["target_count"] == 1
    assert technology["market_trend_score"] == 3
    assert technology["resonance_state"] == "strong_up"
    assert technology["sparkline"]

    chart = expert_sector.chart(root.name, "technology", f"start={selected}&end={sessions[-1].date()}")
    assert chart["status"] == "available"
    assert len(chart["series"]) == len(sessions)
    benchmark = pd.Series(1.001 ** pd.RangeIndex(len(days := pd.bdate_range("2026-01-02", periods=80))))
    assert chart["series"][0]["benchmark_normalized"] == pytest.approx(
        1.008 * float(benchmark.iloc[-len(sessions)] / benchmark.iloc[0]),
    )
    assert chart["series"][-1]["benchmark_normalized"] / chart["series"][0]["benchmark_normalized"] == pytest.approx(
        float(benchmark.iloc[-1] / benchmark.iloc[-len(sessions)]),
    )
    assert chart["downloads"]["chart_csv"].endswith(".csv")
    assert chart["series"][1]["selection_reason"] == "NOT_REBALANCE_DAY"

    candidates = expert_sector.candidates(root.name, "technology", f"date={selected}")
    assert len(candidates["rows"]) == 1
    assert candidates["rows"][0]["stock"] == "600001.SH"
    assert candidates["rows"][0]["purchase_candidate"] is True
    assert candidates["rows"][0]["order_count"] == 1
    relative = expert_sector.authorized_download(root, {}, "sector_daily.csv")
    assert (root / relative).is_file()


def test_sector_overview_http_route(tmp_path, monkeypatch):
    root, _, sessions = _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(expert_api, "_folder", lambda run_id: root if run_id == root.name else tmp_path / run_id)

    class Handler:
        def _json(self, status, payload):
            self.result = status, payload

        def _send(self, status, body, content_type):
            self.download = status, body, content_type

    handler = Handler()
    assert expert_api.handle_get(
        handler, f"/api/expert/replays/{root.name}/sectors", f"date={sessions[0].date()}",
    )
    status, payload = handler.result
    assert status == 200
    assert payload["status"] == "available"
    assert payload["totals"]["purchase_candidate_count"] == 1

    key = expert_sector.sector_key("technology")
    assert expert_api.handle_get(
        handler, f"/api/expert/replays/{root.name}/downloads/sector--{key}--chart.png", "",
    )
    status, body, content_type = getattr(handler, "download", (None, None, None))
    assert status == 200, getattr(handler, "result", None)
    assert content_type == "image/png"
    assert body.startswith(b"\x89PNG")


def test_sector_training_view_uses_only_market_features_and_supports_leave_one_out(tmp_path, monkeypatch):
    _archive(tmp_path, monkeypatch)
    days = pd.bdate_range("2026-01-02", periods=80)
    stocks = pd.MultiIndex.from_product([days, ["600000.SH", "600001.SH"]], names=["date", "stock"]).to_frame(index=False)
    stocks["industry"] = stocks["stock"].eq("600001.SH").map({True: "technology", False: "banks"})
    stocks["open"] = stocks["close"] = stocks["high"] = stocks["low"] = 10.0
    stocks["amount"] = 20_000_000.0
    stocks["turnover"] = 0.02
    benchmark = pd.Series(1.001 ** pd.RangeIndex(len(days)), index=days)
    data = FrozenInputs(stocks, days, benchmark.pct_change(fill_method=None), pd.DataFrame(), pd.DataFrame(), {}, benchmark)
    attached, evidence = expert_research._verified_sector(data, True)
    assert evidence["candidate_features_used"] is False
    assert "sector_market_trend_score" in attached.frame
    assert not any("candidate" in column for column in evidence["features"])
    assert attached.frame["sector_relative_strength_20d"].notna().any()
    unchanged, disabled = expert_research._verified_sector(data, False)
    assert unchanged.frame is stocks and disabled["enabled"] is False

    weights = {"supertrend_atr": 0.25, "domestic_gpu": 0.5, "sector_gpu": 0.25}
    ablations = expert_research._ablations(weights, {"domestic_gpu": (), "sector_gpu": ()})
    without = ablations["fusion_no_sector"]["weights"]
    assert "sector_gpu" not in without
    assert sum(without.values()) == pytest.approx(1.0)
    assert ablations["fusion_no_sector"]["extra_models_fitted"] is False


def test_empty_candidate_ledgers_return_explicit_empty_results(tmp_path, monkeypatch):
    root, _, _ = _archive(tmp_path, monkeypatch)
    for path in (
        root / "features/sector_candidate_daily.parquet",
        root / "backtests/custom/full_pool_normal_selection_candidates.parquet",
    ):
        pd.read_parquet(path).iloc[0:0].to_parquet(path, index=False)

    overview = expert_sector.overview(root.name, "")
    assert overview["status"] == "available"
    assert overview["dates"] == [] and overview["sectors"]
    assert all(value == 0 for value in overview["totals"].values())
    assert all(row["selection_reason"] == "NO_CANDIDATE_ROWS" for row in overview["sectors"])

    chart = expert_sector.chart(root.name, "technology", "")
    assert chart["status"] == "available" and chart["series"]
    assert all(row["purchase_candidate_count"] == 0 for row in chart["series"])
    detail = expert_sector.candidates(root.name, "technology", "")
    assert detail["status"] == "available" and detail["date"] is None and detail["rows"] == []


def test_old_archive_is_explicitly_unavailable_and_ledger_tampering_is_rejected(tmp_path, monkeypatch):
    root, _, sessions = _archive(tmp_path, monkeypatch)
    selected = str(sessions[0].date())
    path = root / "features/sector_candidate_daily.parquet"
    original = pd.read_parquet(path)
    path.rename(root / "features/sector_candidate_daily.saved")
    missing = expert_sector.overview(root.name, "")
    assert missing["status"] == "unavailable"
    assert "未封存" in missing["reason"]
    (root / "features/sector_candidate_daily.saved").rename(path)

    changed = original.copy()
    changed.loc[changed.index[0], "purchase_candidate_count"] += 1
    changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="不一致"):
        expert_sector.overview(root.name, f"date={selected}")


def test_native_ths_archive_separates_display_classification_and_preserves_execution_counts(tmp_path, monkeypatch):
    root, outcome, sessions = _archive(tmp_path, monkeypatch)
    days = pd.bdate_range("2026-01-02", periods=80)
    daily = []
    for code, growth in (("881270.TI", 1.002), ("881273.TI", 1.001)):
        close = 100 * growth ** np.arange(len(days))
        daily.append(pd.DataFrame({
            "date": days, "industry": code, "industry_name": "元件" if code == "881270.TI" else "白酒",
            "open": close * 0.999, "high": close * 1.001, "low": close * 0.998, "close": close,
            "amount": 1e9 + np.arange(len(days)) * 1e6,
        }))
    native_daily = pd.concat(daily, ignore_index=True)
    memberships = pd.DataFrame([
        {"stock": "600001.SH", "industry": "881270.TI", "industry_name": "元件", "effective_from": days[0]},
        {"stock": "600000.SH", "industry": "881273.TI", "industry_name": "白酒", "effective_from": days[0]},
    ])
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"provider": "ths_fuyao"}), encoding="utf-8")
    snapshot = {
        "root": root, "daily": native_daily, "memberships": memberships,
        "manifest": {"observed_at": "2026-01-02T09:00:00+08:00", "effective_from": "2026-01-02",
                     "catalog_rows": 2, "classified_stocks": 2},
    }
    monkeypatch.setattr("my_strategy.research_workbench.fuyao_sector.load_latest_snapshot", lambda: snapshot)
    days_index = pd.DatetimeIndex(days)
    stocks = pd.MultiIndex.from_product([days_index, ["600000.SH", "600001.SH"]], names=["date", "stock"]).to_frame(index=False)
    stocks["industry"] = stocks["stock"].eq("600001.SH").map({True: "technology", False: "banks"})
    stocks["open"] = stocks["close"] = stocks["high"] = stocks["low"] = 10.0
    stocks["amount"] = 20_000_000.0
    benchmark = pd.Series(1.001 ** np.arange(len(days)), index=days)
    report = expert_sector.write_sector_artifacts(
        root, stocks, benchmark, outcome["selection_candidates"], outcome["sector_candidate_daily"], provider="ths_fuyao",
    )
    assert report["provider"] == "ths_fuyao" and report["price_basis"] == "ths_native_index"
    assert set(report["industries"]) == {"881270.TI", "881273.TI"}
    overview = expert_sector.overview(root.name, f"date={sessions[0].date()}")
    assert overview["provider"] == "ths_fuyao"
    expected_orders = int(outcome["sector_candidate_daily"].loc[
        outcome["sector_candidate_daily"]["date"].eq(str(sessions[0].date())), "order_count",
    ].sum())
    assert overview["totals"]["order_count"] == expected_orders
    technology = next(row for row in overview["sectors"] if row["industry"] == "881270.TI")
    assert technology["industry_name"] == "元件" and technology["purchase_candidate_count"] == 1
    assert technology["breadth_above_ma20"] is None and technology["constituent_count"] is None
    chart = expert_sector.chart(root.name, "881270.TI", f"start={sessions[0].date()}&end={sessions[-1].date()}")
    assert chart["industry_name"] == "元件"
    assert all(row["resonance_state"] == "insufficient" for row in chart["series"])
    assert all(row["breadth_above_ma20"] is None for row in chart["series"])
    detail = expert_sector.candidates(root.name, "881270.TI", f"date={sessions[0].date()}")
    assert detail["rows"][0]["stock"] == "600001.SH"
    assert detail["downloads"]["selection_candidates"].endswith("sector_selection_candidates.parquet")
    relative = expert_sector.authorized_download(root, {}, "sector_selection_candidates.parquet")
    assert (root / relative).is_file()

    late_memberships = memberships.copy()
    late_memberships["effective_from"] = days[-1]
    snapshot["memberships"] = late_memberships
    expert_sector.write_sector_artifacts(
        root, stocks, benchmark, outcome["selection_candidates"], outcome["sector_candidate_daily"], provider="ths_fuyao",
    )
    before_snapshot = expert_sector.overview(root.name, f"date={sessions[0].date()}")
    assert set(row["industry"] for row in before_snapshot["sectors"]) == {"881270.TI", "881273.TI", "unknown"}
    native_rows = [row for row in before_snapshot["sectors"] if row["industry"] != "unknown"]
    assert all(row["purchase_candidate_count"] is None for row in native_rows)
    assert all(row["selection_reason"] == "NO_POINT_IN_TIME_MEMBERSHIP" for row in native_rows)
    unknown = next(row for row in before_snapshot["sectors"] if row["industry"] == "unknown")
    assert unknown["purchase_candidate_count"] == 1
    native_chart = expert_sector.chart(
        root.name, "881270.TI", f"start={sessions[0].date()}&end={sessions[0].date()}",
    )
    assert native_chart["series"][0]["purchase_candidate_count"] is None
    assert native_chart["series"][0]["selection_reason"] == "NO_POINT_IN_TIME_MEMBERSHIP"
    unknown_chart = expert_sector.chart(root.name, "unknown", f"date={sessions[0].date()}")
    assert unknown_chart["status"] == "unavailable" and unknown_chart["series"] == []
    unknown_detail = expert_sector.candidates(root.name, "unknown", f"date={sessions[0].date()}")
    assert len(unknown_detail["rows"]) == 2
