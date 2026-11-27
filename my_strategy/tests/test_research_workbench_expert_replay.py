from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core import paths
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, scaled_study_cost
from my_strategy.research_workbench import expert_replay, expert_stock_inference, latent_api, stock_chart_api
from my_strategy.storage.expert_artifacts import ExpertArtifactStore


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(latent_api, "RUNS", runs)
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(stock_chart_api, "_security", lambda symbol: {
        "symbol": symbol, "name": "样例股票", "first_date": "2025-02-28", "last_date": "2025-03-07",
    })
    baseline = runs / "base"
    expert = runs / "expert"
    _json(baseline / "status.json", {"status": "completed"})
    _json(baseline / "reports/latent_quant.json", {"run": {"run_id": "base"}})
    data = baseline / "data_snapshots"
    data.mkdir()
    stocks = ["600000.SH", "000001.SZ"]
    days = pd.bdate_range("2025-02-28", "2025-03-07")
    tape = pd.MultiIndex.from_product([days, stocks], names=["date", "stock"]).to_frame(index=False)
    tape = tape.assign(trade_open=10.0, trade_high=10.5, trade_low=9.5, trade_close=10.0,
                       amount=20_000_000, volume=2_000_000, source="raw", has_trade_price=True,
                       board="sh_main", industry="bank", signal_eligible=True)
    tape.to_parquet(data / "panel.parquet", index=False)
    monkeypatch.setattr(stock_chart_api, "_warehouse_series", lambda identity, *_args, **_kwargs: (
        tape.loc[tape["stock"].eq(stocks[0])], pd.DataFrame(columns=["date", "stock", "market_status", "source"]),
        None, None, identity,
    ))
    pd.DataFrame({"stock": stocks, "delist_date": [pd.NaT, pd.NaT]}).to_parquet(data / "roster.parquet", index=False)
    pd.DataFrame({"date": days, "open": 4000.0, "close": 4000.0}).to_parquet(data / "benchmark.parquet", index=False)
    pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "stock": pd.Series(dtype=str),
                  "source": pd.Series(dtype=str), "market_status": pd.Series(dtype=str)}).to_parquet(
        data / "market_status.parquet", index=False,
    )
    _json(expert / "status.json", {"status": "completed"})
    _json(expert / "config/study.json", {
        "source_run_id": "base", "stock_cap": 2, "seed": 42,
        "portfolio": asdict(Strategy(top_n=1, rebalance_days=1)), "initial_cash": 1_000_000,
        "expense_model": asdict(StudyCostModel()),
        "cost_multipliers": {"normal": 1, "double": 2},
    })
    _json(expert / "data_snapshots/coverage.json", {"source_run_id": "base", "frozen_input_sha256": {
        file.name: expert_replay._sha(file) for file in data.glob("*.parquet")
    }})
    folds = []
    for fold, start, end, label, positive in (
        ("oos_1", "2025-03-03", "2025-03-04", "2025-02-28", stocks[0]),
        ("oos_2", "2025-03-05", "2025-03-07", "2025-03-04", stocks[1]),
    ):
        folds.append({
            "period": fold, "evaluation_start": start, "evaluation_end": end,
            "models": {part: {"last_label_end": label} for part in ("train", "validation", "calibration")},
            "arms": {"fusion": {"execution": {"normal": {}, "double": {}}}},
        })
        positions = tape.loc[tape["date"].between(start, end), ["date", "stock"]].copy()
        positions["score"] = positions["stock"].eq(positive).map({True: 1, False: -1})
        out = expert / "backtests" / fold
        out.mkdir(parents=True)
        positions.assign(q10=-0.1, q50=0.1, q90=0.3).to_parquet(out / "fusion_predictions.parquet", index=False)
    _json(expert / "reports/expert_research.json", {"run": {"run_id": "expert"}, "folds": folds})
    release = {"status": "candidate", "release_id": "candidate-test", "source_run_id": "expert", "arm": "fusion"}
    return runs, {
        **{key: release[key] for key in ("release_id", "source_run_id", "arm")},
        "release": release, "diagnostic": True, "symbol": stocks[0],
        "start": "2025-03-03", "end": "2025-03-07", "cost": "normal",
    }


def test_custom_portfolio_replay_keeps_positions_across_folds(archive):
    runs, spec = archive
    identity = expert_replay.replay_identity(spec)
    assert [row["fold"] for row in identity["model_versions"]] == ["oos_1", "oos_2"]
    output = expert_replay.run(run_id="interval-result", spec=spec)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["window"]["stock_count"] == 2 and report["window"]["sessions"] == 5
    trades = pd.read_parquet(runs / "interval-result/backtests/custom/fusion_normal_trades.parquet")
    assert trades["symbol"].nunique() == 2
    position = pd.read_parquet(runs / "interval-result/backtests/custom/fusion_normal_positions.parquet")
    assert position.loc[position["date"].eq("2025-03-05") & position["stock"].eq(spec["symbol"]), "shares"].gt(0).all()
    exit_event = next(event for event in report["chart"]["strategies"][0]["events"] if event["action"] == "SELL")
    assert exit_event["time"] == "2025-03-05" or exit_event["signal_date"] == "2025-03-05"
    assert all(event["side"] in {"BUY", "SELL"} for event in report["chart"]["strategies"][0]["events"])
    assert len(report["chart"]["bars"]) == 5
    assert report["ledger_metrics"]["trade_count"] == len(trades)
    assert report["source"] == identity
    shortened = expert_replay.run(run_id="short-result", spec={**spec, "start": "2025-03-05", "end": "2025-03-06"})
    sliced = json.loads(shortened.read_text(encoding="utf-8"))
    assert sliced["window"]["portfolio_anchor"] == "2025-03-03"
    assert sliced["window"]["opening_cash"] < 1_000_000
    assert [bar["time"] for bar in sliced["chart"]["bars"]] == ["2025-03-05", "2025-03-06"]
    for kind in ("daily", "positions"):
        full = pd.read_parquet(runs / f"interval-result/backtests/custom/fusion_normal_{kind}.parquet")
        narrow = pd.read_parquet(runs / f"short-result/backtests/custom/fusion_normal_{kind}.parquet")
        same = full.loc[full["date"].isin(narrow["date"])].reset_index(drop=True)
        pd.testing.assert_frame_equal(same.drop(columns="run_id"), narrow.drop(columns="run_id"))


def test_missing_dates_and_future_publication_cannot_pass(archive):
    _, spec = archive
    output = expert_replay.run(run_id="partial", spec={**spec, "start": "2025-02-28", "end": "2025-03-10"})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["coverage"] == {
        "requested_start": "2025-02-28", "requested_end": "2025-03-10",
        "market_start": "2025-02-28", "market_end": "2025-03-07",
        "model_start": "2025-03-03", "model_end": "2025-03-07",
    }
    assert report["ledger_scope"]["end"] == "2025-03-07"
    output = expert_replay.run(run_id="market-only", spec={**spec, "start": "2025-02-28", "end": "2025-03-02"})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "expert_market_only" and report["ledger_metrics"] is None
    assert len(report["chart"]["bars"]) == 1
    assert report["chart"]["strategies"][0]["events"] == []
    with pytest.raises(ValueError, match="候选研究"):
        expert_replay.replay_identity({**spec, "diagnostic": False})
    release = {
        **spec["release"], "status": "approved", "manifest_sha256": "b" * 64,
        "model_versions": [{"run_id": "expert", "fold": "oos_1", "valid_from": "2025-03-03",
                            "valid_through": "2025-03-04", "training_max_label_date": "2025-02-28",
                            "available_at": "2025-03-05T09:00:00+08:00"}],
    }
    with pytest.raises(ValueError, match="尚未发布"):
        expert_replay.replay_identity({**spec, "release": release})
    release["model_versions"][0]["available_at"] = "2025-03-01T09:00:00+08:00"
    with pytest.raises(ValueError, match="超出已有验证"):
        expert_replay.replay_identity({**spec, "release": release, "end": "2025-03-10"})


def test_external_rank_honors_positive_threshold_and_symbol_tie_break():
    days = pd.date_range("2025-03-03", periods=4)
    external = pd.DataFrame({"date": days, "stock": "000002.SZ", "score": [0.2, 0.2, -0.1, 0.3]})
    original = pd.DataFrame({
        "date": [days[0], days[1], days[2], days[3]],
        "stock": ["000001.SZ", "000003.SZ", "000003.SZ", "000003.SZ"], "score": [0.2, 0.2, -0.2, 0.4],
    })
    result = expert_stock_inference.original_pool_rank(external, original, top_n=1, min_score=0)
    assert result["pool_rank"].tolist() == [2, 1, 1, 2]
    assert result["execution_score"].tolist() == [0, 0.2, 0, 0]


def test_hybrid_replay_uses_decision_scores_without_fake_quantiles(archive):
    runs, original = archive
    expert = runs / "expert"
    report_path = expert / "reports/expert_research.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arm = "knn60_multimodal40"
    for fold in report["folds"]:
        period = fold["period"]
        fusion = pd.read_parquet(expert / f"backtests/{period}/fusion_predictions.parquet")
        mixed = fusion[["date", "stock", "score"]].copy()
        knn_rank = fusion["score"].map({-1: 0.5, 1: 1.0})
        multi_rank = 1.5 - knn_rank
        mixed["knn_q50"] = fusion["score"]
        mixed["multimodal_q50"] = -fusion["score"]
        for prefix in ("knn", "multimodal"):
            mixed[f"{prefix}_q10"] = mixed[f"{prefix}_q50"] - 1
            mixed[f"{prefix}_q90"] = mixed[f"{prefix}_q50"] + 1
            for quantile in ("q10", "q50", "q90"):
                mixed[f"{prefix}_raw_{quantile}"] = mixed[f"{prefix}_{quantile}"]
        mixed["knn_rank"], mixed["multimodal_rank"] = knn_rank, multi_rank
        mixed["hybrid_score"] = 0.6 * knn_rank + 0.4 * multi_rank
        mixed["score"] = mixed["hybrid_score"]
        mixed["effective_knn_weight"], mixed["effective_multimodal_weight"] = 0.6, 0.4
        mixed["fallback_reason"] = ""
        mixed.to_parquet(expert / f"backtests/{period}/{arm}_predictions.parquet", index=False)
        fold["arms"][arm] = {"execution": {"normal": {}, "double": {}}}
    _json(report_path, report)
    release = {"status": "candidate", "release_id": "hybrid-candidate", "source_run_id": "expert", "arm": arm}
    spec = original | {"release": release, "release_id": release["release_id"], "arm": arm}
    identity = expert_replay.replay_identity(spec)
    assert "expert_hybrid_study.py" not in identity["code_sha256"]
    output = expert_replay.run(run_id="hybrid-replay", spec=spec | {"input_identity": identity})
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ledger_metrics"]["trade_count"] > 0
    assert all(result["verification"].values())
    assert "KNN 60% + 多模态 40%" in result["chart"]["strategies"][0]["label"]
    saved = pd.read_parquet(runs / f"hybrid-replay/backtests/custom/{arm}_predictions.parquet")
    assert not {"q10", "q50", "q90"}.intersection(saved.columns)
    prediction = result["chart"]["strategies"][0]["predictions"][0]
    assert prediction["q50"] is None and prediction["score_semantics"] == "knn60_multimodal40_decision_rank"
    assert prediction["effective_knn_weight"] == 0.6


def test_configurable_replay_uses_verified_content_and_original_costs(archive, monkeypatch):
    runs, spec = archive
    store = ExpertArtifactStore(runs.parent / "content")
    monkeypatch.setattr(expert_replay, "ExpertArtifactStore", lambda: store)
    snapshots = runs / "base/data_snapshots"
    refs = {file.stem: store.put_file(file) for file in snapshots.glob("*.parquet")}
    store.write_manifest("expert", {f"frozen_inputs/{name}/{name}.parquet": ref for name, ref in refs.items()},
                         metadata={"task": "latent-expert-research", "source_run_id": "base"})

    config_path = runs / "expert/config/study.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rates = StudyCostModel(commission=0.0012)
    config.update(
        model_spec={"preset": "custom"}, sources={name: asdict(ref) for name, ref in refs.items()},
        expense_model=asdict(rates), execution_costs={cost: asdict(scaled_study_cost(value, rates))
                                                      for cost, value in config["cost_multipliers"].items()},
    )
    _json(config_path, config)
    output = expert_replay.run(run_id="custom-cost", spec={**spec, "cost": "double"})
    replay = json.loads(output.read_text(encoding="utf-8"))
    assert replay["source"]["execution_costs"]["commission"] == 0.0024
    trades = pd.read_parquet(runs / "custom-cost/backtests/custom/fusion_double_trades.parquet")
    assert len(trades) > 0
    for trade in trades.itertuples(index=False):
        side = "BUY" if trade.action == "BUY" else "SELL"
        assert trade.fee == pytest.approx(scaled_study_cost(2, rates).fees(side, trade.price * trade.shares))
    store.write_manifest("knn-expert", {f"frozen_inputs/{name}/{name}.parquet": ref for name, ref in refs.items()},
                         metadata={"task": "expert-knn-calibration-replay", "source_run_id": "model-parent"})
    config["model_source_run_id"] = "model-parent"
    files, digests = expert_replay._content_files(runs / "knn-expert", config, expert_replay.PANEL_FILES)
    assert set(files) == set(expert_replay.PANEL_FILES)
    assert digests == {f"{name}.parquet": ref.sha256 for name, ref in refs.items()}
    config["sources"]["panel"]["sha256"] = "f" * 64
    _json(config_path, config)
    with pytest.raises(ValueError, match="工件清单与配置"):
        expert_replay.replay_identity(spec)
