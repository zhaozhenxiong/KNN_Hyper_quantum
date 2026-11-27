from dataclasses import asdict
from datetime import datetime, time
import json

import pandas as pd
import pytest

from my_strategy.core import paths
from my_strategy.latent_quant import forward, models
from my_strategy.latent_quant.execution import Strategy, StudyCostModel
from my_strategy.model_registry import expert_release
from my_strategy.model_registry.registry import ModelRegistry
from my_strategy.research_workbench import expert_api, expert_forward, expert_replay, latent_api, stock_chart_api
from my_strategy.research_workbench.runner import write_json


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "expert-candidate"
    monkeypatch.setattr(latent_api, "RUNS", root.parent)
    monkeypatch.setattr(expert_release, "ARTIFACT_RUNS_ROOT", root.parent)
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", root.parent)
    clock = {"now": datetime(2025, 1, 3, 16, tzinfo=forward.SHANGHAI)}
    monkeypatch.setattr(forward, "_now", lambda: clock["now"])
    calendar = pd.date_range("2025-01-01", "2025-04-30")
    calendar_path = tmp_path / "calendar.csv"
    pd.DataFrame({"date": calendar, "is_trading_day": (calendar.day_of_week < 5).astype(int)}).to_csv(calendar_path, index=False)
    dates = pd.bdate_range("2025-01-06", periods=61).date.tolist()
    roster = tmp_path / "roster.parquet"
    pd.DataFrame({"stock": ["000001.SZ"], "board": ["sz_main"], "delist_date": [pd.NaT]}).to_parquet(roster)
    monkeypatch.setattr(expert_replay, "_data_files", lambda *_: ({"roster.parquet": roster}, {"roster.parquet": forward._sha(roster)}))
    write_json(root / "status.json", {"status": "completed"})
    report = root / expert_release.REPORT
    write_json(report, {"run": {"run_id": root.name}, "final_portfolio": {"arm": "domestic_gpu"},
                        "decision": {"historical_numerical_gate": True, "point_in_time_verified": True}})
    write_json(root / "config/study.json", {
        "end": "2025-01-03", "initial_cash": 1_000_000, "seed": 42, "features": {"domestic_gpu": ["example"]},
        "cost_multipliers": {"normal": 1, "double": 2}, "expense_model": asdict(StudyCostModel()),
        "portfolio": asdict(Strategy(top_n=1)),
    })
    files = ["models/forward/train.parquet", *(f"models/forward/domestic_gpu/{q}.cbm" for q in ("q10", "q50", "q90"))]
    for file in files:
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_bytes(file.encode("ascii"))
    write_json(root / expert_forward.SCORER, {
        "schema_version": 1, "run_id": root.name, "arm": "domestic_gpu", "research_report_sha256": forward._sha(report),
        "point_in_time_verified": True, "sealed_at": clock["now"].isoformat(), "training_max_label_date": "2025-01-02",
        "training_input_through": "2025-01-03", "weights": {"domestic_gpu": 1}, "features": {"domestic_gpu": ["example"]},
        "offsets": {"domestic_gpu": [0, 0, 0], "historical_quantiles": [0, 0, 0]},
        "files": {file: forward._sha(root / file) for file in files},
    })

    def forecast(locked, panel, stamp):
        keys = panel[["date", "stock"]]
        return pd.concat([
            keys.assign(arm="domestic_gpu", q10=0.035, q50=0.04, q90=0.05, score=0.04),
            keys.assign(arm="historical_quantiles", q10=-0.09, q50=-0.05, q90=-0.01, score=-0.05),
        ], ignore_index=True)

    monkeypatch.setattr(expert_forward, "_forecast", forecast)

    def bundle(index, *, contaminated=False):
        day = dates[index]
        clock["now"] = datetime.combine(day, time(16), tzinfo=forward.SHANGHAI)
        raw = 20 * 1.008 ** index
        frames = {
            "panel": pd.DataFrame({
                "date": [str(day)], "stock": ["000001.SZ"], "open": [raw], "close": [raw * 1.004],
                "trade_open": [raw], "trade_close": [raw * 1.004], "volume": [2e6], "amount": [2e7],
                "trade_high": [raw * 1.008], "trade_low": [raw * 0.995], "source": ["fixture"],
                "board": ["sz_main"], "industry": ["bank"], "signal_eligible": [True], "example": [0.5],
            }),
            "benchmark": pd.DataFrame({"date": [str(day)], "symbol": ["000300.SH"], "open": [4000.0], "close": [4000.0]}),
            "market_status": pd.DataFrame({"date": [str(day)], "stock": ["000001.SZ"], "market_status": ["tradable"], "source": ["fixture"]}),
            "announcements": pd.DataFrame(columns=["published_at", "source"]),
        }
        if contaminated:
            frames["panel"]["target"] = 0.5
        inputs = {"calendar": calendar_path}
        for role, table in frames.items():
            path = tmp_path / f"daily-{role}.parquet"
            table.to_parquet(path, index=False)
            inputs[role] = path
        result = tmp_path / "bundle.json"
        write_json(result, {"trade_date": str(day), "inputs": {
            role: {"path": path.name, "sha256": forward._sha(path), "source": "fixture", "observed_at": clock["now"].isoformat()}
            for role, path in inputs.items()
        }})
        return result

    return root, clock, dates, calendar_path, bundle


def test_model_features_allow_only_nonrequired_native_model_inputs_to_be_missing(study):
    model = {
        "weights": {"domestic_gpu": 0.5, "sector_gpu": 0.2, "overseas_gpu": 0.2, "supertrend_atr": 0.1},
        "features": {
            "domestic_gpu": ["shared", "domestic_only"],
            "sector_gpu": ["shared", "sector_only"],
            "overseas_gpu": ["overseas_only"],
        },
    }
    assert expert_forward._features(model) == (
        "shared", "domestic_only", "sector_only", "overseas_only", *models._TREND_COLUMNS,
    )
    assert expert_forward._optional_features(model) == {"sector_only", "overseas_only"}


def test_outcomes_and_late_or_skipped_days_fail_closed(study):
    root, clock, dates, calendar, bundle = study
    assert expert_forward.status(root.name, calendar_path=calendar)["state"] == "ready_to_lock"
    expert_forward.lock(root.name, dates[0], calendar_path=calendar)
    with pytest.raises(ValueError, match="future outcome"):
        expert_forward.capture(root.name, bundle(0, contaminated=True), calendar_path=calendar)
    clock["now"] = datetime.combine(dates[0], time(18, 1), tzinfo=forward.SHANGHAI)
    with pytest.raises(ValueError, match="15:00"):
        expert_forward.capture(root.name, root / "unused.json", calendar_path=calendar)
    with pytest.raises(ValueError, match="下一待封存"):
        expert_forward.capture(root.name, bundle(1), calendar_path=calendar)
    assert expert_forward.status(root.name, calendar_path=calendar)["state"] == "missed_capture"
    with pytest.raises(ValueError, match="0/60"):
        expert_forward.accept(root.name, calendar_path=calendar)
    assert not (root / expert_release.MANIFEST).exists()


def test_verified_60_day_release_and_tamper_gate(study, tmp_path, monkeypatch):
    root, clock, dates, calendar, bundle = study
    release_registry = ModelRegistry(tmp_path / "registry.sqlite", snapshot_path=None)
    candidate = expert_release.record_candidate(root.name, root / expert_release.REPORT, release_registry)
    assert candidate.params["arm"] == "domestic_gpu"
    expert_forward.lock(root.name, dates[0], calendar_path=calendar)
    for index in range(60):
        expert_forward.capture(root.name, bundle(index), calendar_path=calendar)
    assert expert_forward.status(root.name, calendar_path=calendar)["state"] == "ready_for_acceptance"
    report_path = expert_forward.accept(root.name, calendar_path=calendar)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["qualified_forward_results"] is True
    assert len(report["sessions"]) == 60 and report["max_scoring_signal_sessions"] == 54
    assert report["run"]["artifact_root"] == str(root / expert_forward.ROOT / "acceptance")
    assert expert_forward.status(root.name, calendar_path=calendar)["state"] == "qualified"
    assert expert_release.chart_release(release_registry)["status"] == "unavailable"
    expert_release.promote_verified(root.name, release_registry)
    published = expert_release.chart_release(release_registry)
    assert published["status"] == "approved" and published["arm"] == "domestic_gpu"
    assert published["model_versions"][0]["fold"] == "expert_forward"
    current = expert_release.current_strategy(release_registry)
    assert (current["status"], current["run_id"], current["model_id"]) == ("approved", root.name, candidate.model_id)
    assert current["as_of"] == str(dates[-2]) and current["publication"] == published
    monkeypatch.setattr(expert_release, "default_registry", lambda: release_registry)
    monkeypatch.setattr(stock_chart_api, "_security", lambda symbol: {"symbol": symbol, "name": "Fixture"})
    later = pd.DataFrame({"date": [str(dates[-1])], "stock": ["000001.SZ"], "trade_open": [30.0], "trade_close": [30.1],
                          "trade_high": [30.3], "trade_low": [29.8], "volume": [1000], "source": ["raw"]})
    monkeypatch.setattr(stock_chart_api, "_warehouse_series", lambda *args, **kwargs: (later, pd.DataFrame(), None, None, None))
    spec = {"release": published, **{key: published[key] for key in ("release_id", "source_run_id", "arm")},
            "symbol": "000001.SZ", "cost": "double", "start": str(dates[0]), "end": str(dates[-1])}
    identity = expert_replay.replay_identity(spec)
    replay = expert_replay.run(run_id="approved-replay", spec={**spec, "input_identity": identity})
    replayed = json.loads(replay.read_text(encoding="utf-8"))
    chart = replayed["chart"]
    assert chart["bars"][-1]["time"] == str(dates[-1]) and chart["strategies"][0]["events"]
    assert len(chart["strategies"][0]["predictions"]) == 60
    assert all(date["time"] <= str(dates[-2]) for date in chart["strategies"][0]["events"])
    accepted_file = root / expert_forward.ROOT / "acceptance/domestic_gpu_double_trades.parquet"
    assert forward._sha(accepted_file) == forward._sha(replay.parent.parent / "backtests/custom/domestic_gpu_double_trades.parquet")
    interval = {"start": str(dates[0]), "end": str(dates[4]), "cost": "normal"}
    for field, symbols, account in (("symbol", "000001.SZ", "single_stock"), ("symbols", ["000001.SZ"], "watchlist")):
        personal = expert_api._current_replay_spec(interval | {field: symbols}, single=field == "symbol")
        assert personal["release"]["status"] == "approved" and personal["model_id"] == candidate.model_id
        personal["input_identity"] = expert_replay.replay_identity(personal)
        result = expert_replay.run(run_id=f"approved-{account}", spec=personal)
        payload = json.loads(result.read_text(encoding="utf-8"))
        assert payload["source"]["approval"] == "approved"
        assert payload["accounts"][account]["verification"]["next_session_verified"] is True
    (root / expert_forward.ROOT / "captures" / str(dates[0]) / "panel.parquet").write_text("tampered", encoding="utf-8")
    assert expert_release.chart_release(release_registry)["status"] == "unavailable"
