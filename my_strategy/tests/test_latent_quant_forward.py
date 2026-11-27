from dataclasses import replace
from datetime import date, datetime, time
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.latent_quant import forward


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def study(tmp_path, monkeypatch):
    baseline = tmp_path / "prior"
    (baseline / "config").mkdir(parents=True)
    (baseline / "data_snapshots").mkdir()
    _write_json(baseline / "config/study.json", {
        "prospective": {"start_at_or_after": "2026-11-23", "market_sessions": 7},
        "arms": ["simple", "model"],
        "base_features": ["atr_pct"],
        "added_features": [],
        "cost_multipliers": {"normal": 1},
    })
    pd.DataFrame([{"stock": "600000.SH", "board": "sh_main"}]).set_index("stock").to_parquet(
        baseline / "data_snapshots/roster.parquet",
    )
    root = tmp_path / "new"
    now = [datetime.fromisoformat("2026-11-23T16:00:00+08:00")]
    monkeypatch.setattr(forward, "_now", lambda: now[0])
    return baseline, root, now


def _bundle(tmp_path: Path, day: date, next_day: date, days: pd.DatetimeIndex) -> Path:
    path = tmp_path / str(day)
    path.mkdir()
    values = {
        "calendar": [{"date": str(d.date()), "is_trading_day": int(d.weekday() < 5)} for d in days],
        "panel": [{"date": str(day), "stock": "600000.SH", "trade_open": 10, "trade_close": 11,
                   "open": 30, "close": 33, "amount": 1_000_000, "board": "sh_main",
                   "industry": "unknown", "signal_eligible": True, "atr_pct": 0.04}],
        "benchmark": [{"date": str(day), "symbol": "000300.SH", "open": 4000, "close": 4010}],
        "market_status": [],
        "announcements": [],
        "predictions": [{"date": str(day), "stock": "600000.SH", "arm": arm,
                         "score": 0.01, "q10": -0.04, "q50": 0.01, "q90": 0.05} for arm in ["simple", "model"]],
        "orders": [{"signal_date": str(day), "date": str(next_day), "stock": "600000.SH", "arm": "model",
                    "cost": "normal", "side": "BUY", "requested_shares": 100}],
    }
    bundle = path / "bundle.json"
    _write_json(bundle, {
        "trade_date": str(day),
        "inputs": {name: {"path": f"{name}.csv", "observed_at": f"{day}T15:30:00+08:00", "source": "fixture"}
                   for name in values},
    })
    for name, rows in values.items():
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=forward.FIELDS[name])
        frame.to_csv(path / f"{name}.csv", index=False)
    return bundle


def _capture(bundle, study):
    baseline, root, _ = study
    return forward.capture(bundle, baseline=baseline, root=root, calendar_path=None)


def _status(study):
    baseline, root, _ = study
    return forward.status(baseline=baseline, root=root, calendar_path=None)


def test_read_only_preclose_and_same_day_guards(study, tmp_path):
    _, root, now = study
    now[0] = datetime.fromisoformat("2026-11-23T10:15:00+08:00")
    state = _status(study)
    assert state["status"] == "awaiting_market_close"
    assert state["daily_captures_remaining"] == 7
    assert state["calendar_needed"]
    assert not root.exists()
    with pytest.raises(ValueError, match="current Shanghai date"):
        _capture(tmp_path / "missing.json", study)
    now[0] = datetime.fromisoformat("2026-11-24T16:15:00+08:00")
    with pytest.raises(ValueError, match="current Shanghai date"):
        forward.capture(tmp_path / "missing.json", trade_date=date(2026, 11, 23))
    now[0] = datetime.fromisoformat("2026-11-24T18:01:00+08:00")
    with pytest.raises(ValueError, match="current Shanghai date"):
        _capture(tmp_path / "missing.json", study)
    assert not root.exists()


def test_successful_capture_is_hashed_immutable_and_uses_actual_time(study, tmp_path):
    _, root, now = study
    days = pd.date_range("2026-11-23", periods=20)
    bundle = _bundle(tmp_path, date(2026, 11, 23), date(2026, 11, 24), days)
    manifest = _capture(bundle, study)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["captured_at"] == now[0].isoformat()
    assert record["producer_bundle_sha256"] == forward._sha(bundle)
    assert record["next_trade_date"] == "2026-11-24"
    assert len(record["files"]) == len(forward.FIELDS) + 2
    assert _status(study)["valid_daily_captures"] == 1
    with pytest.raises(FileExistsError):
        _capture(bundle, study)
    (manifest.parent / "panel.csv").write_text("invalid data", encoding="utf-8")
    state = _status(study)
    assert state["status"] == "missed_or_invalid_capture"
    assert "hash verification" in state["invalid_captures"]["2026-11-23"]
    assert not state["ready_for_scoring"]
    assert root.exists()


@pytest.mark.parametrize(("mutation", "message"), [
    ("late_input", "Unverified evidence"),
    ("target", "future outcome"),
    ("executed", "omit next-session"),
    ("missing", "Incomplete frozen roster"),
])
def test_forward_capture_rejects_lookahead_or_missing_evidence(study, tmp_path, mutation, message):
    _, root, _ = study
    days = pd.date_range("2026-11-23", periods=9)
    bundle = _bundle(tmp_path, date(2026, 11, 23), date(2026, 11, 24), days)
    if mutation == "late_input":
        config = json.loads(bundle.read_text())
        config["inputs"]["panel"]["observed_at"] = "2026-11-23T16:01:00+08:00"
        _write_json(bundle, config)
    elif mutation in {"target", "missing"}:
        panel = bundle.parent / "panel.csv"
        table = pd.read_csv(panel)
        if mutation == "target":
            table["target"] = 0.05
        else:
            table = table.iloc[0:0]
        table.to_csv(panel, index=False)
    else:
        orders = bundle.parent / "orders.csv"
        pd.read_csv(orders).assign(execution_price=11).to_csv(orders, index=False)
    with pytest.raises(ValueError, match=message):
        _capture(bundle, study)
    assert not list((root / "captures").iterdir())


def test_optional_native_features_accept_nan_but_reject_infinity(study, tmp_path, monkeypatch):
    baseline, root, now = study
    policy = forward._policy(baseline)
    optional = replace(
        policy, features=(*policy.features, "sector_market"), optional_features=frozenset({"sector_market"}),
    )
    monkeypatch.setattr(forward, "_policy", lambda _: optional)
    days = pd.date_range("2026-11-23", periods=20)
    bundle = _bundle(tmp_path, date(2026, 11, 23), date(2026, 11, 24), days)
    panel_path = bundle.parent / "panel.csv"
    pd.read_csv(panel_path).assign(sector_market=float("nan")).to_csv(panel_path, index=False)
    assert _capture(bundle, study).is_file()

    now[0] = datetime.fromisoformat("2026-11-24T16:00:00+08:00")
    bundle = _bundle(tmp_path, date(2026, 11, 24), date(2026, 11, 25), days)
    panel_path = bundle.parent / "panel.csv"
    pd.read_csv(panel_path).assign(sector_market=float("inf")).to_csv(panel_path, index=False)
    with pytest.raises(ValueError, match="finite frozen feature"):
        _capture(bundle, study)


def test_continuous_capture_readiness_with_maturity_separate(study, tmp_path):
    baseline, root, now = study
    days = pd.date_range("2026-11-23", periods=20)
    sessions = days[days.day_of_week < 5]
    for current, nxt in zip(sessions[:7], sessions[1:8], strict=True):
        day = current.date()
        now[0] = datetime.combine(day, time(16), tzinfo=forward.SHANGHAI)
        _capture(_bundle(tmp_path, day, nxt.date(), days), study)
    report = _status(study)
    assert report["ready_for_scoring"]
    assert report["acceptance_window_end"] == sessions[6].date()
    assert report["all_signal_labels_end_if_known"] == sessions[12].date()
    assert report["max_main_scoring_signal_sessions"] == 1
    after_gap = forward.status(baseline=baseline, root=root / "missing", calendar_path=(
        tmp_path / "2026-11-23" / "calendar.csv"
    ))
    assert len(after_gap["missing_past_captures"]) == 6
    assert not after_gap["ready_for_scoring"]
