from dataclasses import asdict
from datetime import datetime
import json

import pandas as pd
import pytest

from my_strategy.latent_quant import forward, forward_models, forward_prepare
from my_strategy.latent_quant.execution import Strategy


def test_preparation_replays_sealed_execution_and_refuses_a_gap(tmp_path, monkeypatch):
    root = tmp_path / "forward"
    baseline = tmp_path / "baseline"
    (baseline / "config").mkdir(parents=True)
    (baseline / "data_snapshots").mkdir()
    arms = ["historical_quantiles", "supertrend_atr", "catboost_base", "catboost_supertrend_atr"]
    configuration = {
        "prospective": {"start_at_or_after": "2026-11-23", "market_sessions": 60},
        "arms": arms, "base_features": ["atr_pct"], "added_features": [],
        "cost_multipliers": {"normal": 1, "double": 2}, "portfolio": asdict(Strategy(rebalance_days=1)),
        "initial_cash": 1_000_000,
    }
    (baseline / "config/study.json").write_text(json.dumps(configuration), encoding="utf-8")
    pd.DataFrame([{"stock": "600000.SH", "board": "sh_main"}]).to_parquet(
        baseline / "data_snapshots/roster.parquet", index=False,
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_lock = model_dir / "baseline_models.json"
    model_lock.write_text(json.dumps({
        "first_signal": "2026-11-23", "input_cutoff": "2026-11-20",
        "sealed_at": "2026-11-23T09:00:00+08:00",
        "source_files": {
            "config": {"sha256": forward._sha(baseline / "config/study.json")},
            "roster": {"sha256": forward._sha(baseline / "data_snapshots/roster.parquet")},
        },
        "verification": {"four_arms_exact_replay": True},
    }), encoding="utf-8")
    now = [datetime.fromisoformat("2026-11-23T16:00:00+08:00")]
    monkeypatch.setattr(forward, "_now", lambda: now[0])
    day = now[0].date().isoformat()
    tables = {
        "calendar": [{"date": date, "is_trading_day": 1} for date in [day, "2026-11-24", "2026-11-25"]],
        "panel": [{"date": day, "stock": "600000.SH", "trade_open": 10, "trade_close": 10,
                   "open": 30, "close": 30, "amount": 20_000_000, "board": "sh_main",
                   "industry": "banks", "atr_pct": 0.04, "signal_eligible": True}],
        "benchmark": [{"date": day, "symbol": "000300.SH", "open": 4000, "close": 4010}],
        "announcements": [], "market_status": [],
    }
    bundle = tmp_path / "inputs.json"

    def write_inputs():
        inputs = {}
        for role, rows in tables.items():
            table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=forward.FIELDS[role])
            filename = f"{role}.csv"
            table.to_csv(tmp_path / filename, index=False)
            inputs[role] = {"path": filename, "source": "test provider", "observed_at": now[0].isoformat()}
        bundle.write_text(json.dumps({"trade_date": now[0].date().isoformat(), "inputs": inputs}), encoding="utf-8")

    write_inputs()

    def replay(frame, model_root):
        assert model_root == model_dir
        return {arm: pd.DataFrame({
            "q10": -0.01, "q50": 0.01, "q90": 0.05,
            "score": frame["date"].eq(pd.Timestamp(day)).map({True: 0.01, False: -0.01})
            if arm != "supertrend_atr" else -0.01,
        }, index=frame.index) for arm in arms}

    monkeypatch.setattr(forward_models, "replay_frozen_baseline", replay)
    args = {"models": model_dir, "root": root, "baseline": baseline, "calendar_path": None}
    original_lock = model_lock.read_bytes()
    bad_lock = json.loads(original_lock)
    bad_lock["sealed_at"] = "2026-11-23T15:00:00+08:00"
    model_lock.write_text(json.dumps(bad_lock), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed before the first session opens"):
        forward_prepare.prepare(bundle, **args)
    model_lock.write_bytes(original_lock)
    manifest = forward_prepare.prepare(bundle, **args)
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    orders = pd.read_parquet(manifest.parent / files["orders"]["file"])
    predictions = pd.read_parquet(manifest.parent / files["predictions"]["file"])
    assert set(predictions["arm"]) == set(arms)
    assert set(orders["arm"]) == set(arms) - {"supertrend_atr"}
    assert len(orders) == 6
    assert orders["requested_shares"].eq(9500).all()
    assert orders["date"].eq("2026-11-24").all()
    assert "filled_shares" not in orders
    assert forward._sha(model_lock) in files["predictions"]["source"]
    assert forward.status(root=root, baseline=baseline, calendar_path=None)["valid_daily_captures"] == 1
    with pytest.raises(FileExistsError, match="already sealed"):
        forward_prepare.prepare(bundle, **args)

    now[0] = datetime.fromisoformat("2026-11-24T16:00:00+08:00")
    tables["panel"][0].update(date="2026-11-24", trade_open=10.50, trade_close=10.50)
    tables["benchmark"][0].update(date="2026-11-24")
    write_inputs()
    with pytest.raises(ValueError, match="Missing previously sealed sessions: 2026-11-23"):
        forward_prepare.prepare(bundle, **{**args, "root": root / "gap"})
    changed = orders.assign(requested_shares=9400)
    order_file = manifest.parent / files["orders"]["file"]
    original_orders, original_manifest = order_file.read_bytes(), manifest.read_bytes()
    changed.to_parquet(order_file, index=False)
    record = json.loads(original_manifest)
    record["files"]["orders"]["sha256"] = forward._sha(order_file)
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Replayed orders differ"):
        forward_prepare.prepare(bundle, **args)
    order_file.write_bytes(original_orders)
    manifest.write_bytes(original_manifest)
    second = forward_prepare.prepare(bundle, **args)
    second_orders = pd.read_parquet(second.parent / "orders.parquet")
    assert len(second_orders) == 6
    assert second_orders["side"].eq("SELL").all()
    assert second_orders["requested_shares"].eq(9500).all()
    assert second_orders["date"].eq("2026-11-25").all()
    assert forward.status(root=root, baseline=baseline, calendar_path=None)["valid_daily_captures"] == 2
