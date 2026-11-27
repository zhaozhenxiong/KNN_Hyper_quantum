from datetime import date
import json

import pandas as pd
import pytest

from my_strategy.research_workbench import (
    expert_candidates, expert_personal_replay, expert_replay,
)
from my_strategy.tests.test_research_workbench_expert_replay import archive


def _aligned_scores(runs):
    for path in (runs / "expert/backtests").glob("oos_*/fusion_predictions.parquet"):
        frame = pd.read_parquet(path)
        frame["q50"] = frame["score"]
        frame["q10"] = frame["score"] - 0.2
        frame["q90"] = frame["score"] + 0.2
        frame.to_parquet(path, index=False)


def _spec(base, **changes):
    return {
        **base, "scope": "watchlist", "symbols": [], "model_id": "test-model",
        "model_version": "unified-test-version", "plan_next": True,
        "candidate_of": "parent-view", "end": "2025-03-05", **changes,
    }


def test_full_pool_plan_next_projects_pending_orders_without_fills(archive):
    runs, base = archive
    _aligned_scores(runs)
    expert_replay.run(run_id="parent-view", spec={**base, "scope": "watchlist", "symbols": [], "end": "2025-03-05"})
    output = expert_replay.run(run_id="candidate-view", spec=_spec(base))
    report = json.loads(output.read_text(encoding="utf-8"))
    root = runs / "candidate-view"
    pending = pd.read_parquet(root / "backtests/custom/full_pool_pending.parquet")
    assert report["source"]["plan_next"] is True
    assert report["source"]["candidate_of"] == "parent-view"
    assert pending["filled_shares"].fillna(0).eq(0).all()
    assert pending["execution_price"].isna().all()

    payload = expert_candidates.read_replay(root)
    assert payload["status"] == "research_only"
    assert payload["evidence_mode"] == "research_observation"
    assert payload["score_basis"] == "walk_forward_replay"
    assert payload["freshness"] == "stale"
    assert payload["model_version"] == "unified-test-version"
    assert payload["signal_date"] == "2025-03-05"
    assert payload["trade_date"] > payload["signal_date"]
    assert [row["stock"] for row in payload["candidates"]] == ["000001.SZ"], [
        {key: row.get(key) for key in ("stock", "side", "status", "requested_shares", "reason", "score")}
        for row in payload["orders"]
    ]
    assert payload["candidates"][0]["requested_shares"] > 0
    assert all(row["execution_price"] is None for row in payload["orders"])
    assert (root / "backtests/custom/candidates.csv").is_file()
    assert expert_personal_replay.authorized_download(report, "custom--candidates.csv") == "backtests/custom/candidates.csv"
    assert expert_personal_replay.authorized_download(report, "expert_candidates.json") == "reports/expert_candidates.json"

    with (root / "backtests/custom/candidates.csv").open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    with pytest.raises(ValueError, match="哈希"):
        expert_candidates.read_replay(root)


def test_empty_candidate_list_and_missing_next_session_are_explicit(archive, monkeypatch):
    runs, base = archive
    _aligned_scores(runs)
    output = expert_replay.run(run_id="empty-candidates", spec=_spec(base, initial_cash=1))
    report = json.loads(output.read_text(encoding="utf-8"))
    payload = expert_candidates.read_replay(runs / "empty-candidates")
    assert payload["candidates"] == []
    assert "未形成有效买入预申报" in payload["empty_reason"]
    assert pd.read_parquet(runs / "empty-candidates/backtests/custom/candidates.parquet").empty
    assert expert_personal_replay.authorized_download(report, "custom--candidates.parquet")

    monkeypatch.setattr(expert_personal_replay, "_next_session", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="下一交易日"):
        expert_replay.replay_identity(_spec(base))


def test_next_session_uses_verified_forward_calendar(tmp_path, monkeypatch):
    calendar = tmp_path / "calendar_20261231.csv"
    calendar.write_text(
        "date,is_trading_day\n2026-11-20,1\n2026-11-21,0\n2026-11-22,0\n2026-11-23,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(expert_personal_replay.latent_api, "PAPER_CALENDAR", calendar)
    monkeypatch.setattr(expert_personal_replay.forward, "CALENDAR", tmp_path / "missing.csv")

    assert expert_personal_replay._next_session(date(2026, 11, 20)) == "2026-11-23"
