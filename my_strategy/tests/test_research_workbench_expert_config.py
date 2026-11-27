"""Expert cohorts, rolling preflight and end-bounded feature evidence."""

from contextlib import closing
from dataclasses import replace
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.latent_quant.data import BASE_FEATURES, EXTRA_FEATURES
from my_strategy.research_workbench import expert_config as config
from my_strategy.research_workbench import expert_full_data as full_data
from my_strategy.research_workbench.expert_full_data import freeze_expert_full_inputs, read_expert_full_freeze
from my_strategy.tests.test_latent_quant_data import frozen_inputs


def test_normalize_keeps_original_preset_and_canonicalizes_stock_codes():
    original = config.normalize_expert_spec()
    assert original["preset"] == "frozen_v2"
    assert (original["pool"], original["stock_cap"]) == ("frozen_120", 120)
    assert config.normalize_expert_spec(original) == original
    manual = config.normalize_expert_spec({
        "pool": "manual", "symbols": ["600000.sh", " 002436.sz ", "600000.SH"], "commission_bps": "1.5",
    })
    assert manual["symbols"] == ["002436.SZ", "600000.SH"]
    assert (manual["preset"], manual["stock_cap"], manual["commission_bps"]) == ("custom", 2, "1.5")


@pytest.mark.parametrize("options", [
    {"end": "2026-11-23"}, {"start": "2025-3-03"}, {"start": "2026-02-30"},
    {"pool": "board", "boards": ["gem"]}, {"pool": "manual", "symbols": []},
    {"pool": "manual", "symbols": ["../../stock"]},
    {"pool": "manual", "symbols": ["002436.SZ", "600000.SH"], "stock_cap": 1},
    {"pool": "frozen_120", "stock_cap": 4}, {"seed": True}, {"raw_db": "custom.sqlite"},
    {"pool": "full", "universe_policy": "bse"}, {"pool": "frozen_120", "universe_policy": "shsz"},
])
def test_normalize_rejects_unsupported_or_ambiguous_requests(options):
    with pytest.raises(config.ExpertSpecError):
        config.normalize_expert_spec(options)


requires_snapshot = pytest.mark.skipif(
    not all(path.is_file() for path in config._frozen_paths().values()) or not config._ARCHIVE["calendar"].is_file(),
    reason="requires the verified workspace snapshot",
)
requires_archive = pytest.mark.skipif(
    not config._RAW.is_file() or not all(path.is_file() for path in config._ARCHIVE.values()),
    reason="requires the verified workspace archive",
)


@requires_snapshot
def test_frozen_preflight_preserves_three_original_windows_and_reports_maturity():
    preview = config.expert_preflight()
    assert preview["status"] == "ready"
    assert preview["selected_stocks"] == 120
    assert preview["board_counts"] == {"sh_main": 40, "star": 40, "sz_main": 40}
    assert preview["anchor_session"] == "2023-12-29"
    assert [window["evaluation_start"] for window in preview["windows"]] == ["2025-03-03", "2025-10-09", "2026-04-01"]
    assert all(window["maturity"]["basis"] == "eligible_mature_labels" and
               window["maturity"]["train_sessions"] >= 100 for window in preview["windows"])
    one_day = config.expert_preflight({"start": "2026-07-27", "end": "2026-07-27"})
    assert one_day["status"] == "blocked"
    assert one_day["evaluation_sessions"] == 1
    assert "90" in one_day["blockers"][0]
    weekend = config.expert_preflight({"start": "2026-07-25", "end": "2026-07-26"})
    assert weekend["status"] == "blocked"
    assert weekend["first_trade_session"] is None


@requires_snapshot
def test_frozen_subset_preserves_features_and_excludes_future_label_prices():
    spec = {"start": "2025-03-02", "end": "2026-07-26"}
    bounded = config.prepare_expert_inputs(spec)
    cutoff = pd.Timestamp("2026-07-24")
    assert bounded.sessions[0] == pd.Timestamp("2024-01-02")
    assert bounded.sessions[-1] == cutoff
    assert bounded.coverage["first_evaluation_session"] == "2025-03-03"
    assert bounded.frame.loc[bounded.frame["target"].notna(), "label_end"].le(cutoff).all()
    assert bounded.frame["trade_date"].dropna().le(cutoff).all()
    assert bounded.market_status["date"].le(cutoff).all()
    columns = ["stock", "date", "signal_eligible", *BASE_FEATURES, *EXTRA_FEATURES]
    original = pd.read_parquet(config._frozen_paths()["panel.parquet"], columns=columns,
                               filters=[("date", "<=", cutoff)]).sort_values(["date", "stock"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(bounded.frame[columns], original)
    assert config.expert_input_identity(spec) == config.expert_input_identity(spec | {"commission_bps": "1.5"})
    assert bounded.coverage["input_identity"] == config.expert_input_identity(spec)


@requires_snapshot
@requires_archive
def test_archive_preflight_finds_outside_stock_and_distinguishes_evidence_limits():
    spec = {"pool": "manual", "symbols": ["002436.SZ"], "start": "2026-04-01", "end": "2026-07-28"}
    preview = config.expert_preflight(spec)
    assert preview["selected_symbols"] == ["002436.SZ"]
    assert preview["coverage"]["by_symbol"] == {"002436.SZ": 80}
    assert preview["status"] == "blocked"  # 80 actual trading dates cannot provide three 30-day folds.
    data = config.prepare_expert_inputs(spec)
    assert data.roster.index.tolist() == ["002436.SZ"]
    assert data.coverage["input_identity"]["source_kind"] == "verified_local_archive"
    assert data.coverage["coverage_by_symbol"]["002436.SZ"]["evaluation_eligible_days"] == 80
    assert np.isfinite(data.frame.loc[data.frame["signal_eligible"], [*BASE_FEATURES, *EXTRA_FEATURES]]).all().all()
    assert data.frame["label_end"].dropna().le(pd.Timestamp(spec["end"])).all()
    excluded = config.expert_preflight(spec | {"symbols": ["300750.SZ"]})
    assert excluded["status"] == "blocked"
    assert excluded["rejections"] == {"300750.SZ": "缺少同口径交易所名单"}


@requires_snapshot
@requires_archive
def test_board_preflight_respects_seed_and_trading_calendar():
    spec = {"pool": "board", "boards": ["star", "sz_main"], "stock_cap": 4, "seed": 42,
            "start": "2025-03-02", "end": "2026-07-26"}
    preview = config.expert_preflight(spec)
    assert preview["selected_symbols"] == config.expert_preflight(spec)["selected_symbols"]
    assert preview["board_counts"] == {"star": 2, "sz_main": 2}
    assert (preview["first_trade_session"], preview["last_trade_session"]) == ("2025-03-03", "2026-07-24")
    assert all(window["maturity"]["basis"] == "adjusted_coverage_upper_bound" for window in preview["windows"])


def test_full_pool_accepts_no_sampling_and_defaults_to_verified_cutoff():
    spec = config.normalize_expert_spec({"pool": "full"})
    assert spec["stock_cap"] is None
    assert (spec["start"], spec["end"]) == ("2025-03-03", "2026-11-20")
    assert config.normalize_expert_spec(spec) == spec
    assert config.normalize_expert_spec(spec | {"universe_policy": "main_star"}) == spec
    assert "universe_policy" not in spec  # Legacy hashes already mean main_star.
    assert config.universe_policy(spec) == "main_star"
    assert config.universe_scope_label(spec) == "沪深主板+科创板"
    shsz = config.normalize_expert_spec(spec | {"universe_policy": "shsz"})
    assert shsz["universe_policy"] == "shsz"
    assert config.universe_scope_label(shsz) == "沪深主板+创业板+科创板"
    for settings in ({"stock_cap": 1000}, {"boards": ["star"]}, {"symbols": ["002436.SZ"]}):
        with pytest.raises(config.ExpertSpecError):
            config.normalize_expert_spec({"pool": "full", **settings})


def _policy_fixture(tmp_path, monkeypatch, *, with_chinext=False):
    dates = pd.bdate_range("2024-01-02", periods=40)
    stocks = ["600000.SH", *(["300001.SZ"] if with_chinext else [])]
    roster = pd.DataFrame({
        "stock": stocks, "exchange": [stock[-2:] for stock in stocks],
        "board": ["sh_main", *(["chinext"] if with_chinext else [])],
        "list_date": pd.Timestamp("2020-01-02"), "delist_date": pd.NaT,
        "source": "fixture", "source_url": "https://example.invalid/roster",
    }).set_index("stock")
    archives = {name: tmp_path / f"{name}.csv" for name in (
        "roster", "calendar", "benchmark", "classification", "status_roster", "status_updates", "status_star",
    )}
    roster.reset_index().to_csv(archives["roster"], index=False)
    for name in ("calendar", "benchmark"):
        archives[name].write_text("fixture\n", encoding="utf-8")
    pd.DataFrame({
        "stock": stocks, "effective_from": "2020-01-02", "available_at": "2020-01-02T00:00:00+08:00",
        "source": "fixture", "version": "v1", "industry": "J66",
    }).to_csv(archives["classification"], index=False)
    status = pd.DataFrame(columns=["stock", "date", "market_status", "source", "source_url"])
    for name in ("status_roster", "status_updates", "status_star"):
        status.to_csv(archives[name], index=False)
    adjusted = pd.DataFrame({
        "stock": stocks, "date": [dates.date for _ in stocks],
    }).explode("date").reset_index(drop=True)
    archives["adjusted"] = tmp_path / "adjusted.parquet"
    adjusted.to_parquet(archives["adjusted"], index=False)
    reference = adjusted.copy()
    reference["date"] = pd.to_datetime(reference["date"])
    for name in ("open", "high", "low", "close"):
        reference[name] = 10.0
    reference["volume"] = 1000
    reference["amount"] = 10000
    archives["reference"] = tmp_path / "reference.parquet"
    reference.to_parquet(archives["reference"], index=False)
    raw = tmp_path / "raw.sqlite"
    with closing(sqlite3.connect(raw)) as connection, connection:
        connection.execute(
            "CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,has_trade_price INTEGER,open REAL,volume REAL)",
        )
        connection.executemany(
            "INSERT INTO stock_daily_normalized VALUES (?,?,1,10,1000)",
            [(stock, str(day.date())) for stock in stocks for day in dates],
        )
    spec = config.normalize_expert_spec({
        "pool": "full", "start": str(dates[20].date()), "end": str(dates[-1].date()), "universe_policy": "shsz",
    })
    scope = config._Scope(spec, dates, dates[5:], dates[20:], roster, roster, None, False, {})
    monkeypatch.setattr(config, "_ARCHIVE", archives)
    monkeypatch.setattr(config, "_RAW", raw)
    monkeypatch.setattr(config, "_scope", lambda payload: replace(scope, spec=config.normalize_expert_spec(payload)))
    monkeypatch.setattr(config, "_maturity", lambda *args: ([], []))
    monkeypatch.setattr(config, "_read_raw", lambda scope: pd.DataFrame())
    return scope


def test_explicit_policy_identities_and_versioned_destinations(tmp_path, monkeypatch):
    _policy_fixture(tmp_path, monkeypatch)
    request = {"pool": "full", "start": "2024-02-01", "end": "2024-02-27"}
    main = config.expert_input_identity(request)
    assert main == config.expert_input_identity(request | {"universe_policy": "main_star"})
    assert config.universe_policy(main) == "main_star"
    shsz = config.expert_input_identity(request | {"universe_policy": "shsz"})
    assert shsz["universe_policy"] == "shsz"
    assert stable_hash(shsz) != stable_hash(main)
    assert full_data.full_freeze_destination(request) == full_data.DEFAULT_FULL_FREEZE
    assert full_data.full_freeze_destination(request | {"universe_policy": "shsz"}) == full_data.SHSZ_FULL_FREEZE


def test_shsz_preflight_blocks_missing_chinext_and_never_creates_freeze(tmp_path, monkeypatch):
    _policy_fixture(tmp_path, monkeypatch)
    request = {"pool": "full", "start": "2024-02-01", "end": "2024-02-27", "universe_policy": "shsz"}
    report = config.expert_preflight(request)
    assert report["status"] == "blocked"
    assert report["scope_label"] == "沪深主板+创业板+科创板"
    assert report["board_counts"] == {"sh_main": 1}
    assert report["membership"]["boards"] == ["chinext", "sh_main", "star", "sz_main"]
    assert [blocker["code"] for blocker in report["blocker_details"]] == ["missing_chinext_roster"]
    assert report["policy_evidence"]["roster_chinext_stocks"] == 0

    def forbid_prepare(spec):
        pytest.fail("blocked SHSZ preflight must not build inputs")

    monkeypatch.setattr(config, "prepare_expert_inputs", forbid_prepare)
    destination = tmp_path / "blocked-freeze"
    with pytest.raises(config.ExpertSpecError) as failure:
        freeze_expert_full_inputs(request, destination)
    assert failure.value.blockers[0]["code"] == "missing_chinext_roster"
    assert not destination.exists()


def test_shsz_preflight_accepts_aligned_small_fixture(tmp_path, monkeypatch):
    _policy_fixture(tmp_path, monkeypatch, with_chinext=True)
    report = config.expert_preflight({"pool": "full", "start": "2024-02-01", "end": "2024-02-27",
                                      "universe_policy": "shsz"})
    assert report["status"] == "ready"
    assert report["board_counts"] == {"chinext": 1, "sh_main": 1}
    assert report["policy_evidence"]["aligned_chinext_stocks"] == 1
    assert not report["blocker_details"]


def test_legacy_and_explicit_full_manifest_policies(tmp_path):
    scope = {"pool": "full", "start": "2025-03-03", "end": "2026-11-20"}
    identity = {"pool": "full", "membership": config.FULL_MEMBERSHIP, "input_cutoff": "2026-11-20"}
    manifest = {"schema_version": 1, "spec": scope, "input_identity": identity,
                "input_version": stable_hash(identity), "files": dict.fromkeys(full_data._NAMES, {})}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert full_data.read_expert_full_manifest(scope, tmp_path) == manifest
    with pytest.raises(config.ExpertSpecError):
        full_data.read_expert_full_manifest(scope | {"universe_policy": "shsz"}, tmp_path)

    shsz_scope = scope | {"universe_policy": "shsz"}
    shsz_identity = identity | {"universe_policy": "shsz"}
    shsz_manifest = manifest | {"spec": shsz_scope, "input_identity": shsz_identity,
                                "input_version": stable_hash(shsz_identity)}
    (tmp_path / "manifest.json").write_text(json.dumps(shsz_manifest), encoding="utf-8")
    assert full_data.read_expert_full_manifest(shsz_scope, tmp_path) == shsz_manifest
    with pytest.raises(config.ExpertSpecError):
        full_data.read_expert_full_manifest(scope, tmp_path)


@requires_archive
def test_full_preflight_uses_daily_membership_and_new_ipos_without_frozen_120(monkeypatch):
    def fail_if_called():
        pytest.fail("full universe depends only on official archive inputs")

    monkeypatch.setattr(config, "_frozen_paths", fail_if_called)
    report = config.expert_preflight({"pool": "full"})
    assert report["status"] == "partial"
    assert report["anchor_session"] is None
    assert report["membership"] == {
        "basis": "daily_exchange_list_delist", "boards": ["sh_main", "star", "sz_main"],
        "historical_members": 3902, "listed_after_input_start": 139,
        "delisted_by_cutoff": 89, "active_on_cutoff": 3813,
    }
    assert sum(window["evaluation_sessions"] for window in report["windows"]) == 420
    assert not report["blockers"]


def test_full_freeze_daily_lifetimes_and_future_price_invariance(frozen_inputs, tmp_path, monkeypatch):
    sourced, dates = frozen_inputs
    classification = tmp_path / "industry.csv"
    classification.write_text(
        f"stock,effective_from,available_at,source,version,industry\n600000.SH,{dates[34].date()},"
        f"{dates[34].date()},fixture,v1,J66\n", encoding="utf-8",
    )
    archives = {
        "roster": sourced["roster_path"], "calendar": sourced["calendar_path"], "benchmark": sourced["benchmark_path"],
        "adjusted": sourced["adjusted_path"], "reference": sourced["execution_reference_path"],
        "classification": classification,
        **{key: sourced["status_paths"][0] for key in ("status_roster", "status_updates", "status_star")},
    }
    monkeypatch.setattr(config, "_ARCHIVE", archives)
    monkeypatch.setattr(config, "_RAW", sourced["raw_db"])
    monkeypatch.setattr(config, "START", sourced["start"])
    cutoff = dates[57]
    spec = {"pool": "full", "start": sourced["start"], "end": str(cutoff.date())}
    before = config.prepare_expert_inputs(spec)
    assert len(before.roster) == 4
    assert before.frame.loc[before.frame["stock"].eq("600001.SH"), "date"].min() == dates[38]
    assert before.frame.loc[before.frame["stock"].eq("688001.SH"), "date"].max() == dates[54]
    assert before.coverage["daily_coverage"][str(dates[37].date())]["active_members"] == 3
    assert before.coverage["daily_coverage"][str(dates[38].date())]["active_members"] == 4
    assert before.coverage["daily_coverage"][str(dates[55].date())]["active_members"] == 3
    assert before.frame["label_end"].dropna().le(cutoff).all()
    assert before.frame.loc[before.frame["date"].eq(cutoff), "target"].isna().all()

    stage = tmp_path / "sealed"
    manifest = freeze_expert_full_inputs(spec, stage)
    assert manifest == freeze_expert_full_inputs(spec | {"commission_bps": "8"}, stage)
    restored = read_expert_full_freeze(spec, stage)
    pd.testing.assert_frame_equal(
        restored.frame.drop(columns="concepts"), before.frame.drop(columns="concepts"), check_dtype=False,
    )
    pd.testing.assert_series_equal(restored.benchmark_returns, before.benchmark_returns)
    assert restored.coverage == before.coverage
    with (stage / "coverage.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(config.ExpertSpecError, match="SHA-256"):
        read_expert_full_freeze(spec, stage)

    for name in ("adjusted", "reference", "benchmark"):
        frame = pd.read_csv(archives[name], parse_dates=["date"])
        frame.loc[frame["date"].gt(cutoff), "open"] *= 4
        frame.to_csv(archives[name], index=False)
    with closing(sqlite3.connect(sourced["raw_db"])) as connection, connection:
        connection.execute("UPDATE stock_daily_normalized SET open=open*4 WHERE date>?", (str(cutoff.date()),))
        connection.execute("UPDATE stock_daily_raw SET trade_open=trade_open*4 WHERE date>?", (str(cutoff.date()),))

    after = config.prepare_expert_inputs(spec)
    pd.testing.assert_frame_equal(after.frame, before.frame)
    pd.testing.assert_series_equal(after.benchmark_returns, before.benchmark_returns)
    assert after.coverage["daily_coverage"] == before.coverage["daily_coverage"]
