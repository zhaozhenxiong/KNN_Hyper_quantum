"""Synthetic checks for the expert research data boundary and evidence output."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.domestic_runner import FrozenInputs
from my_strategy.research_workbench import expert_research as expert
from my_strategy.research_workbench.runner import Window
from my_strategy.storage.expert_artifacts import ArtifactRef


def _panel() -> tuple[FrozenInputs, Window]:
    dates = pd.bdate_range("2024-01-03", periods=175)
    symbols = ("600001.SH", "600002.SH", "600003.SH")
    price = 15 + 0.015 * np.arange(len(dates)).repeat(3)
    frame = pd.DataFrame({
        "date": dates.repeat(3), "stock": list(symbols) * len(dates), "signal_eligible": True,
        "target": np.linspace(-0.01, 0.035, len(dates) * 3),
        "label_end": np.repeat(pd.Series(dates).shift(-6).to_numpy(), 3),
        "trade_date": np.repeat(pd.Series(dates).shift(-1).to_numpy(), 3),
        "trade_close": price, "trade_open": price, "amount": 15e6,
        "st_direction": 1, "atr_pct": 0.02, "st_distance": 0.03,
        "board": "sh_main", "industry": "unknown", "x": price,
    })
    frame["target"] = frame["target"].where(frame["label_end"].notna())
    state = pd.DataFrame(columns=["date", "stock", "market_status", "source"])
    roster = pd.DataFrame({"delist_date": pd.NaT}, index=pd.Index(symbols, name="stock"))
    inputs = FrozenInputs(frame, dates, pd.Series(0.0, index=dates), state, roster, {})
    return inputs, Window("oos_1", dates[0], dates[112], dates[126], dates[145:])


def test_verified_macro_drops_nonofficial_or_late_values(tmp_path):
    data, _ = _panel()
    dates = data.sessions
    file = tmp_path / "overseas.parquet"
    macro = pd.DataFrame({
        "date": dates, "fomc_availability_basis": "official_release_header",
        "fomc_available_at": dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=12),
        "cutoff_at": dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=18),
    })
    for field in expert.MACRO_FEATURES:
        macro[field] = 1.0
    macro.loc[0, "fomc_availability_basis"] = "nominal_regular_close"
    macro.loc[1, "fomc_available_at"] += pd.Timedelta(hours=12)
    macro.to_parquet(file)
    merged, info = expert._verified_macro(data, file)
    assert info["verified_sessions"] == len(dates) - 2
    assert merged.frame.loc[merged.frame["date"].isin(dates[:2]), list(expert.MACRO_FEATURES)].isna().all(axis=None)
    assert merged.frame.loc[merged.frame["date"].gt(dates[1]), list(expert.MACRO_FEATURES)].notna().all(axis=None)


def test_fixed_references_remain_available_for_default_frozen_research(tmp_path, monkeypatch):
    source = tmp_path / "panel.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "stock": ["600000.SH"]}).to_parquet(source, index=False)
    reference = ArtifactRef(sha256="a" * 64, size_bytes=source.stat().st_size)
    monkeypatch.setattr(expert, "_sources", lambda: {"panel": source})

    class Store:
        def put_file(self, path):
            assert path == source
            return reference

    identity = {"sources": {"panel": {"sha256": "a" * 64, "size_bytes": source.stat().st_size}}}
    assert expert._fixed_references(Store(), identity) == {"frozen_inputs/panel/panel.parquet": reference}
    with pytest.raises(ValueError, match="冻结输入在研究运行中变化"):
        expert._fixed_references(Store(), {"sources": {"panel": {"sha256": "b" * 64, "size_bytes": 1}}})


def test_nested_oof_trains_before_predictions_and_writes_verified_ledgers(tmp_path, monkeypatch):
    data, window = _panel()
    train_calls = []

    class Model:
        def __init__(self, value):
            self.value = value
            self.tree_count_ = 1

        def predict(self, matrix, **kwargs):
            return np.full(len(matrix), self.value)

        def save_model(self, path, **kwargs):
            Path(path).write_bytes(b"synthetic")

    def fit(train, valid, features, parameters, seed):
        train_calls.append((train["label_end"].max(), None if valid is None else valid["date"].min()))
        return [Model(value) for value in (-0.01, 0.017, 0.04)]

    class Context:
        run_id = "synthetic-expert"

        def subdir(self, *parts):
            output = tmp_path.joinpath(*parts)
            output.mkdir(parents=True, exist_ok=True)
            return output

    monkeypatch.setattr(expert, "_fit_gpu", fit)
    fold, returns = expert._fold(data, window, Context(), {"domestic_gpu": ("x",)})
    assert set(fold["arms"]) == {
        "historical_quantiles", "supertrend_atr", "domestic_gpu", "fusion", "fusion_equal", "fusion_no_trend",
    }
    assert "fusion_no_overseas" not in fold["ablations"]
    assert all(not ablation["extra_models_fitted"] for ablation in fold["ablations"].values())
    assert len(train_calls) == 4
    assert fold["inference"]["domestic_gpu"]["calls"] == {"GPU": 15, "CPU": 0}
    for step in fold["fusion"]["subfolds"]:
        assert step["trained"]["last_label_end"] < step["out_of_fold"]["start"]
        assert pd.Timestamp(step["out_of_fold"]["last_label_end"]) < window.valid_start
    assert fold["models"]["calibration"]["last_label_end"] < str(window.evaluate[0].date())
    assert fold["train"]["target"]["rows"] > 0
    assert fold["arms"]["fusion"]["score_evidence"]["entry"]["comparison"] == "strictly_greater"
    assert not fold["model_health"]["passed"]
    assert fold["model_health"]["failed_experts"] == ["domestic_gpu"]
    assert sum(fold["fusion"]["winner"]["weights"].values()) == 1
    assert fold["fusion"]["winner"]["weights"]["domestic_gpu"] >= 0.5
    for cost in expert.COSTS:
        assert returns[cost]["fusion"].index.equals(window.evaluate)
        assert fold["arms"]["fusion"]["execution"][cost]["verification"]["next_session_verified"]
    table = pd.read_parquet(tmp_path / "backtests" / window.period / "fusion_predictions.parquet")
    assert table["date"].nunique() == len(window.evaluate)
    assert table.loc[table["scored"], "label_end"].le(window.evaluate[-1]).all()
    final = expert._final_portfolio(tmp_path, [fold], window.period)
    assert final["channel"] == "research" and final["arm"] == "fusion"
    assert all(action["signal_date"] == final["latest_action_date"] for action in final["latest_actions"])
    assert len(final["latest_actions"]) > 0
    decision = expert._decision([fold], fold["contrasts"], False)
    assert decision["state"] == "research_only" and decision["paper_model"]["state"] == "blocked"
    assert decision["median_model_health_gate"] is False
    assert decision["risk_acceptance"]["folds"]["oos_1"]["normal"]["turnover_pass"] in (True, False)


def test_gpu_policy_discloses_unavailable_and_mismatched_predictions():
    catboost = pytest.importorskip("catboost")
    frame = pd.DataFrame({"x": [0.0, 1.0]})

    class Model:
        def __init__(self, case):
            self.case = case

        def predict(self, matrix, *, task_type, thread_count):
            if task_type == "GPU" and self.case == "unsupported":
                raise catboost.CatBoostError("fake accelerator absence")
            if task_type == "GPU" and self.case == "crosses_threshold":
                return np.array([-1e-9, -1e-9])
            return np.array([1e-9, 1e-9])

    audit = expert._inference_audit()
    chosen = expert._predict([Model(name) for name in ("supported", "unsupported", "crosses_threshold")], frame, ("x",), audit)
    np.testing.assert_allclose(chosen, 1e-9, rtol=0, atol=0)
    assert audit["calls"] == {"GPU": 1, "CPU": 2}
    assert len(audit["fallback_reasons"]) == 2
    assert audit["max_absolute_error"] == 2e-9


def test_ablations_renormalize_and_strict_risk_gate_uses_no_worse_than_domestic():
    result = expert._ablations({"domestic_gpu": 0.5, "overseas_gpu": 0.25, "supertrend_atr": 0.25}, {"domestic_gpu": (), "overseas_gpu": ()})
    assert result["fusion_no_overseas"]["weights"] == {"domestic_gpu": 2 / 3, "supertrend_atr": 1 / 3}
    assert result["fusion_no_trend"]["weights"] == {"domestic_gpu": 2 / 3, "overseas_gpu": 1 / 3}
    common = {"max_drawdown": -0.04, "mean_daily_turnover": 0.08, "reject_rate": 0.02, "order_count": 100}
    arms = {
        "domestic_gpu": {"execution": dict.fromkeys(expert.COSTS, common)},
        "fusion": {"execution": {"normal": common, "double": {**common, "max_drawdown": -0.05, "reject_rate": 0.03}}},
    }
    risk = expert._risk_checks([{"period": "oos_1", "arms": arms}])
    assert risk["folds"]["oos_1"]["normal"]["passed"]
    assert not risk["folds"]["oos_1"]["double"]["drawdown_pass"]
    assert not risk["folds"]["oos_1"]["double"]["reject_rate_pass"]
    assert not risk["passed"]


def test_cost_only_replay_keeps_frozen_predictions_and_exposes_actual_fees(tmp_path, monkeypatch):
    data, window = _panel()

    class Model:
        tree_count_ = 1

        def __init__(self, value):
            self.value = value

        def predict(self, matrix, **kwargs):
            return np.full(len(matrix), self.value)

        def save_model(self, path, **kwargs):
            Path(path).write_text(str(self.value), encoding="utf-8")

    class Context:
        def __init__(self, name):
            self.root = tmp_path / name
            self.run_id = name

        def subdir(self, *parts):
            output = self.root.joinpath(*parts)
            output.mkdir(parents=True, exist_ok=True)
            return output

    monkeypatch.setattr(expert, "_fit_gpu", lambda *args: [Model(value) for value in (-0.01, 0.017, 0.04)])
    reference = Context("reference")
    _, original_policy, original_cost = expert._execution_spec({"commission_bps": "3"})
    baseline, _ = expert._fold(data, window, reference, {"domestic_gpu": ("x",)}, policy=original_policy, cost_model=original_cost)
    sealed = expert._prediction_evidence(reference.root, [baseline])
    assert sealed
    frozen = Context("repriced")
    expert._copy_predictions(reference.root, frozen.root, {"files": sealed})

    def forbid_fit(*args, **kwargs):
        pytest.fail("reprice must use already saved scores")

    monkeypatch.setattr(expert, "_fit_gpu", forbid_fit)
    basis, policy, cost = expert._execution_spec({"commission_bps": "12.000", "rebalance_days": 10, "top_n": 2})
    result, returns = expert._reprice_fold(data, window, frozen, baseline, policy=policy, cost_model=cost, seed=42)
    assert basis == "12" and policy.rebalance_days == 10 and policy.top_n == 2
    assert expert._prediction_evidence(frozen.root, [result]) == sealed
    assert expert._prediction_evidence(reference.root, [baseline]) == sealed
    assert result["arms"]["fusion"]["prediction"] == baseline["arms"]["fusion"]["prediction"]
    for name, rate in {"normal": 0.0012, "double": 0.0024}.items():
        assert returns[name]["fusion"].index.equals(window.evaluate)
        trades = pd.read_parquet(frozen.root / "backtests" / window.period / f"fusion_{name}_trades.parquet")
        assert len(trades) > 0 and trades["commission_rate"].eq(rate).all()
        assert trades["sell_stamp_tax_rate"].eq(original_cost.stamp_tax).all()
        np.testing.assert_allclose(trades["fee"], trades["commission_paid"] + trades["stamp_tax_paid"] + trades["transfer_fee_paid"])


def test_custom_folds_do_not_overlap_and_identity_excludes_just_execution(monkeypatch):
    from my_strategy.research_workbench import expert_config

    data, _ = _panel()
    dates = pd.bdate_range("2024-01-03", periods=280)
    rows = pd.DataFrame({"date": dates, "signal_eligible": True, "target": 0.02, "label_end": pd.Series(dates).shift(-6)})
    complete = FrozenInputs(rows, dates, data.benchmark_returns, data.market_status, data.roster, {})
    scope = {"preset": "custom", "pool": "frozen_120", "start": str(dates[185].date()), "end": str(dates[-1].date()), "seed": 2}
    windows = expert.plan_expert_windows(complete, scope)
    assert [len(window.evaluate) for window in windows] == [32, 32, 31]
    assert all(a.evaluate[-1] < b.evaluate[0] for a, b in zip(windows, windows[1:]))
    assert windows[-1].evaluate[-1] == dates[-1]
    with pytest.raises(ValueError, match="90"):
        expert.plan_expert_windows(complete, scope | {"end": str(dates[226].date())})
    monkeypatch.setattr(expert_config, "expert_input_identity", lambda spec: {"test_scope": spec["pool"]})
    monkeypatch.setattr(expert, "research_inputs", lambda: {"model_source": "stable"})
    key = expert.prediction_key(scope)
    assert key == expert.prediction_key(scope | {"commission_bps": "9", "rebalance_days": 10, "top_n": 2})
    assert key != expert.prediction_key(scope | {"seed": 3})


def test_full_calendar_preserves_original_folds_and_isolates_review():
    dates = pd.bdate_range("2024-01-02", "2026-11-20")
    scope = {"pool": "full", "start": "2025-03-03", "end": "2026-11-20"}
    windows = expert.plan_expert_calendar(dates, scope)
    assert [window.period for window in windows] == ["oos_1", "oos_2", "oos_3", "historical_replay"]
    assert [str(window.evaluate[0].date()) for window in windows] == ["2025-03-03", "2025-10-09", "2026-04-01", "2026-07-29"]
    assert windows[-1].evaluate[-1] == pd.Timestamp("2026-11-20")
    with pytest.raises(ValueError, match="截止"):
        expert.plan_expert_calendar(dates.append(pd.DatetimeIndex(["2026-11-23"])), scope)


def test_full_grid_chooses_validation_score_and_repeats_only_winner(monkeypatch):
    class Scheduler:
        def __init__(self):
            self.calls = []

        def fit(self, train, valid, columns, params, seed, *, iterations, patience):
            self.calls.append((params, seed, iterations, patience))
            return [params], [{"trees": 10}] * 3

    scheduler = Scheduler()
    winner = expert.expert_gpu.GRID[-1]
    monkeypatch.setattr(expert, "_predict", lambda model, *args: np.full((2, 3), 1 if model[0] == winner else 2))
    monkeypatch.setattr(expert, "_pinball", lambda rows, estimate: estimate[0, 0])
    fitted, evidence = expert._tune_full(None, None, ("x",), 42, {}, scheduler)
    assert fitted == [winner]
    assert len(evidence["candidates"]) == 8
    assert scheduler.calls[:-1] == [(params, 42, 160, 20) for params in expert.expert_gpu.GRID]
    assert scheduler.calls[-1] == (winner, 43, 160, 20)
    assert not evidence["seed_stability"]["used_for_selection"]


def test_single_champion_requires_both_references_and_never_releases():
    def arm(pinball):
        metrics = {"max_drawdown": -0.01, "mean_daily_turnover": 0.01, "reject_rate": 0., "order_count": 1}
        return {"prediction": {"pinball": pinball}, "execution": {cost: metrics for cost in expert.COSTS}}

    arms = {"fusion": arm(0.012), "domestic_gpu": arm(0.010), "domestic_fixed": arm(0.020), "historical_quantiles": arm(0.022)}
    names = [*expert.CONTRASTS, *expert.FULL_CONTRASTS]
    contrasts = {name: {"double": {"net_return_difference": 0.02}} for name in names}
    folds = [{"period": f"oos_{fold}", "arms": arms, "contrasts": contrasts} for fold in range(1, 4)]
    pooled = {name: {"double": {"ci95_low": -0.01 if name == "fusion_vs_fixed" else 0.01}} for name in names}
    result = expert._single_champion(folds, pooled)
    assert result["selected_arm"] == "domestic_gpu"
    assert result["state"] == "historical_candidate"
    assert not result["release_eligible"] and not result["prospective_ready"]
    pooled["domestic_vs_fixed"]["double"]["ci95_low"] = 0.
    assert expert._single_champion(folds, pooled)["state"] == "best_research_only"


def test_matched_inner_experts_use_same_grid_and_never_see_oof_labels(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=215)
    source = pd.DataFrame({"date": dates[:202], "label_end": dates[6:208], "stock": "A", "target": 0.01,
                           "st_direction": 1.0, "atr_pct": 0.015, "st_distance": 0.016})
    calls = []

    def tune(fit, valid, features, seed, audit, scheduler, **kwargs):
        calls.append((fit, valid, kwargs))
        assert fit.label_end.max() < valid.date.min()
        assert valid.date.nunique() == 29
        return [], {"features": list(features), "validation_pinball": 0.01}

    monkeypatch.setattr(expert, "_tune_full", tune)
    monkeypatch.setattr(expert, "_expert_raw", lambda fitted, rows, *args, **kwargs: np.tile([0.0, 0.01, 0.02], (len(rows), 1)))
    _, details, oof = expert._inner_fusion(source, {"domestic_gpu": ("atr_pct",)}, 42, {"domestic_gpu": {}}, object(), matched_fit=True)
    assert len(calls) == 2
    assert details["fit_protocol"] == expert.MATCHED_PROTOCOL
    assert details["expert_grid"] == expert.expert_gpu.GRID
    assert len(oof) == 80
    for (fit, valid, kwargs), evidence in zip(calls, details["subfolds"], strict=True):
        assert kwargs == {"repeat_winner": False}
        assert fit.label_end.max() < valid.date.min() < valid.label_end.max() < pd.Timestamp(evidence["out_of_fold"]["start"])


def test_optimized_gate_preserves_no_order_evidence_as_missing():
    empty = {"max_drawdown": 0., "mean_daily_turnover": 0., "reject_rate": None, "order_count": 0}
    measured = {**empty, "reject_rate": 0.1, "order_count": 10}
    fold = {"period": "oos_1", "arms": {arm: {"execution": {cost: result for cost in expert.COSTS}}
            for arm, result in (("fusion", empty), ("domestic_fixed", measured))}}
    result = expert._risk_checks([fold], "fusion", "domestic_fixed", explicit_evidence=True)
    assert result["folds"]["oos_1"]["double"]["reject_rate_pass"] is None
    assert result["folds"]["oos_1"]["double"]["execution_evidence"] == "insufficient"
    assert result["execution_evidence_folds"] == {"normal": 0, "double": 0}
    assert result["passed"] is False
