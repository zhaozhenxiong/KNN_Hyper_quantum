"""Three-round, research-only signal and fusion study on a verified expert archive."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.dynamic_fusion import (
    dynamic_predict, fit_dynamic_fusion, fit_shadow_exposure, shadow_exposure_actions,
)
from my_strategy.latent_quant.evaluation import equity_returns, paired_block_interval, score_quantiles
from my_strategy.latent_quant.execution import StudyCostModel
from my_strategy.latent_quant.runner import COSTS, HISTORICAL_END, PORTFOLIO
from my_strategy.latent_quant.signal_experts import ALL_EXPERTS, BASE_EXPERTS, defensive_exposure, forecast_experts
from my_strategy.research_workbench.runner import _write_table, write_json

from . import expert_calibration as calibration
from . import expert_research as research
from . import expert_score_health as health


LOG = logging.getLogger(__name__)
SOURCE = "20261124-011833-455085-20261120-expert-research-19f8ef2"
REPORT = "expert_multiview_study.json"
RANK_POLICY = replace(PORTFOLIO, selection_mode="cross_sectional_rank", min_score=None)
DEVELOPMENT = ("oos_1", "oos_2", "oos_3")
ROUND = {
    **{name: "round_1_rules" for name in BASE_EXPERTS},
    "majority_vote": "round_2_online_and_votes", "online_knn_volume_trend": "round_2_online_and_votes",
}


def _regimes(exposure: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    risk = exposure["defensive_state"].isin(("defensive_cash", "reduced"))
    return pd.Series(risk.reindex(rows["date"]).fillna(False).to_numpy(dtype=bool), index=rows.index)


def _inner_signals(
    archive: calibration.Archive,
    fold: dict,
    train: pd.DataFrame,
    exposure: pd.DataFrame,
    weights: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.Series, dict]:
    folder = archive.root / "backtests" / fold["period"]
    oof = pd.read_parquet(folder / "inner_oof.parquet")
    current = research._fuse(weights, {
        name: oof[[f"{name}_{q}" for q in models.QUANTILE_COLUMNS]].to_numpy() for name in weights
    })
    forecasts = {"current_fusion": current}
    collected = {name: [] for name in ALL_EXPERTS}
    keys, boundaries = [], []
    for past, unseen in research._inner_windows(train):
        predicted = forecast_experts(past, unseen)
        for name, forecast in predicted.items():
            collected[name].append(forecast.quantiles)
        keys.append(unseen[["date", "stock"]])
        boundaries.append({
            "train": models._sample_evidence(past), "out_of_fold": models._sample_evidence(unseen),
            "signals": {name: forecast.evidence for name, forecast in predicted.items()},
        })
    expected = pd.concat(keys).reset_index(drop=True)
    if not expected.equals(oof[["date", "stock"]].reset_index(drop=True)):
        raise ValueError("New signal OOF keys do not match the archived joint OOF rows")
    forecasts.update({name: np.vstack(parts) for name, parts in collected.items()})
    return oof, forecasts, _regimes(exposure, oof), {"subfolds": boundaries}


def _archived_prediction(archive: calibration.Archive, fold: dict, evaluation: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = pd.read_parquet(archive.root / "backtests" / fold["period"] / f"{arm}_predictions.parquet")
    keys = ["date", "stock"]
    if not out[keys].reset_index(drop=True).equals(evaluation[keys].reset_index(drop=True)):
        raise ValueError("Archived prediction keys differ from the frozen evaluation rows")
    out.index = evaluation.index
    return out


def _legacy_replay(archive: calibration.Archive, fold: dict, folder: Path) -> dict:
    checks = {}
    for cost in COSTS:
        for name in ("daily", "orders", "trades", "positions"):
            old = pd.read_parquet(archive.root / "backtests" / fold["period"] / f"fusion_{cost}_{name}.parquet")
            new = pd.read_parquet(folder / f"fusion_absolute_{cost}_{name}.parquet")
            columns = [column for column in old.columns if column != "run_id"]
            pd.testing.assert_frame_equal(
                old[columns].reset_index(drop=True), new[columns].reset_index(drop=True),
                check_dtype=False, check_exact=True,
            )
        checks[cost] = "exact_daily_orders_trades_positions_match"
    return checks


def _prediction_metrics(common: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    joined = common.join(prediction)
    measured = score_quantiles(joined)
    measured["all_signal_rows"] = len(joined)
    return measured


def _fold(
    archive: calibration.Archive,
    fold: dict,
    context,
    exposure: pd.DataFrame,
) -> tuple[dict, dict, list[dict]]:
    period = fold["period"]
    frame = archive.data.frame
    train, cal = (calibration._settled(frame, fold[stage]) for stage in ("train", "calibration"))
    evaluation = frame.loc[
        frame["date"].between(pd.Timestamp(fold["evaluation_start"]), pd.Timestamp(fold["evaluation_end"]))
        & frame["signal_eligible"].eq(True)
    ]
    if not train["label_end"].max() < cal["date"].min() or not cal["label_end"].max() < evaluation["date"].min():
        raise ValueError("Outer train/calibration/evaluation labels overlap")
    dates = pd.DatetimeIndex(evaluation["date"].unique())
    expected_dates = archive.data.sessions[
        (archive.data.sessions >= pd.Timestamp(fold["evaluation_start"]))
        & (archive.data.sessions <= pd.Timestamp(fold["evaluation_end"]))
    ]
    if not dates.equals(expected_dates):
        raise ValueError("Evaluation calendar differs from archive")
    folder = context.subdir("backtests", period)
    model_file = research._json(archive.root / "models" / period / "fusion.json")
    weights = model_file["winner"]["weights"]
    rows = pd.concat([cal, evaluation])
    inference = {}
    current = calibration._raw_predict(archive, fold, weights, train, rows, inference)
    archived = _archived_prediction(archive, fold, evaluation, "fusion")
    offsets = np.asarray(model_file["calibration"]["fusion"]["offsets"])
    replayed = calibration._apply(current[len(cal):], offsets)
    if not np.allclose(replayed, archived[list(models.QUANTILE_COLUMNS)], rtol=0, atol=calibration.REPLAY_TOLERANCE):
        raise ValueError("Current fusion native replay does not reproduce archived predictions")

    LOG.info("%s: outer signal experts on %s rows", period, len(rows))
    signals = forecast_experts(train, rows)
    raw = {"current_fusion": current, **{name: forecast.quantiles for name, forecast in signals.items()}}
    LOG.info("%s: inner OOF signal experts", period)
    oof, inner, oof_regimes, inner_evidence = _inner_signals(archive, fold, train, exposure, weights)
    dynamic_model = fit_dynamic_fusion(oof, oof_regimes, inner)
    dynamic_oof, _, _ = dynamic_predict(dynamic_model, oof, oof_regimes, inner)
    shadow_model = fit_shadow_exposure(oof, oof_regimes, dynamic_oof)
    dynamic_raw, weight_daily, dynamic_evidence = dynamic_predict(dynamic_model, rows, _regimes(exposure, rows), raw)
    shadow_daily = shadow_exposure_actions(shadow_model, evaluation, _regimes(exposure, evaluation), dynamic_raw[len(cal):])
    _write_table(folder / "dynamic_weights.parquet", weight_daily)
    _write_table(folder / "shadow_exposure.parquet", shadow_daily)
    write_json(context.subdir("models", period) / "dynamic_fusion.json", {
        "model": dynamic_model, "inner": inner_evidence, "prediction": dynamic_evidence, "shadow": shadow_model,
        "current_fusion_weights": weights, "inference": inference,
    })

    raw["diverse_equal"] = np.mean(np.stack(list(raw.values())), axis=0)
    raw["dynamic_fusion"] = dynamic_raw
    best_single = dynamic_model["evidence"]["best_single_expert"]
    raw["best_single"] = raw[best_single].copy()
    for removed in ALL_EXPERTS:
        raw[f"leave_out_{removed}"] = np.mean(np.stack([values for name, values in raw.items() if name in {"current_fusion", *ALL_EXPERTS} and name != removed]), axis=0)

    common = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
    common["scored"] = common["target"].notna() & common["label_end"].le(dates[-1])
    common["target"] = common["target"].where(common["scored"])
    if not common["scored"].any():
        raise ValueError("No matured common evaluation labels")

    specifications = {
        "fusion_absolute": {"source": "fusion", "round": "round_1_selection", "policy": PORTFOLIO},
        "fusion_rank": {"source": "fusion", "round": "round_1_selection", "policy": RANK_POLICY},
        "fusion_equal_absolute": {"source": "fusion_equal", "round": "round_2_baselines", "policy": PORTFOLIO},
        "fusion_equal_rank": {"source": "fusion_equal", "round": "round_2_baselines", "policy": RANK_POLICY},
        **{name: {"raw": name, "round": ROUND[name], "policy": RANK_POLICY} for name in ALL_EXPERTS},
        "diverse_equal_absolute": {"raw": "diverse_equal", "round": "round_2_baselines", "policy": PORTFOLIO},
        "diverse_equal_rank": {"raw": "diverse_equal", "round": "round_2_baselines", "policy": RANK_POLICY},
        "best_single_rank": {"raw": "best_single", "round": "round_3_dynamic", "policy": RANK_POLICY},
        "dynamic_fusion_absolute": {"raw": "dynamic_fusion", "round": "round_3_dynamic", "policy": PORTFOLIO},
        "dynamic_fusion_rank": {"raw": "dynamic_fusion", "round": "round_3_dynamic", "policy": RANK_POLICY},
        "dynamic_fusion_defensive": {"raw": "dynamic_fusion", "round": "round_3_dynamic", "policy": RANK_POLICY, "defensive": True},
        **{f"leave_out_{name}": {"raw": f"leave_out_{name}", "round": "round_2_ablations", "policy": RANK_POLICY} for name in ALL_EXPERTS},
    }
    arms, returns, matrix = {}, {cost: {} for cost in COSTS}, []
    written_predictions = set()
    for arm, specification in specifications.items():
        policy = specification["policy"]
        if "source" in specification:
            source = specification["source"]
            prediction = _archived_prediction(archive, fold, evaluation, source)[[*models.QUANTILE_COLUMNS, "score"]]
            calibration_evidence = model_file["calibration"][source]
            raw_values = current[len(cal):] if source == "fusion" else None
            prediction_file = f"{source}_predictions.parquet"
        else:
            values = raw[specification["raw"]]
            quantiles, calibration_evidence = models._calibrate(values, cal["target"].to_numpy())
            prediction = models._prediction_table(quantiles, evaluation.index)
            raw_values = values[len(cal):]
            prediction_file = f"{specification['raw']}_predictions.parquet"
        if prediction_file not in written_predictions:
            _write_table(folder / prediction_file, common.join(prediction))
            written_predictions.add(prediction_file)
        measures = _prediction_metrics(common, prediction)
        qualified = int(prediction["score"].notna().sum()) if policy.selection_mode == "cross_sectional_rank" else int(prediction["score"].gt(policy.min_score).sum())
        score_evidence = {
            "raw_evaluation_q50": health._distribution(raw_values[:, 1]) if raw_values is not None else None,
            "calibrated_evaluation_q50": health._distribution(prediction["q50"]),
            "qualified_rows": qualified,
            "entry": {
                "selection_mode": policy.selection_mode, "score": "calibrated_q50",
                "min_score": policy.min_score,
                "comparison": "finite_cross_sectional_rank" if policy.selection_mode == "cross_sectional_rank" else "strictly_greater",
            },
        }
        target_exposure = exposure["target_exposure"] if specification.get("defensive") else None
        execution, timeseries = research._execution(
            archive.data, prediction["score"], dates, context, folder, arm, policy, StudyCostModel(), target_exposure,
        )
        arms[arm] = {
            "round": specification["round"], "prediction": measures, "score_evidence": score_evidence,
            "calibration": calibration_evidence, "execution": execution, "strategy": asdict(policy),
            "defensive_overlay": bool(specification.get("defensive")),
            "best_single_source": best_single if arm == "best_single_rank" else None,
        }
        for cost in COSTS:
            returns[cost][arm] = timeseries[cost]
            matrix.append({
                "period": period, "round": specification["round"], "arm": arm, "cost": cost,
                "selection_mode": policy.selection_mode, "defensive_overlay": bool(specification.get("defensive")),
                "pinball": measures["pinball"], "rank_ic": measures.get("rank_ic"),
                "top10_mean_matured_excess": measures.get("top10_mean_matured_excess"),
                **{key: value for key, value in execution[cost].items() if key not in {"verification", "reasons"}},
                "selection_reasons": json.dumps(execution[cost]["reasons"], ensure_ascii=False, sort_keys=True),
            })
    replay = _legacy_replay(archive, fold, folder)
    result = {
        "period": period, "evaluation_start": dates[0], "evaluation_end": dates[-1], "evaluation_sessions": len(dates),
        "train": models._sample_evidence(train), "calibration": models._sample_evidence(cal),
        "arms": arms, "signals": {name: forecast.evidence for name, forecast in signals.items()},
        "dynamic_fusion": dynamic_model, "dynamic_prediction": dynamic_evidence,
        "shadow_rl": {"model": shadow_model, "action_count": len(shadow_daily), "orders_emitted": 0},
        "legacy_fusion_replay": replay,
    }
    write_json(context.subdir("reports") / f"{period}.json", result)
    return result, returns, matrix


def _contrasts(folds: list[dict], returns: list[dict]) -> dict:
    pairs = {
        "dynamic_vs_diverse_equal": ("dynamic_fusion_rank", "diverse_equal_rank"),
        "dynamic_vs_current_rank": ("dynamic_fusion_rank", "fusion_rank"),
        "dynamic_vs_best_single": ("dynamic_fusion_rank", "best_single_rank"),
        "current_rank_vs_absolute": ("fusion_rank", "fusion_absolute"),
        "dynamic_defensive_vs_dynamic": ("dynamic_fusion_defensive", "dynamic_fusion_rank"),
    }
    return {
        name: {
            cost: paired_block_interval([(fold[cost][candidate], fold[cost][reference]) for fold in returns], seed=42, block_size=20)
            for cost in COSTS
        }
        for name, (candidate, reference) in pairs.items()
    }


def _verify_outputs(root: Path, folds: list[dict], matrix: pd.DataFrame) -> dict:
    checks = []
    for fold in folds:
        period = fold["period"]
        folder = root / "backtests" / period
        for arm in fold["arms"]:
            for cost in COSTS:
                daily = pd.read_parquet(folder / f"{arm}_{cost}_daily.parquet")
                trades = pd.read_parquet(folder / f"{arm}_{cost}_trades.parquet")
                orders = pd.read_parquet(folder / f"{arm}_{cost}_orders.parquet")
                positions = pd.read_parquet(folder / f"{arm}_{cost}_positions.parquet")
                selection = pd.read_parquet(folder / f"{arm}_{cost}_selection_daily.parquet")
                metrics = fold["arms"][arm]["execution"][cost]
                if (daily["date"].duplicated().any() or len(daily) != fold["evaluation_sessions"]
                        or len(trades) != metrics["trade_count"] or len(selection) != len(daily)
                        or (not orders.empty and orders["date"].min() < daily["date"].min())
                        or (not positions.empty and positions["date"].max() > daily["date"].max())):
                    raise ValueError(f"Artifact verification failed for {period}/{arm}/{cost}")
                checks.append({"period": period, "arm": arm, "cost": cost, "daily": len(daily), "trades": len(trades)})
    if len(matrix) != len(checks):
        raise ValueError("Experiment matrix and verified ledger counts differ")
    return {"ledger_tables": len(checks), "matrix_rows": len(matrix), "all_daily_dates_unique": True, "all_trade_counts_match": True}


def run(*, source_run_id: str = SOURCE, run_id: str, periods: tuple[str, ...] | None = None) -> Path:
    if run_id != slugify(run_id, limit=120):
        raise ValueError("Invalid run identifier")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    status = root / "status.json"
    write_json(status, {"status": "running", "phase": "verify_archive", "source_run_id": source_run_id})
    try:
        archive = calibration.open_archive(source_run_id)
        available = {fold["period"]: fold for fold in research._archived_folds(archive.report)}
        selected = periods or tuple(available)
        if set(selected).difference(available):
            raise ValueError("Requested period is absent from the verified archive")
        code = [Path(__file__), Path(calibration.__file__), Path(research.__file__)]
        code += [Path(models.__file__).with_name(name) for name in ("signal_experts.py", "dynamic_fusion.py", "execution.py", "models.py")]
        config = {
            "source_run_id": source_run_id, "periods": list(selected), "development_periods": list(DEVELOPMENT),
            "historical_replay_role": "diagnostic_only_not_independent_holdout", "seed": 42,
            "base_experts": list(BASE_EXPERTS), "all_signal_experts": list(ALL_EXPERTS),
            "fusion_experts": ["current_fusion", *ALL_EXPERTS],
            "absolute_policy": asdict(PORTFOLIO), "rank_policy": asdict(RANK_POLICY),
            "cost_multipliers": COSTS, "expense_model": asdict(StudyCostModel()),
            "dynamic_regularization": {"global": 0.01, "risk_off": 0.05, "minimum_regime_dates": 20},
            "defensive_exposure": {"actions": [0, 0.5, 1], "trend_days": 20, "volatility_days": 20, "reference_min_days": 60},
            "shadow_rl": {"enabled": True, "stock_orders": False},
            "release": "research_only_no_registry_write", "code_sha256": {path.name: research._sha(path) for path in code},
        }
        context = create_run_context(
            task="expert-multiview-study", as_of_date=HISTORICAL_END, config=config,
            data_version=stable_hash(archive.report["prediction_evidence_sha256"]), seed=42, run_id=run_id,
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", archive.report["input_coverage"])
        exposure = defensive_exposure(archive.data.benchmark_returns)
        _write_table(context.subdir("data_snapshots") / "defensive_exposure.parquet", exposure.reset_index(names="date"))
        folds, development_returns, matrix = [], [], []
        returns_by_period = {}
        for period in selected:
            write_json(status, {"status": "running", "phase": period, "source_run_id": source_run_id})
            result, returns, rows = _fold(archive, available[period], context, exposure)
            folds.append(result)
            returns_by_period[period] = returns
            matrix.extend(rows)
            if period in DEVELOPMENT:
                development_returns.append(returns)
        development_folds = [fold for fold in folds if fold["period"] in DEVELOPMENT]
        contrasts = _contrasts(development_folds, development_returns) if len(development_folds) == 3 else {}
        risk = research._risk_checks(development_folds, "dynamic_fusion_rank", "fusion_rank", explicit_evidence=True) if len(development_folds) == 3 else {
            "passed": False, "reason": "full_three_fold_development_not_run",
        }
        dynamic_gate = bool(
            contrasts and risk["passed"]
            and all(contrasts[name]["double"]["ci95_low"] > 0 for name in ("dynamic_vs_diverse_equal", "dynamic_vs_current_rank"))
            and contrasts["dynamic_vs_diverse_equal"]["double"]["net_return_difference"] > 0
        )
        decision = {
            "release_eligible": False, "registry_written": False, "status": "research_only",
            "retain_dynamic_for_forward_research": dynamic_gate,
            "fallback": "current_fusion" if not dynamic_gate else "dynamic_fusion_remains_unapproved",
            "reasons": [
                "The historical replay period was already used for diagnosis and is not an independent holdout.",
                "No separate 60-session prospective forward evidence is available.",
                "Historical numerical and risk gates do not authorize promotion.",
            ],
        }
        table = pd.DataFrame(matrix)
        _write_table(context.subdir("reports") / "experiment_matrix.csv", table)
        verification = _verify_outputs(root, folds, table)
        write_json(context.subdir("reports") / "artifact_verification.json", verification)
        report = {
            "run": context.to_dict(), "source_run_id": source_run_id, "config": config,
            "folds": folds, "development_contrasts": contrasts, "development_risk": risk,
            "decision": decision, "artifact_verification": verification,
            "interpretation": {
                "score": "q50 is a forecast of T+1-to-T+6 open excess return versus CSI 300, not absolute stock return.",
                "rank": "cross_sectional_rank is an explicit research selection policy, not evidence of positive expected absolute return.",
                "defensive_cash": "a separate causal exposure overlay; zero exposure can legitimately produce no buys.",
                "shadow_rl": "shadow exposure actions only; no stock orders and no promotion evidence.",
            },
        }
        path = context.subdir("reports") / REPORT
        write_json(path, report)
        write_json(status, {"status": "completed", "phase": "complete", "release_eligible": False})
        return path
    except Exception as error:
        write_json(status, {"status": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=SOURCE)
    parser.add_argument("--run-id")
    parser.add_argument("--period", action="append", dest="periods")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(source_run_id=args.source_run_id, run_id=args.run_id or make_run_id("expert-multiview-study", HISTORICAL_END), periods=tuple(args.periods or []) or None))


if __name__ == "__main__":
    main()
