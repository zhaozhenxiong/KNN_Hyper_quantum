"""Train and validate the production entry using only the causal online KNN expert."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.execution import StudyCostModel, execution_evidence_identity, scaled_study_cost
from my_strategy.latent_quant.runner import COSTS, HISTORICAL_END, INITIAL_CASH, _features_validation, _summary_table
from my_strategy.latent_quant.signal_experts import forecast_experts
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_research as research
from . import expert_score_health as health
from .runner import _write_table, write_json


LOG = logging.getLogger(__name__)
TASK = "expert-online-knn"
ENGINE = "online_knn_volume_trend"
ARM = ENGINE
REPORT = research.REPORT
DEVELOPMENT = ("oos_1", "oos_2", "oos_3")
KNN_FEATURES = ("adx", "mfi", "volume_change")
KNN_NEIGHBORS = 12
KNN_MAX_HISTORY = 2000


def prediction_identity(spec: dict | None = None) -> dict:
    """Pin the same frozen market data without inheriting the multimodal fit contract."""
    identity = research.prediction_identity(spec)
    code = [
        Path(__file__), Path(research.__file__), Path(models.__file__),
        Path(models.__file__).with_name("signal_experts.py"),
        Path(models.__file__).with_name("execution.py"),
        Path(models.__file__).with_name("evaluation.py"),
    ]
    identity["research"] = {
        **identity["research"],
        "code_sha256": {path.name: research._sha(path) for path in code},
    }
    identity["knn"] = {
        "engine": ENGINE, "features": list(KNN_FEATURES), "neighbors": KNN_NEIGHBORS,
        "max_available_history": KNN_MAX_HISTORY, "selection_mode": "cross_sectional_rank",
        "label_rule": "label_end_strictly_before_signal_date",
    }
    return identity


def _fold(data, window, context, *, policy, cost_model) -> tuple[dict, dict]:
    frame = data.frame
    settled = frame["signal_eligible"].eq(True) & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])
    masks = window.masks(frame)
    train, validation, cal = (frame.loc[mask & settled] for mask in masks[:3])
    evaluation = frame.loc[frame["signal_eligible"].eq(True) & frame["date"].isin(window.evaluate)]
    if not train["label_end"].max() < cal["date"].min() or not cal["label_end"].max() < evaluation["date"].min():
        raise ValueError(f"{window.period} 的训练、校准或评估标签越过因果边界")
    dates = pd.DatetimeIndex(evaluation["date"].unique())
    if not dates.equals(window.evaluate):
        raise ValueError(f"{window.period} 的评估交易日与冻结日历不一致")

    rows = pd.concat([cal, evaluation])
    LOG.info("%s: online KNN on %s calibration/evaluation rows", window.period, len(rows))
    forecast = forecast_experts(train, rows, names=(ARM,))[ARM]
    quantiles, calibration_evidence = models._calibrate(forecast.quantiles, cal["target"].to_numpy(dtype=float))
    prediction = models._prediction_table(quantiles, evaluation.index)
    common = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
    common["scored"] = common["target"].notna() & common["label_end"].le(dates[-1])
    common["target"] = common["target"].where(common["scored"])
    if not common["scored"].any():
        raise ValueError(f"{window.period} 没有窗口内成熟的评估标签")

    folder = context.subdir("backtests", window.period)
    _write_table(folder / f"{ARM}_predictions.parquet", common.join(prediction))
    model = {
        "engine": ENGINE, "features": list(KNN_FEATURES), "raw_score": "vwma_sma_state_with_adx",
        "signal": forecast.evidence, "calibration": calibration_evidence,
        "train": models._sample_evidence(train), "validation": models._sample_evidence(validation),
        "calibration_rows": models._sample_evidence(cal), "training_device": "CPU",
    }
    write_json(context.subdir("models", window.period) / "online_knn.json", model)
    execution, returns = research._execution(data, prediction["score"], dates, context, folder, ARM, policy, cost_model)
    measures = research._prediction_metrics(common, prediction) if hasattr(research, "_prediction_metrics") else None
    if measures is None:
        from my_strategy.latent_quant.evaluation import score_quantiles

        measures = score_quantiles(common.join(prediction))
        measures["all_signal_rows"] = len(common)
    raw_q50 = forecast.quantiles[len(cal):, 1]
    score_evidence = {
        "raw_evaluation_q50": health._distribution(raw_q50),
        "calibrated_evaluation_q50": health._distribution(prediction["q50"]),
        "qualified_rows": int(prediction["score"].notna().sum()),
        "entry": {
            "selection_mode": "cross_sectional_rank", "score": "calibrated_q50", "min_score": None,
            "comparison": "finite_cross_sectional_rank",
        },
    }
    result = {
        "period": window.period, "evaluation_start": dates[0], "evaluation_end": dates[-1],
        "evaluation_sessions": len(dates), "train": models._sample_evidence(train),
        "validation": models._sample_evidence(validation), "calibration": models._sample_evidence(cal),
        "models": {
            "train": models._sample_evidence(train), "validation": models._sample_evidence(validation),
            "calibration": models._sample_evidence(cal), "arms": {ARM: model},
        },
        "arms": {ARM: {
            "prediction": measures, "score_evidence": score_evidence, "calibration": calibration_evidence,
            "execution": execution, "strategy": asdict(policy), "signals": forecast.evidence,
        }},
        "model_health": {"passed": True, "basis": "causal_online_knn_with_finite_ordered_quantiles"},
    }
    write_json(context.subdir("reports") / f"{window.period}.json", result)
    return result, returns


def _verify_outputs(root: Path, folds: list[dict], matrix: pd.DataFrame) -> dict:
    checks = []
    for fold in folds:
        period = fold["period"]
        for cost in COSTS:
            metrics = fold["arms"][ARM]["execution"][cost]
            if not metrics.get("verification") or not all(metrics["verification"].values()):
                raise ValueError(f"{period}/{cost} 的现金、权益或下一交易日核验失败")
            tables = {}
            for name in ("daily", "orders", "trades", "positions", "selection_daily"):
                tables[name] = pd.read_parquet(root / "backtests" / period / f"{ARM}_{cost}_{name}.parquet")
            daily, orders, trades, positions = (tables[name] for name in ("daily", "orders", "trades", "positions"))
            if (daily["date"].duplicated().any() or len(daily) != fold["evaluation_sessions"]
                    or len(trades) != metrics["trade_count"]
                    or len(orders) != metrics["order_count"] + metrics.get("pretrade_rejected_count", 0)
                    or len(tables["selection_daily"]) != len(daily)
                    or (not orders.empty and orders["date"].min() < daily["date"].min())
                    or (not positions.empty and positions["date"].max() > daily["date"].max())):
                raise ValueError(f"{period}/{cost} 的 KNN 账本工件不完整")
            checks.append({"period": period, "arm": ARM, "cost": cost, "daily": len(daily),
                           "orders": len(orders), "trades": len(trades)})
    if len(matrix) != len(checks):
        raise ValueError("实验矩阵与已核验账本数量不一致")
    return {
        "ledger_tables": len(checks), "matrix_rows": len(matrix), "all_daily_dates_unique": True,
        "all_trade_counts_match": True, "all_order_counts_match": True, "all_ledger_checks_pass": True,
    }


def run(*, run_id: str, spec: dict | None = None) -> Path:
    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号必须为最多 120 个字符的单级目录名")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)

    def update(status: str, phase: str, **extra) -> None:
        write_json(root / "status.json", {"run_id": run_id, "status": status, "phase": phase, **extra})

    update("running", "loading_inputs")
    try:
        from .expert_config import prepare_expert_inputs
        from .expert_full_data import read_expert_full_freeze

        identity = prediction_identity(spec)
        model_spec = identity["spec"]
        full = model_spec["pool"] == "full"
        key = stable_hash(identity)
        commission_bps, absolute_policy, cost_model = research._execution_spec(spec)
        policy = replace(absolute_policy, selection_mode="cross_sectional_rank", min_score=None)
        seed = model_spec.get("seed", research.SEED)
        source_identity = identity["research"]
        store = ExpertArtifactStore()
        if full:
            stage, manifest = research._full_freeze(model_spec)
            if manifest != identity["full_freeze"]:
                raise ValueError("全量研究的封存输入清单在提交后变化")
            raw_data = read_expert_full_freeze(model_spec, stage)
            references = research._full_references(store, stage, manifest, source_identity)
        elif model_spec["preset"] == "frozen_v2":
            raw_data = research._load_frozen()
            references = research._fixed_references(store, source_identity)
        else:
            raw_data = prepare_expert_inputs(model_spec)
            references = research._custom_references(store, raw_data, source_identity)
        if model_spec["preset"] != "frozen_v2" and raw_data.coverage["input_identity"] != identity["inputs"]:
            raise ValueError("KNN 面板在准备前后输入身份不一致")
        data, macro = research._verified_macro(raw_data, research._sources().get("overseas"))
        data, sector = research._verified_sector(
            data, model_spec.get("sector_view", False), model_spec.get("sector_provider", "capco"),
        )
        del raw_data
        coverage = {
            **data.coverage, "overseas": macro, "sector_view": sector, "source_identity": source_identity,
            "model_features": list(KNN_FEATURES), "multimodal_features_used": False,
        }
        windows = research.plan_expert_windows(data, model_spec)
        if [window.period for window in windows if window.period in DEVELOPMENT] != list(DEVELOPMENT):
            raise ValueError("KNN 研究必须包含顺序固定的三个开发折")
        checks = _features_validation(data.frame, data.sessions)
        actual_costs = {cost: asdict(scaled_study_cost(multiplier, cost_model)) for cost, multiplier in COSTS.items()}
        actual_end = data.sessions[-1].date().isoformat()
        config = {
            **source_identity, "sources": research._reference_values(references),
            "source_run_id": research.BASELINE_ID, "end": actual_end, "seed": seed,
            "stock_cap": len(data.roster), "model_spec": model_spec, "training_engine": ENGINE,
            "knn": identity["knn"], "development_periods": list(DEVELOPMENT),
            "portfolio": asdict(policy), "rank_policy": asdict(policy), "initial_cash": INITIAL_CASH,
            "expense_model": asdict(cost_model), "cost_multipliers": COSTS, "execution_costs": actual_costs,
            "execution_evidence": execution_evidence_identity(), "prediction_key": key,
            "commission_bps": commission_bps, "reused_predictions_from": None,
            "forward_isolation": research.PROSPECTIVE_START, "release": "research_only_no_registry_write",
        }
        context = create_run_context(
            task=TASK, as_of_date=actual_end, seed=seed, config=config,
            data_version=stable_hash(config["sources"]), run_id=run_id,
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", coverage)
        write_json(context.subdir("validation") / "feature_label_timing.json", checks)
        folds, review = [], None
        for window in windows:
            update("running", window.period)
            fold, _ = _fold(data, window, context, policy=policy, cost_model=cost_model)
            if window.period == "historical_replay":
                review = fold | {"scope": "retrospective_review", "included_in_champion_selection": False}
            else:
                folds.append(fold)
        all_folds = [*folds, *([review] if review else [])]
        coverage.update(
            first_evaluation_session=str(folds[0]["evaluation_start"])[:10],
            last_evaluation_session=str(folds[-1]["evaluation_end"])[:10],
        )
        write_json(context.subdir("data_snapshots") / "coverage.json", coverage)
        predicted_files = research._prediction_evidence(root, all_folds)
        write_json(root / research.PREDICTION_EVIDENCE, {"prediction_key": key, "files": predicted_files})
        manifest = store.write_manifest(run_id, references, metadata={
            "source_run_id": research.BASELINE_ID, "task": TASK,
        })
        final_portfolio = research._final_portfolio(root, folds, folds[-1]["period"], ARM)
        update("running", "sector_artifacts")
        sector_report = research._research_sector_artifacts(
            root, data, folds, ARM, model_spec.get("sector_provider", "capco"),
        )
        final_portfolio["sector_resonance"] = "reports/sector_resonance.json"
        matrix = _summary_table(all_folds)
        _write_table(context.subdir("reports") / "cost_and_prediction.csv", matrix)
        verification = _verify_outputs(root, all_folds, matrix)
        write_json(context.subdir("reports") / "artifact_verification.json", verification)
        decision = {
            "state": "research_only", "status": "research_only", "registry_written": False,
            "historical_numerical_gate": False,
            "point_in_time_verified": bool(data.coverage["point_in_time"]["verified"]),
            "qualified_forward_results": False,
            "reasons": [
                "训练入口仅使用因果在线 KNN，不拟合或融合任何 GPU、多模态或其他专家模型。",
                "历史复核区间已参与诊断，不是独立前瞻证据；本运行不提升正式冠军。",
                "生产回退由单独的冻结 KNN 清单和同口径收益比较控制。",
            ],
        }
        report = {
            "run": context.to_dict(), "input_coverage": coverage, "validation": checks,
            "experts": {ARM: {"status": "active", "role": "sole_model", "training_device": "CPU"}},
            "model_spec": model_spec, "training_engine": ENGINE, "portfolio": asdict(policy),
            "commission_bps": commission_bps, "execution_costs": actual_costs,
            "execution_evidence": config["execution_evidence"], "prediction_key": key,
            "prediction_evidence_sha256": stable_hash(predicted_files), "reused_predictions_from": None,
            "fit_executed": True, "folds": folds, "historical_review": review,
            "decision": decision, "final_portfolio": final_portfolio, "sector_resonance": sector_report,
            "artifact_verification": verification, "artifact_manifest": str(manifest),
            "oos_block_bootstrap": {}, "ablation_block_bootstrap": {}, "block_sensitivity": {},
            "interpretation": {
                "execution": "唯一在线 KNN 臂使用共同冻结股票池，以前一收盘信号在次日开盘模拟；双档成本并逐折从现金开始。",
                "score": "q50 是相对沪深300的超额收益分位数；生产选股采用横截面排名，不要求 q50 大于零。",
                "scope": "历史开发研究与回放，不是独立前瞻收益证明。",
            },
        }
        report_dir = context.subdir("reports")
        write_json(report_dir / "selected_hyperparameters.json", {
            fold["period"]: {"engine": ENGINE, "knn": identity["knn"], "model": fold["models"]["arms"][ARM]}
            for fold in all_folds
        })
        path = report_dir / REPORT
        write_json(path, report)
        update("completed", "complete", release_state="research_only")
        return path
    except Exception as error:
        update("failed", "failed", error=str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--full", action="store_true", help="使用已经封存的全量历史股票池")
    parser.add_argument("--activate", action="store_true", help="收益高于当前生产时切换 KNN 回退")
    parser.add_argument("--force-production", action="store_true", help="按用户明确决定强制切换；仍需完整账本证据")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = args.run_id or make_run_id(TASK, HISTORICAL_END)
    path = run(run_id=run_id, spec={"pool": "full"} if args.full else None)
    if args.activate or args.force_production:
        from my_strategy.model_registry import expert_fallback

        proof = expert_fallback.activate(path.parent.parent, force=args.force_production)
        print({"report": str(path), "production": proof})
    else:
        print(path)


if __name__ == "__main__":
    main()
