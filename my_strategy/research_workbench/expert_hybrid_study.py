"""Train native multimodal experts and blend their ranks with a CUDA online KNN."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.data import BASE_FEATURES
from my_strategy.latent_quant.evaluation import paired_block_interval, score_quantiles
from my_strategy.latent_quant.execution import execution_evidence_identity, scaled_study_cost
from my_strategy.latent_quant.runner import COSTS, HISTORICAL_END, INITIAL_CASH, _features_validation, _summary_table
from my_strategy.latent_quant.signal_experts import forecast_experts
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_gpu, expert_research as research, expert_score_health as health
from .expert_knn_study import KNN_FEATURES, KNN_MAX_HISTORY, KNN_NEIGHBORS
from .runner import _write_table, write_json


LOG = logging.getLogger(__name__)
TASK = "expert-knn60-multimodal40"
ENGINE = "knn60_multimodal40"
ARM = ENGINE
KNN_ARM = "online_knn_volume_trend"
FUSION_ARM = "fusion"
REPORT = research.REPORT
DEVELOPMENT = ("oos_1", "oos_2", "oos_3")
KNN_WEIGHT = 0.60
MULTIMODAL_WEIGHT = 0.40


def _options(spec: dict | None) -> tuple[dict, str, int]:
    payload = dict(spec or {})
    device = payload.pop("knn_device", "cuda")
    chunk = payload.pop("knn_query_chunk", 256)
    device = str(device).strip().lower()
    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("KNN 设备仅支持 auto、cpu 或 cuda")
    if type(chunk) is not int or chunk < 1:
        raise ValueError("KNN 查询分块必须为正整数")
    return payload, device, chunk


def prediction_identity(spec: dict | None = None) -> dict:
    payload, device, chunk = _options(spec)
    identity = research.prediction_identity(payload)
    code = [
        Path(__file__), Path(research.__file__), Path(models.__file__),
        Path(models.__file__).with_name("signal_experts.py"),
        Path(models.__file__).with_name("knn_backend.py"),
        Path(models.__file__).with_name("execution.py"),
        Path(models.__file__).with_name("evaluation.py"),
        Path(research.__file__).with_name("expert_gpu.py"),
    ]
    identity["research"] = {
        **identity["research"],
        "code_sha256": {**identity["research"]["code_sha256"],
                        **{path.name: research._sha(path) for path in code}},
    }
    identity["knn"] = {
        "engine": KNN_ARM, "features": list(KNN_FEATURES), "neighbors": KNN_NEIGHBORS,
        "max_available_history": KNN_MAX_HISTORY, "selection_mode": "cross_sectional_rank",
        "label_rule": "label_end_strictly_before_signal_date", "requested_device": device,
        "query_chunk": chunk, "tie_rule": "distance_then_original_sample_position",
    }
    identity["hybrid"] = {
        "engine": ENGINE, "knn_weight": KNN_WEIGHT, "multimodal_weight": MULTIMODAL_WEIGHT,
        "score": "0.60*knn_calibrated_q50_daily_percentile+0.40*fusion_calibrated_q50_daily_percentile",
        "rank": "same_date_finite_scores_method_average_pct_true",
        "join": "knn_valid_rows_left_join_multimodal",
        "fallback": "knn_weight_1_when_multimodal_prediction_is_missing_nonfinite_or_unordered",
        "health_policy": "causal_prediction_validity_only; post_evaluation_metrics_never_change_weights",
        "selection_mode": "cross_sectional_rank", "min_score": None,
    }
    return identity


def _common(evaluation: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    common = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
    common["scored"] = common["target"].notna() & common["label_end"].le(dates[-1])
    common["target"] = common["target"].where(common["scored"])
    if not common["scored"].any():
        raise ValueError("评估窗口内没有成熟标签")
    return common


def _blend_predictions(knn: pd.DataFrame, multimodal: pd.DataFrame) -> pd.DataFrame:
    """Blend only decision ranks; native quantiles keep their original semantics."""
    if not knn.index.equals(multimodal.index) or not knn.index.is_unique:
        raise ValueError("混合预测需要唯一且完全对齐的 KNN/多模态股票日")
    result = pd.DataFrame(index=knn.index)
    for prefix, table in (("knn", knn), ("multimodal", multimodal)):
        for column in models.QUANTILE_COLUMNS:
            result[f"{prefix}_{column}"] = table[column].to_numpy(dtype=float)
            raw = f"raw_{column}"
            if raw in table:
                result[f"{prefix}_{raw}"] = table[raw].to_numpy(dtype=float)
    knn_quantiles = knn[list(models.QUANTILE_COLUMNS)].to_numpy(dtype=float)
    multi_quantiles = multimodal[list(models.QUANTILE_COLUMNS)].to_numpy(dtype=float)
    knn_valid = np.isfinite(knn_quantiles).all(axis=1) & (knn_quantiles[:, 0] <= knn_quantiles[:, 1]) & (knn_quantiles[:, 1] <= knn_quantiles[:, 2])
    multi_valid = np.isfinite(multi_quantiles).all(axis=1) & (multi_quantiles[:, 0] <= multi_quantiles[:, 1]) & (multi_quantiles[:, 1] <= multi_quantiles[:, 2])
    dates = knn["date"] if "date" in knn else multimodal["date"]
    if "date" in knn and "date" in multimodal and not knn["date"].reset_index(drop=True).equals(multimodal["date"].reset_index(drop=True)):
        raise ValueError("混合预测的信号日期不一致")
    knn_rank = knn["q50"].where(knn_valid).groupby(dates).rank(method="average", pct=True)
    multi_rank = multimodal["q50"].where(multi_valid).groupby(dates).rank(method="average", pct=True)
    both = knn_valid & multi_valid
    result["knn_rank"] = knn_rank
    result["multimodal_rank"] = multi_rank
    result["hybrid_score"] = knn_rank.where(knn_valid)
    result.loc[both, "hybrid_score"] = (
        KNN_WEIGHT * knn_rank.loc[both] + MULTIMODAL_WEIGHT * multi_rank.loc[both]
    )
    result["effective_knn_weight"] = np.where(knn_valid, np.where(both, KNN_WEIGHT, 1.0), 0.0)
    result["effective_multimodal_weight"] = np.where(both, MULTIMODAL_WEIGHT, 0.0)
    result["fallback_reason"] = np.select(
        [~knn_valid, both], ["knn_missing_or_invalid", ""], default="multimodal_missing_or_invalid",
    )
    result["score"] = result["hybrid_score"]
    return result


def _fold(data, window, context, features, *, policy, cost_model, seed, scheduler, device, chunk):
    fold, returns = research._fold(
        data, window, context, features, seed=seed, policy=policy, cost_model=cost_model,
        scheduler=scheduler, include_raw_quantiles=True,
    )
    frame = data.frame
    eligible = frame["signal_eligible"].eq(True)
    settled = eligible & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])
    train, validation, cal = (frame.loc[mask & settled] for mask in window.masks(frame))
    evaluation = frame.loc[eligible & frame["date"].isin(window.evaluate)]
    rows = pd.concat([cal, evaluation])
    LOG.info("%s: %s KNN on %s calibration/evaluation rows", window.period, device, len(rows))
    forecast = forecast_experts(
        train, rows, names=(KNN_ARM,), knn_device=device, knn_query_chunk=chunk,
    )[KNN_ARM]
    quantiles, calibration = models._calibrate(forecast.quantiles, cal["target"].to_numpy(dtype=float))
    knn_prediction = models._prediction_table(quantiles, evaluation.index).join(pd.DataFrame(
        forecast.quantiles[len(cal):], index=evaluation.index,
        columns=[f"raw_{column}" for column in models.QUANTILE_COLUMNS],
    ))
    common = _common(evaluation, window.evaluate)
    folder = context.subdir("backtests", window.period)
    _write_table(folder / f"{KNN_ARM}_predictions.parquet", common.join(knn_prediction))
    knn_execution, knn_returns = research._execution(
        data, knn_prediction["score"], window.evaluate, context, folder, KNN_ARM, policy, cost_model,
    )
    knn_measures = score_quantiles(common.join(knn_prediction))
    knn_measures["all_signal_rows"] = len(common)
    fold["arms"][KNN_ARM] = {
        "prediction": knn_measures,
        "score_evidence": {
            "raw_evaluation_q50": health._distribution(forecast.quantiles[len(cal):, 1]),
            "calibrated_evaluation_q50": health._distribution(knn_prediction["q50"]),
            "qualified_rows": int(knn_prediction["score"].notna().sum()),
            "entry": {"selection_mode": "cross_sectional_rank", "score": "calibrated_q50",
                      "min_score": None, "comparison": "finite_cross_sectional_rank"},
        },
        "calibration": calibration, "execution": knn_execution, "strategy": asdict(policy),
        "signals": forecast.evidence,
    }

    native = pd.read_parquet(folder / f"{FUSION_ARM}_predictions.parquet")
    expected = evaluation[["date", "stock"]].reset_index(drop=True)
    if len(native) != len(evaluation) or not native[["date", "stock"]].reset_index(drop=True).equals(expected):
        raise ValueError(f"{window.period} 的多模态融合预测与 KNN 股票日不对齐")
    native = native.set_index(evaluation.index)
    hybrid = _blend_predictions(knn_prediction, native)
    _write_table(folder / f"{ARM}_predictions.parquet", common.join(hybrid))
    execution, hybrid_returns = research._execution(
        data, hybrid["score"], window.evaluate, context, folder, ARM, policy, cost_model,
    )
    both = hybrid["effective_multimodal_weight"].eq(MULTIMODAL_WEIGHT)
    fallback = hybrid["effective_knn_weight"].eq(1.0)
    invalid = hybrid["score"].isna()
    hybrid_evidence = {
        "rows": len(hybrid), "pinball": None, "coverage_80": None, "width_80": None,
        "all_signal_rows": len(common), "semantics": "decision_score_only_no_hybrid_quantiles",
    }
    coverage = {
        "evaluation_rows": len(hybrid), "full_60_40_rows": int(both.sum()),
        "knn_fallback_rows": int(fallback.sum()), "invalid_knn_rows": int(invalid.sum()),
        "full_60_40_rate": float(both.mean()) if len(hybrid) else None,
        "knn_fallback_rate": float(fallback.mean()) if len(hybrid) else None,
        "fallback_reasons": {key: int(value) for key, value in hybrid["fallback_reason"].value_counts().items() if key},
        "hybrid_score": health._distribution(hybrid["hybrid_score"]),
        "knn_rank": health._distribution(hybrid["knn_rank"]),
        "multimodal_rank": health._distribution(hybrid["multimodal_rank"]),
        "entry": {"selection_mode": "cross_sectional_rank", "score": "hybrid_score", "min_score": None},
        "post_evaluation_health_used_for_weights": False,
    }
    fold["arms"][ARM] = {
        "prediction": hybrid_evidence, "score_evidence": coverage,
        "execution": execution, "strategy": asdict(policy),
    }
    for cost in COSTS:
        returns[cost][KNN_ARM] = knn_returns[cost]
        returns[cost][ARM] = hybrid_returns[cost]
    fold["hybrid"] = {
        "engine": ENGINE, "knn_weight": KNN_WEIGHT, "multimodal_weight": MULTIMODAL_WEIGHT,
        "rank": "same_date_finite_calibrated_q50_average_pct",
        "join": "knn_valid_rows_left_join_multimodal",
        "post_evaluation_health_used_for_weights": False,
        "coverage": coverage, "knn_backend": forecast.evidence["backend"],
        "fusion_weights": fold.get("fusion", {}).get("winner", {}).get("weights", {}),
        "multimodal_validation_health": fold.get("model_health"),
    }
    write_json(context.subdir("models", window.period) / "hybrid.json", {
        "engine": ENGINE, "knn_weight": KNN_WEIGHT, "multimodal_weight": MULTIMODAL_WEIGHT,
        "fusion_weights": fold.get("fusion", {}).get("winner", {}).get("weights", {}),
        "knn": {"calibration": calibration, "signal": forecast.evidence},
        "coverage": coverage, "training_device": forecast.evidence["backend"]["actual_device"],
    })
    return fold, returns


def _verify_outputs(root: Path, folds: list[dict], matrix: pd.DataFrame) -> dict:
    checks = []
    arms = (ARM, KNN_ARM, FUSION_ARM)
    for fold in folds:
        for arm in arms:
            for cost in COSTS:
                metrics = fold["arms"][arm]["execution"][cost]
                if not metrics.get("verification") or not all(metrics["verification"].values()):
                    raise ValueError(f"{fold['period']}/{arm}/{cost} 的现金、权益或下一交易日核验失败")
                tables = {
                    name: pd.read_parquet(root / "backtests" / fold["period"] / f"{arm}_{cost}_{name}.parquet")
                    for name in ("daily", "orders", "trades", "positions", "selection_daily")
                }
                daily, orders, trades, positions = (tables[name] for name in ("daily", "orders", "trades", "positions"))
                if (daily["date"].duplicated().any() or len(daily) != fold["evaluation_sessions"]
                        or len(trades) != metrics["trade_count"]
                        or len(orders) != metrics["order_count"] + metrics.get("pretrade_rejected_count", 0)
                        or len(tables["selection_daily"]) != len(daily)
                        or (not orders.empty and orders["date"].min() < daily["date"].min())
                        or (not positions.empty and positions["date"].max() > daily["date"].max())):
                    raise ValueError(f"{fold['period']}/{arm}/{cost} 的混合账本工件不完整")
                checks.append({"period": fold["period"], "arm": arm, "cost": cost,
                               "daily": len(daily), "orders": len(orders), "trades": len(trades)})
    if len(matrix.loc[matrix["arm"].isin(arms)]) != len(checks):
        raise ValueError("混合实验矩阵与已核验账本数量不一致")
    return {
        "ledger_tables": len(checks), "matrix_rows": len(checks), "total_matrix_rows": len(matrix),
        "verified_matrix_rows": len(checks),
        "verified_arms": list(arms), "all_daily_dates_unique": True,
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

        payload, device, chunk = _options(spec)
        identity = prediction_identity(spec)
        model_spec = identity["spec"]
        full = model_spec["pool"] == "full"
        key = stable_hash(identity)
        commission_bps, absolute_policy, cost_model = research._execution_spec(payload)
        policy = replace(absolute_policy, selection_mode="cross_sectional_rank", min_score=None)
        seed = model_spec.get("seed", research.SEED)
        source_identity = identity["research"]
        store = ExpertArtifactStore()
        import catboost
        from catboost.utils import get_gpu_device_count

        if get_gpu_device_count() < 1:
            raise RuntimeError("多模态 CatBoost 训练需要可用 CUDA 设备")
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
            raise ValueError("混合面板在准备前后输入身份不一致")
        data, macro = research._verified_macro(raw_data, research._sources().get("overseas"))
        data, sector = research._verified_sector(
            data, model_spec.get("sector_view", False), model_spec.get("sector_provider", "capco"),
        )
        del raw_data
        features = {"domestic_gpu": tuple(BASE_FEATURES)}
        if sector["enabled"]:
            features["sector_gpu"] = (*BASE_FEATURES, *sector["features"])
        if macro["verified_sessions"]:
            features["overseas_gpu"] = (*BASE_FEATURES, *research.MACRO_FEATURES)
        coverage = {
            **data.coverage, "overseas": macro, "sector_view": sector, "source_identity": source_identity,
            "model_features": list(KNN_FEATURES), "multimodal_features_used": True,
            "hybrid": identity["hybrid"],
        }
        windows = research.plan_expert_windows(data, model_spec)
        if [window.period for window in windows if window.period in DEVELOPMENT] != list(DEVELOPMENT):
            raise ValueError("混合研究必须包含顺序固定的三个开发折")
        checks = _features_validation(data.frame, data.sessions)
        actual_costs = {cost: asdict(scaled_study_cost(multiplier, cost_model)) for cost, multiplier in COSTS.items()}
        actual_end = data.sessions[-1].date().isoformat()
        config = {
            **source_identity, "sources": research._reference_values(references),
            "source_run_id": research.BASELINE_ID, "end": actual_end, "seed": seed,
            "catboost_version": catboost.__version__, "stock_cap": len(data.roster), "model_spec": model_spec,
            "model_grid": expert_gpu.GRID if full else research.GPU_GRID,
            "iterations": expert_gpu.ITERATIONS if full else research.GPU_ITERATIONS,
            "quantiles": models.QUANTILE_LEVELS, "features": features, "training_engine": ENGINE,
            "knn": identity["knn"], "hybrid": identity["hybrid"], "development_periods": list(DEVELOPMENT),
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
        folds, all_returns, review, pilot = [], [], None, None
        with expert_gpu.DualGPU() if full else nullcontext() as scheduler:
            if scheduler is not None:
                update("running", "gpu_resource_pilot")
                train_mask = windows[0].masks(data.frame)[0] & data.frame["signal_eligible"].eq(True) & np.isfinite(data.frame["target"])
                train = data.frame.loc[train_mask, ["stock", "date", "target", *BASE_FEATURES]]
                pilot = scheduler.pilot(train, BASE_FEATURES, seed)
                del train, train_mask
                write_json(context.subdir("reports") / "gpu_resource_pilot.json", pilot)
            for window in windows:
                update("running", window.period)
                with scheduler.measure() if scheduler is not None else nullcontext() as resource:
                    fold, returns = _fold(
                        data, window, context, features, policy=policy, cost_model=cost_model,
                        seed=seed, scheduler=scheduler, device=device, chunk=chunk,
                    )
                if resource is not None:
                    fold["gpu_resources"] = resource
                if window.period == "historical_replay":
                    review = fold | {"scope": "retrospective_review", "included_in_champion_selection": False}
                    fold = review
                else:
                    folds.append(fold)
                    all_returns.append(returns)
                write_json(context.subdir("reports") / f"{window.period}.json", fold)
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
        sector_report = research._research_sector_artifacts(
            root, data, folds, ARM, model_spec.get("sector_provider", "capco"),
        )
        final_portfolio["sector_resonance"] = "reports/sector_resonance.json"
        matrix = _summary_table(all_folds)
        _write_table(context.subdir("reports") / "cost_and_prediction.csv", matrix)
        verification = _verify_outputs(root, all_folds, matrix)
        write_json(context.subdir("reports") / "artifact_verification.json", verification)
        comparisons = {
            f"hybrid_vs_{reference}": {
                cost: paired_block_interval(
                    [(returns[cost][ARM], returns[cost][reference]) for returns in all_returns],
                    seed=seed, block_size=research.BOOTSTRAP_MAIN,
                ) for cost in COSTS
            } for reference in (KNN_ARM, FUSION_ARM)
        }
        decision = {
            "state": "research_only", "status": "research_only", "registry_written": False,
            "historical_numerical_gate": False,
            "point_in_time_verified": bool(data.coverage["point_in_time"]["verified"]),
            "qualified_forward_results": False,
            "reasons": [
                "60/40 仅融合同日决策排名，不构造混合收益分位数。",
                "历史复核和开发折均不是独立前瞻证据；本运行不提升正式冠军。",
                "生产回退由独立冻结清单和同口径双倍成本复合收益比较控制。",
            ],
        }
        report = {
            "run": context.to_dict(), "input_coverage": coverage, "validation": checks,
            "experts": {
                KNN_ARM: {"status": "active", "role": "primary_60_percent", "requested_device": device},
                FUSION_ARM: {"status": "active", "role": "multimodal_40_percent", "training_device": "GPU:0,1" if full else "GPU:0"},
                ARM: {"status": "active", "role": "decision_rank_blend"},
            },
            "model_spec": model_spec, "training_engine": ENGINE, "portfolio": asdict(policy),
            "commission_bps": commission_bps, "execution_costs": actual_costs,
            "execution_evidence": config["execution_evidence"], "prediction_key": key,
            "prediction_evidence_sha256": stable_hash(predicted_files), "reused_predictions_from": None,
            "fit_executed": True, "folds": folds, "historical_review": review,
            "oos_block_bootstrap": research._paired(all_returns, research.BOOTSTRAP_MAIN, seed),
            "hybrid_comparisons": comparisons, "decision": decision,
            "final_portfolio": final_portfolio, "sector_resonance": sector_report,
            "artifact_verification": verification, "artifact_manifest": str(manifest),
            "gpu_resource_pilot": pilot, "ablation_block_bootstrap": {}, "block_sensitivity": {},
            "interpretation": {
                "execution": "三方使用共同冻结股票池，以前一收盘信号在次日开盘模拟；双档成本并逐折从现金开始。",
                "score": "混合分是 60% KNN 与 40% 多模态校准 q50 的同日百分位加权和，不是收益分位数。",
                "fallback": "多模态缺失、非有限或分位数无序时仅使用 KNN；事后预测或收益指标不改变历史权重。",
                "scope": "历史开发研究与回放，不是独立前瞻收益证明。",
            },
        }
        report_dir = context.subdir("reports")
        write_json(report_dir / "selected_hyperparameters.json", {
            fold["period"]: {"hybrid": fold["hybrid"], "models": fold["models"], "fusion": fold["fusion"]}
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
    parser.add_argument("--knn-device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--knn-query-chunk", type=int, default=256)
    parser.add_argument("--activate", action="store_true", help="双倍成本复合收益高于当前生产时切换混合回退")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = args.run_id or make_run_id(TASK, HISTORICAL_END)
    path = run(run_id=run_id, spec={
        "pool": "full", "knn_device": args.knn_device, "knn_query_chunk": args.knn_query_chunk,
    } if args.full else {"knn_device": args.knn_device, "knn_query_chunk": args.knn_query_chunk})
    if args.activate:
        from my_strategy.model_registry import expert_fallback

        proof = expert_fallback.activate(path.parent.parent)
        print({"report": str(path), "production": proof})
    else:
        print(path)


if __name__ == "__main__":
    main()
