"""Execute a frozen multimodal archive with conditional KNN residual calibration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.domestic_runner import _sha
from my_strategy.latent_quant.evaluation import score_quantiles
from my_strategy.latent_quant.execution import Strategy, StudyCostModel
from my_strategy.latent_quant.runner import COSTS, _summary_table
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_calibration as calibration
from . import expert_research as research
from . import expert_score_health as health
from .runner import _write_table, write_json


SOURCE = "20261126-154812-998943-20261120-expert-research-d3e68b3"
ARM = "fusion"


def _fold(archive: calibration.Archive, source_fold: dict, context, *, method: str = "knn_residual") -> tuple[dict, dict]:
    period = source_fold["period"]
    rows, variants, evidence = calibration.recover_fold(archive, source_fold)
    if method == "online_knn_residual":
        from . import expert_online_knn_calibration as online

        values, online_evidence = online.recover_online(archive, source_fold, rows, variants, evidence)
        evidence = {**evidence, "knn": online_evidence}
    else:
        values = variants[method]
    frame = archive.data.frame
    evaluation = frame.loc[
        frame["date"].between(pd.Timestamp(source_fold["evaluation_start"]), pd.Timestamp(source_fold["evaluation_end"]))
        & frame["signal_eligible"].eq(True)
    ]
    if not rows[["date", "stock"]].reset_index(drop=True).equals(evaluation[["date", "stock"]].reset_index(drop=True)):
        raise ValueError(f"{period} 的 KNN 校准预测与冻结评估行不一致")
    dates = pd.DatetimeIndex(evaluation["date"].unique())
    prediction = models._prediction_table(values, evaluation.index)
    common = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
    common["scored"] = common["target"].notna() & common["label_end"].le(dates[-1])
    common["target"] = common["target"].where(common["scored"])
    joined = common.join(prediction)
    measures = score_quantiles(joined)
    measures["all_signal_rows"] = len(joined)
    policy = Strategy(**archive.config["portfolio"])
    costs = StudyCostModel(**archive.config["expense_model"])
    folder = context.subdir("backtests", period)
    _write_table(folder / f"{ARM}_predictions.parquet", joined)
    execution, returns = research._execution(
        archive.data, prediction["score"], dates, context, folder, ARM, policy, costs,
    )
    raw = variants["raw"]
    score_evidence = {
        "raw_evaluation_q50": health._distribution(raw[:, 1]),
        "calibrated_evaluation_q50": health._distribution(values[:, 1]),
        "raw_qualified_rows": int(np.sum(raw[:, 1] > policy.min_score)),
        "qualified_rows": int(np.sum(values[:, 1] > policy.min_score)),
        "entry": {"score": "knn_calibrated_q50", "min_score": policy.min_score, "comparison": "strictly_greater"},
    }
    result = {
        "period": period,
        "evaluation_start": source_fold["evaluation_start"],
        "evaluation_end": source_fold["evaluation_end"],
        "evaluation_sessions": len(dates),
        "train": copy.deepcopy(source_fold["train"]),
        "validation": copy.deepcopy(source_fold["validation"]),
        "calibration": copy.deepcopy(source_fold["calibration"]),
        "models": {
            "train": copy.deepcopy(source_fold["models"]["train"]),
            "validation": copy.deepcopy(source_fold["models"]["validation"]),
            "calibration": copy.deepcopy(source_fold["models"]["calibration"]),
            "arms": {},
        },
        "fusion": {
            "method": evidence["knn"]["method"],
            "source_run_id": archive.root.name,
            "source_fusion_weights": evidence["weights"],
            "knn": evidence["knn"],
        },
        "arms": {
            ARM: {
                "prediction": measures,
                "score_evidence": score_evidence,
                "calibration": evidence["knn"],
                "execution": execution,
                "strategy": archive.config["portfolio"],
            },
        },
        "contrasts": {},
        "ablations": {},
        "model_health": {"passed": True, "basis": "frozen_native_models_with_finite_ordered_knn_quantiles"},
        "source_fold_sha256": stable_hash(source_fold),
        "calibration_recovery": {
            "method": evidence["recovery"],
            "offset_replay_max_error": evidence["calibration_offset_replay_max_error"],
            "inference": evidence["inference"],
        },
    }
    write_json(context.subdir("reports") / f"{period}.json", result)
    return result, returns


def run(*, source_run_id: str = SOURCE, run_id: str, method: str = "knn_residual") -> Path:
    if method not in {"knn_residual", "online_knn_residual"}:
        raise ValueError("不支持的 KNN 校准方法")
    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号无效")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    status = root / "status.json"
    write_json(status, {"status": "running", "phase": "verify_archive"})
    try:
        archive = calibration.open_archive(source_run_id)
        store = ExpertArtifactStore()
        references = store.read_manifest(source_run_id)["files"]
        if research._reference_values(references) != archive.config["sources"]:
            raise ValueError("来源模型的冻结输入清单不一致")
        calibration_identity = {
            "method": method,
            "neighbors": calibration.KNN_NEIGHBORS,
            "per_date_sample_cap": calibration.KNN_PER_DATE,
            "features": list(calibration.KNN_FEATURES),
            "selection": ("fixed_before_outer_evaluation" if method == "knn_residual" else
                          "causal_online_updates;_method_reviewed_after_static_outer_results"),
            "implementation_sha256": {
                "expert_calibration.py": _sha(Path(calibration.__file__)),
                "expert_knn_calibration_replay.py": _sha(Path(__file__)),
                **({"expert_online_knn_calibration.py": _sha(Path(__file__).with_name("expert_online_knn_calibration.py"))}
                   if method == "online_knn_residual" else {}),
            },
        }
        if method == "online_knn_residual":
            from . import expert_online_knn_calibration as online

            calibration_identity["lookback_sessions"] = online.LOOKBACK_SESSIONS
        key = stable_hash({"source_prediction_key": archive.config["prediction_key"], "calibration": calibration_identity})
        config = copy.deepcopy(archive.config) | {
            "prediction_key": key,
            "model_source_run_id": source_run_id,
            "calibration": calibration_identity,
        }
        context = create_run_context(
            task="expert-knn-calibration-replay", as_of_date=archive.config["end"], config=config,
            data_version=stable_hash(config["sources"]), seed=archive.config["seed"], run_id=run_id,
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", archive.report["input_coverage"])
        folds, review = [], None
        baseline = {}
        for source_fold in research._archived_folds(archive.report):
            period = source_fold["period"]
            write_json(status, {"status": "running", "phase": period})
            fold, _ = _fold(archive, source_fold, context, method=method)
            baseline[period] = source_fold["arms"][ARM]["execution"]
            if period == "historical_replay":
                review = fold | {"scope": "retrospective_review", "included_in_champion_selection": False}
            else:
                folds.append(fold)
        all_folds = [*folds, *([review] if review else [])]
        predicted_files = research._prediction_evidence(root, all_folds)
        write_json(root / research.PREDICTION_EVIDENCE, {"prediction_key": key, "files": predicted_files})
        manifest = store.write_manifest(run_id, references, metadata={
            "source_run_id": source_run_id, "task": "expert-knn-calibration-replay",
        })
        final_portfolio = research._final_portfolio(root, folds, folds[-1]["period"], ARM)
        write_json(status, {"status": "running", "phase": "sector_artifacts"})
        sector = research._research_sector_artifacts(
            root, archive.data, folds, ARM, archive.config["model_spec"].get("sector_provider", "capco"),
        )
        final_portfolio["sector_resonance"] = "reports/sector_resonance.json"
        decision = {
            "state": "research_only",
            "historical_numerical_gate": False,
            "point_in_time_verified": archive.report["decision"].get("point_in_time_verified") is True,
            "qualified_forward_results": False,
            "reasons": [
                "KNN 残差校准目前是冻结历史归档上的独立回放，尚无自身封存的 60 日前瞻证据。",
                "该运行不提升正式冠军；生产回退仍须按相同口径收益单独比较。",
            ],
        }
        baseline_returns = {
            period: {
                cost: baseline[period][cost]["total_return"]
                for cost in COSTS
            }
            for period in baseline
        }
        report = {
            "run": context.to_dict(),
            "input_coverage": copy.deepcopy(archive.report["input_coverage"]),
            "validation": copy.deepcopy(archive.report["validation"]),
            "experts": copy.deepcopy(archive.report["experts"]) | {
                "knn_residual_calibrator": {"status": "active", "role": "calibration", "training_device": "CPU"},
            },
            "model_spec": copy.deepcopy(archive.config["model_spec"]),
            "portfolio": copy.deepcopy(archive.config["portfolio"]),
            "commission_bps": archive.config["commission_bps"],
            "execution_costs": copy.deepcopy(archive.config["execution_costs"]),
            "execution_evidence": copy.deepcopy(archive.config["execution_evidence"]),
            "training_engine": "multimodal_knn_residual_calibration",
            "prediction_key": key,
            "prediction_evidence_sha256": stable_hash(predicted_files),
            "reused_predictions_from": source_run_id,
            "fit_executed": True,
            "folds": folds,
            "historical_review": review,
            "decision": decision,
            "final_portfolio": final_portfolio,
            "sector_resonance": sector,
            "baseline": {
                "source_run_id": source_run_id,
                "method": "additive_residual_quantiles",
                "fold_returns": baseline_returns,
                "development_double_compound_return": float(np.prod([
                    1 + fold["arms"][ARM]["execution"]["double"]["total_return"] for fold in archive.report["folds"]
                ]) - 1),
            },
            "oos_block_bootstrap": {},
            "ablation_block_bootstrap": {},
            "block_sensitivity": {},
            "artifact_manifest": str(manifest),
            "interpretation": {
                "execution": "同一冻结原生融合预测，仅将加性残差校准替换为配置指定的因果 KNN 条件残差分位数；交易成本和账本规则不变。",
                "scope": "历史校准回放，不是独立前瞻验收，也不证明未来收益。",
            },
        }
        reports = context.subdir("reports")
        write_json(reports / "selected_hyperparameters.json", {
            fold["period"]: {"models": fold["models"], "fusion": fold["fusion"]} for fold in all_folds
        })
        _write_table(reports / "cost_and_prediction.csv", _summary_table(all_folds))
        destination = reports / research.REPORT
        write_json(destination, report)
        write_json(status, {"status": "completed", "phase": "complete", "release_state": "research_only"})
        return destination
    except Exception as error:
        write_json(status, {"status": "failed", "phase": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=SOURCE)
    parser.add_argument("--run-id")
    parser.add_argument("--method", choices=["knn_residual", "online_knn_residual"], default="knn_residual")
    args = parser.parse_args()
    print(run(source_run_id=args.source_run_id, run_id=args.run_id or make_run_id("expert-knn-calibration"), method=args.method))


if __name__ == "__main__":
    main()
