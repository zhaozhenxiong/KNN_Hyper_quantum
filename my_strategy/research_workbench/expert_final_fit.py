"""Fit and seal the single forward model for one completed research report.

The historical report is the immutable anchor: it is hashed before fitting and
never updated with model information.  The model binds that hash, then the two
final reports bind the model and native files, avoiding a circular identity.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import date
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.latent_quant import forward, models
from my_strategy.latent_quant.domestic_runner import _sha
from my_strategy.storage.expert_artifacts import _publish

from . import expert_forward, expert_gpu, expert_research, expert_score_health
from .runner import Window, _write_table, write_json


FINAL_FIT_REPORT = "reports/expert_final_fit.json"
UNIFIED_REPORT = "reports/unified_expert.json"
MODEL = "models/forward/model.json"
MODEL_ARMS = {"domestic_gpu", "sector_gpu", "overseas_gpu"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _final_window(
    sessions: pd.DatetimeIndex, *, train_sessions: int = 315,
    validation_sessions: int = 35, calibration_sessions: int = 35,
) -> Window:
    """Use the established lookback; the synthetic boundary is never an input row."""
    sessions = pd.DatetimeIndex(sessions)
    required = train_sessions + validation_sessions + calibration_sessions
    if (min(train_sessions, validation_sessions, calibration_sessions) < 1
            or len(sessions) < required or not sessions.is_monotonic_increasing or not sessions.is_unique):
        raise ValueError("最终拟合需要有序且足够的独立交易时段")
    cutoff = sessions[-1]
    return Window(
        "final_fit", sessions[-required], sessions[-validation_sessions - calibration_sessions],
        sessions[-calibration_sessions], pd.DatetimeIndex([cutoff + pd.Timedelta(days=1)]),
    )


def _split(
    frame: pd.DataFrame, sessions: pd.DatetimeIndex, cutoff: str, **window_sizes,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Window]:
    """Reject future observations/known labels before constructing mature partitions."""
    sessions = pd.DatetimeIndex(sessions)
    through = pd.Timestamp(date.fromisoformat(cutoff))
    if not len(sessions) or sessions[-1] != through:
        raise ValueError("最终拟合输入截止必须等于最后一个冻结交易日")
    dates = pd.to_datetime(frame["date"])
    labels = pd.to_datetime(frame["label_end"])
    known = frame["target"].notna()
    if (dates.isna().any() or dates.gt(through).any()
            or (known & (labels.isna() | labels.gt(through) | labels.le(dates))).any()):
        raise ValueError("最终拟合不得读取输入截止后的观察或未来标签")
    window = _final_window(sessions, **window_sizes)
    eligible = frame["signal_eligible"].eq(True)
    settled = eligible & known & np.isfinite(frame["target"]) & labels.gt(dates)
    train, validation, calibration = [frame.loc[mask & settled].copy() for mask in window.masks(frame)]
    if any(rows.empty for rows in (train, validation, calibration)):
        raise ValueError("最终拟合需要非空且标签成熟的训练、验证和校准段")
    if not (train["label_end"].max() < window.valid_start
            and validation["label_end"].max() < window.calibration_start
            and calibration["label_end"].max() <= through):
        raise ValueError("最终拟合分区标签未在进入下一分区前成熟")
    replay = frame.loc[eligible].tail(64).copy()
    if replay.empty:
        raise ValueError("最终拟合缺少精确回放输入")
    return train, validation, calibration, replay, window


def _point_in_time(report: dict) -> bool:
    actual = report["input_coverage"]["point_in_time"]["verified"] is True
    if report["decision"].get("point_in_time_verified") is not actual:
        raise ValueError("研究报告的点时状态与输入证据不一致")
    return actual


def _input_hashes(config: dict) -> dict[str, str]:
    return {name: value["sha256"] for name, value in sorted(config["sources"].items())}


def _universe(config: dict, sessions: pd.DatetimeIndex) -> dict:
    return {
        "pool": config["model_spec"]["pool"], "stock_count": int(config["stock_cap"]),
        "input_through": config["end"], "sessions": len(sessions),
    }


def _version_payload(model: dict, model_sha: str, config: dict) -> dict:
    return {
        "run_id": model["run_id"], "arm": model["arm"],
        "research_report_sha256": model["research_report_sha256"],
        "config_sha256": model["config_sha256"], "final_model_sha256": model_sha,
        "native_file_sha256": model["files"], "input_sha256": _input_hashes(config),
        "final_fit_code_sha256": config["final_fit_code_sha256"],
    }


def verify_final_fit(root: Path, *, source_root: Path | None = None) -> dict:
    """Verify the complete content closure; False PIT remains valid research evidence."""
    root = Path(root)
    source = Path(source_root) if source_root is not None else root
    report = _json(source / "reports/expert_research.json")
    config = _json(source / "config/study.json")
    model = _json(root / MODEL)
    summary = _json(root / FINAL_FIT_REPORT)
    unified = _json(root / UNIFIED_REPORT)
    report_sha = _sha(source / "reports/expert_research.json")
    config_sha = _sha(source / "config/study.json")
    model_sha = _sha(root / MODEL)
    arm = report["final_portfolio"]["arm"]
    if (model.get("schema_version") != 1 or model.get("run_id") != source.name
            or model.get("arm") != arm or model.get("research_report_sha256") != report_sha
            or model.get("config_sha256") != config_sha
            or model.get("point_in_time_verified") is not _point_in_time(report)
            or model.get("training_input_through") != config["end"]
            or date.fromisoformat(model["training_max_label_date"]) > date.fromisoformat(config["end"])):
        raise ValueError("最终模型未绑定不可变研究、配置和实际拟合时点")
    native = {role: _sha(root / role) for role in model["files"]}
    if native != model["files"]:
        raise ValueError("最终模型原生文件哈希不匹配")
    payload = _version_payload(model, model_sha, config)
    version = stable_hash(payload)
    if (unified.get("schema_version") != 1 or unified.get("model_version") != version
            or unified.get("model_version_basis") != payload or summary.get("model_version") != version
            or unified.get("run_id") != source.name or unified.get("arm") != arm
            or summary.get("run_id") != source.name or summary.get("arm") != arm
            or summary.get("final_model_sha256") != model_sha
            or unified.get("final_model_sha256") != model_sha
            or unified.get("research_report_sha256") != report_sha
            or unified.get("config_sha256") != config_sha
            or unified.get("native_file_sha256") != native
            or unified.get("input_sha256") != _input_hashes(config)
            or unified.get("final_fit_report_sha256") != _sha(root / FINAL_FIT_REPORT)
            or unified.get("research_status", {}).get("point_in_time_verified") is not _point_in_time(report)
            or unified.get("release_status", {}).get("auto_promoted") is not False):
        raise ValueError("统一最终拟合清单的内容哈希未闭合")
    return unified


def fit_final_fit(root: Path, frame: pd.DataFrame, *, scheduler=None, **window_sizes) -> Path:
    """Produce one new bundle; never copy or modify another run's forward assets."""
    root = Path(root)
    report_path = root / "reports/expert_research.json"
    config_path = root / "config/study.json"
    report, config = _json(report_path), _json(config_path)
    if report.get("run", {}).get("run_id") != root.name:
        raise ValueError("最终拟合研究报告与运行目录不一致")
    report_sha, config_sha = _sha(report_path), _sha(config_path)
    sessions = pd.DatetimeIndex(sorted(frame["date"].unique()))
    train, validation, calibration, replay, window = _split(
        frame, sessions, config["end"], **window_sizes,
    )
    arm = report["final_portfolio"]["arm"]
    if arm not in {"fusion", "domestic_gpu"}:
        raise ValueError("最终拟合只接受研究报告已证实的统一策略臂")
    available = {
        name for name, details in report["experts"].items()
        if details.get("status") == "active" and name in MODEL_ARMS
    }
    features = {
        name: tuple(columns) for name, columns in config["features"].items()
        if name in available and (name == "domestic_gpu" or arm == "fusion")
    }
    required_rules = {"historical_quantiles", "supertrend_atr"} if arm == "fusion" else {"historical_quantiles"}
    if "domestic_gpu" not in features or any(
            report["experts"].get(name, {}).get("status") != "active" for name in required_rules):
        raise ValueError("最终拟合缺少研究报告已证实的规则或国内原生专家")
    full = config["model_spec"]["pool"] == "full"
    owned_scheduler = expert_gpu.DualGPU() if full and scheduler is None else None
    if owned_scheduler is not None:
        scheduler = owned_scheduler
    published = []
    try:
        with TemporaryDirectory(dir=root, prefix=".final-fit-") as temporary:
            stage = Path(temporary)
            model_root = stage / "models/forward"
            model_root.mkdir(parents=True)
            devices = {name: expert_research._inference_audit() for name in features}
            if arm == "domestic_gpu":
                weights = {"domestic_gpu": 1.0}
                fusion = {"method": "selected_single_native_arm", "selection_scope": "train_only"}
            else:
                weights, fusion, _ = expert_research._inner_fusion(
                    train, features, int(config["seed"]), devices, scheduler,
                    matched_fit=config.get("fit_protocol") == expert_research.MATCHED_PROTOCOL,
                )
            needed = set(weights).intersection(MODEL_ARMS) | {"domestic_gpu"}
            features = {name: features[name] for name in sorted(needed)}
            rows = pd.concat([calibration, replay])
            raw = expert_research._rules(train, rows)
            native_evidence = {}
            with (expert_gpu.DualGPU() if full and scheduler is None else nullcontext()) as fitted_scheduler:
                active_scheduler = scheduler or fitted_scheduler
                for name, columns in tuple(features.items()):
                    source = train.dropna(subset=list(expert_research.MACRO_FEATURES)) if name == "overseas_gpu" else train
                    valid = validation.dropna(subset=list(expert_research.MACRO_FEATURES)) if name == "overseas_gpu" else validation
                    if source.empty or valid.empty:
                        raise ValueError(f"{name} 缺少独立的原生拟合或验证样本")
                    fitted, details = (
                        expert_research._tune_full(source, valid, columns, int(config["seed"]), devices[name], active_scheduler)
                        if active_scheduler is not None else
                        expert_research._tune(source, valid, columns, int(config["seed"]), devices[name])
                    )
                    median = expert_score_health._learned_median_health(
                        details["winner"]["tree_counts"][1], details["winner"].get("validation_q50", []),
                    )
                    native_evidence[name] = {**details, "median_health": median}
                    if weights.get(name, 0) > 0 and median["collapsed"]:
                        if name == "domestic_gpu":
                            raise ValueError(f"最终拟合拒绝 {name} 的单树中位分数塌缩")
                        # Validation decides eligibility only; replacement weights
                        # are still selected exclusively by the same train-only OOF.
                        initial = fusion
                        features = {"domestic_gpu": features["domestic_gpu"]}
                        weights, selected, _ = expert_research._inner_fusion(
                            train, features, int(config["seed"]), devices, active_scheduler,
                            matched_fit=config.get("fit_protocol") == expert_research.MATCHED_PROTOCOL,
                        )
                        fusion = {
                            "method": "validation_health_abstention_and_train_oof_reweight",
                            "eligibility_scope": "final_validation", "weight_scope": "same_train_inner_oof",
                            "abstained": {name: median}, "initial": initial, "selected": selected,
                        }
                        continue
                    raw[name] = expert_research._expert_raw(
                        fitted, rows, columns, devices[name],
                        fallback=raw.get("domestic_gpu") if name == "overseas_gpu" else None,
                    )
                    folder = model_root / name
                    folder.mkdir()
                    for quantile, estimator in zip(models.QUANTILE_COLUMNS, fitted, strict=True):
                        estimator.save_model(str(folder / f"{quantile}.cbm"), format="cbm")
            if arm == "fusion":
                raw["fusion"] = expert_research._fuse(weights, raw)
            arms = expert_forward._arms(arm)
            offsets, calibrations, expected = {}, {}, []
            labels = calibration["target"].to_numpy(dtype=float)
            for name in sorted(arms):
                calibrated, calibrations[name] = models._calibrate(raw[name], labels)
                offsets[name] = calibrations[name]["offsets"]
                all_calibrated = np.sort(np.sort(raw[name], axis=1) + np.asarray(offsets[name]), axis=1)
                if not np.isfinite(all_calibrated).all() or (np.diff(all_calibrated, axis=1) < 0).any():
                    raise ValueError(f"最终拟合 {name} 的分位数非有限或无序")
                expected.append(
                    replay[["date", "stock"]].join(models._prediction_table(calibrated, replay.index)).assign(arm=name)
                )
            expected_replay = pd.concat(expected, ignore_index=True)
            _write_table(model_root / "train.parquet", train)
            native_files = {
                path.relative_to(stage).as_posix(): _sha(path)
                for path in sorted(model_root.rglob("*")) if path.is_file()
            }
            model = {
                "schema_version": 1, "run_id": root.name, "arm": arm,
                "research_report_sha256": report_sha, "config_sha256": config_sha,
                "point_in_time_verified": _point_in_time(report),
                "sealed_at": forward._now().isoformat(),
                "training_max_label_date": max(
                    rows["label_end"].max() for rows in (train, validation, calibration)
                ).date().isoformat(),
                "training_input_through": config["end"], "weights": weights,
                "features": {name: list(columns) for name, columns in features.items()},
                "offsets": offsets, "files": native_files,
                "final_fit_code_sha256": config["final_fit_code_sha256"],
            }
            write_json(stage / MODEL, model)
            # Fit-time macro inputs were already availability-filtered by research;
            # capture retains the stricter contemporaneous official-header check.
            replayed = expert_forward._forecast_assets(
                model, model_root, model_root / "train.parquet", arms,
                replay.drop(columns=["target", "label_end"]), forward._timestamp(model["sealed_at"]),
                verify_macro_availability=False,
            )
            pd.testing.assert_frame_equal(replayed, expected_replay, check_exact=True)
            model_sha = _sha(stage / MODEL)
            payload = _version_payload(model, model_sha, config)
            version = stable_hash(payload)
            status = {
                "state": report["decision"]["state"],
                "historical_numerical_gate": report["decision"].get("historical_numerical_gate") is True,
                "point_in_time_verified": _point_in_time(report),
                "reused_predictions_from": report.get("reused_predictions_from"),
            }
            summary = {
                "schema_version": 1, "model_version": version, "run_id": root.name, "arm": arm,
                "research_report_sha256": report_sha, "config_sha256": config_sha,
                "final_model_sha256": model_sha, "native_file_sha256": native_files,
                "input_sha256": _input_hashes(config), "universe": _universe(config, sessions),
                "research_status": status,
                "release_status": {"state": "unregistered", "auto_promoted": False},
                "split": {
                    "train": models._sample_evidence(train), "validation": models._sample_evidence(validation),
                    "calibration": models._sample_evidence(calibration),
                    "window": {"train_sessions": 315, "validation_sessions": 35, "calibration_sessions": 35},
                    "refit_on_validation_or_calibration": False,
                },
                "fusion": fusion, "native_arms": native_evidence, "calibration": calibrations,
                "model_health": {"passed": True, "basis": "weighted_validation_q50_and_finite_ordered_quantiles"},
                "exact_replay": {"passed": True, "rows_per_arm": len(replay), "inference": replayed.attrs["inference"]},
            }
            if window_sizes:
                summary["split"]["window"] = {
                    "train_sessions": window_sizes.get("train_sessions", 315),
                    "validation_sessions": window_sizes.get("validation_sessions", 35),
                    "calibration_sessions": window_sizes.get("calibration_sessions", 35),
                }
            write_json(stage / FINAL_FIT_REPORT, summary)
            unified = {
                key: summary[key] for key in (
                    "schema_version", "model_version", "run_id", "arm", "research_report_sha256", "config_sha256",
                    "final_model_sha256", "native_file_sha256", "input_sha256", "universe", "research_status", "release_status",
                )
            }
            unified.update(
                model_version_basis=payload, final_fit_report_sha256=_sha(stage / FINAL_FIT_REPORT),
                final_fit_report=FINAL_FIT_REPORT, model=MODEL,
            )
            write_json(stage / UNIFIED_REPORT, unified)
            verify_final_fit(stage, source_root=root)
            if _sha(report_path) != report_sha or _sha(config_path) != config_sha:
                raise ValueError("最终拟合期间研究报告或配置发生变化")
            destination = root / "models/forward"
            destination.parent.mkdir(parents=True, exist_ok=True)
            model_root.rename(destination)
            published.append(destination)
            for relative in (FINAL_FIT_REPORT, UNIFIED_REPORT):
                target = root / relative
                _publish(stage / relative, target)
                published.append(target)
        verify_final_fit(root)
        return root / UNIFIED_REPORT
    except Exception:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        if owned_scheduler is not None:
            owned_scheduler.__exit__(None, None, None)
