"""Read-only inference for a stock outside a frozen multimodal research universe."""

from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.latent_quant import models

from . import expert_research, latent_api, stock_replay_data


def stock_input_revision() -> dict:
    """Mark local dependencies that can change even if displayed candles stay the same."""
    stage, hfq, raw, update = (
        stock_replay_data._STAGE, stock_replay_data._HFQ, stock_replay_data._RAW_DIR, stock_replay_data._UPDATE,
    )
    paths = (stage / "roster.csv", stage / "calendar.csv", stage / "csi300.csv",
             stage / "industries/capco-industries.csv", hfq / "verified_market_hfq.parquet",
             hfq / "sina_klc_market_reference.parquet", stage / "verified_nontrading_evidence.csv",
             update / "verified_nontrading.csv", update / "star_suspensions.csv",
             raw / "khquant_raw.db", raw / "khquant_raw.db-wal")
    revision = {}
    for path in paths:
        try:
            stamp = path.stat()
        except FileNotFoundError:
            revision[path.name] = None
        else:
            revision[path.name] = {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}
    return revision


def archive_files(folder: Path, fold: str, report: dict, config: dict, file, sha, *,
                  selected_arm: str = "fusion", macro_resolver=None) -> tuple[dict, dict]:
    """Resolve archived experts by fixed names and verify native files against their pins."""
    fusion_path = file(folder, f"models/{fold}/fusion.json")
    fusion = latent_api._read(fusion_path)
    if selected_arm == "domestic_gpu":
        weights = {"domestic_gpu": 1.0}
    elif selected_arm == "fusion":
        weights = fusion["winner"]["weights"]
        if (fusion["winner"] != report["fusion"]["winner"] or not weights
                or set(weights) - {"historical_quantiles", "supertrend_atr", "domestic_gpu", "overseas_gpu"}
                or not all(np.isfinite(weight) and weight > 0 for weight in weights.values())
                or not np.isclose(sum(weights.values()), 1)):
            raise ValueError("无法验证归档融合权重。")
    else:
        raise ValueError("未知的主策略归档。")
    calibration = fusion["calibration"][selected_arm]
    offsets = np.asarray(calibration["offsets"], dtype=float)
    if calibration["method"] != "additive_residual_quantiles" or offsets.shape != (3,) or not np.isfinite(offsets).all():
        raise ValueError("该版本缺少可核验的融合校准。")
    needed = set(weights) & {"domestic_gpu", "overseas_gpu"}
    if "overseas_gpu" in needed:
        needed.add("domestic_gpu")
    hashes = {"fusion": sha(fusion_path)}
    archives = {}
    for arm in sorted(needed):
        path = file(folder, f"models/{fold}/{arm}/model.json")
        meta = latent_api._read(path)
        evidence = report["models"]["arms"][arm]
        if meta["features"] != config["features"][arm] or any(
            meta[key] != evidence[key] for key in ("features", "model_files", "model_file_sha256")
        ):
            raise ValueError("专家模型侧车与归档研究不一致。")
        hashes[arm] = sha(path)
        paths = []
        for quantile in models.QUANTILE_COLUMNS:
            relative = f"models/{fold}/{arm}/{quantile}.cbm"
            if meta["model_files"][quantile] != relative:
                raise ValueError("归档模型路径不受支持。")
            native = file(folder, relative)
            digest = sha(native)
            if digest != meta["model_file_sha256"][quantile]:
                raise ValueError("归档专家模型哈希不匹配。")
            hashes[f"{arm}/{quantile}"] = digest
            paths.append(native)
        archives[arm] = (paths, tuple(meta["features"]))
    macro = None
    if "overseas_gpu" in needed:
        if macro_resolver is None:
            root = latent_api._run_folder(config["overseas_run_id"])
            macro = file(root, "data_snapshots/overseas.parquet")
            hashes["overseas_macro"] = sha(macro)
        else:
            files, digests = macro_resolver(folder, config, ("overseas.parquet",))
            macro, hashes["overseas_macro"] = files["overseas.parquet"], digests["overseas.parquet"]
        if hashes["overseas_macro"] != config["sources"]["overseas"]["sha256"]:
            raise ValueError("海外输入和原研究的归档不一致。")
    return {"weights": weights, "offsets": offsets, "experts": archives,
            "macro": macro, "fit": report["models"]["train"]}, hashes


def _with_macro(rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    fields = list(expert_research.MACRO_FEATURES)
    macro = pd.read_parquet(path, columns=["date", *fields, "fomc_availability_basis", "fomc_available_at", "cutoff_at"])
    approved = (
        macro["fomc_availability_basis"].eq("official_release_header")
        & pd.to_datetime(macro["fomc_available_at"], utc=True).le(pd.to_datetime(macro["cutoff_at"], utc=True))
        & np.isfinite(macro[fields]).all(axis=1)
    )
    return rows.join(macro.loc[approved].set_index("date")[fields], on="date", validate="many_to_one")


def forecast(archive: dict, training_panel: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply native models or frozen rule cohorts and the archived fusion calibration."""
    from catboost import CatBoostRegressor

    fit = archive["fit"]
    train = training_panel.loc[
        training_panel["date"].between(pd.Timestamp(fit["start"]), pd.Timestamp(fit["end"]))
        & training_panel["signal_eligible"].eq(True) & np.isfinite(training_panel["target"])
        & training_panel["label_end"].gt(training_panel["date"])
        & training_panel["label_end"].le(pd.Timestamp(fit["last_label_end"]))
    ]
    if train.empty or models._sample_evidence(train) != fit:
        raise ValueError("无法逐行重建归档训练规则使用的成熟标签样本。")
    raw = expert_research._rules(train, rows)
    if archive["macro"] is not None:
        rows = _with_macro(rows, archive["macro"])
    device = {}
    for arm, (paths, features) in archive["experts"].items():
        audit = expert_research._inference_audit()
        experts = []
        for path in paths:
            model = CatBoostRegressor()
            model.load_model(str(path), format="cbm")
            if tuple(model.feature_names_) != features:
                raise ValueError("模型原生特征顺序与归档不一致。")
            experts.append(model)
        raw[arm] = expert_research._expert_raw(
            experts, rows, features, audit, fallback=raw["domestic_gpu"] if arm == "overseas_gpu" else None,
        )
        device[arm] = audit
    fused = expert_research._fuse(archive["weights"], raw)
    calibrated = np.sort(np.sort(fused, axis=1) + archive["offsets"], axis=1)
    if not np.isfinite(calibrated).all():
        raise ValueError("归档融合模型未得到有限的分位数。")
    return models._prediction_table(calibrated, rows.index), device


def original_pool_rank(rows: pd.DataFrame, frozen: pd.DataFrame, *, top_n: int, min_score: float) -> pd.DataFrame:
    """Rank an external score against the original candidates prior to allocation limits."""
    frozen = frozen.loc[frozen["score"].gt(min_score)]
    comparison = rows[["date", "stock", "score"]].merge(
        frozen[["date", "stock", "score"]], on="date", validate="one_to_many", suffixes=("", "_pool"),
    )
    ahead = comparison["score_pool"].gt(comparison["score"]) | (
        comparison["score_pool"].eq(comparison["score"]) & comparison["stock_pool"].lt(comparison["stock"])
    )
    ranks = rows["date"].map(comparison.assign(ahead=ahead).groupby("date")["ahead"].sum()).fillna(0).astype(int).add(1)
    return rows.assign(pool_rank=ranks, execution_score=rows["score"].where(rows["score"].gt(min_score) & ranks.le(top_n), min_score))
