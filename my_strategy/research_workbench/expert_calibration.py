"""Compare calibrators against a verified expert archive without refitting models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.domestic_runner import FrozenInputs, _sha
from my_strategy.latent_quant.evaluation import score_quantiles
from my_strategy.research_workbench.runner import _write_table, write_json
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_research as research, expert_stock_inference


SOURCE = "20261123-140207-682234-20261120-expert-research-9920c41"
REPORT = "expert_calibration.json"
SHRINK = 0.5
KNN_NEIGHBORS = 128
KNN_PER_DATE = 256
KNN_CHUNK = 2048
KNN_FEATURES = ("q10", "q50", "q90", "width_80", "daily_q50_median", "daily_q50_rank")
REPLAY_TOLERANCE = 2 * research.GPU_INFERENCE_TOLERANCE
Q = list(models.QUANTILE_COLUMNS)
REGIMES = {"breadth": "breadth", "volatility": "volatility_20d", "liquidity": "amount"}


@dataclass(frozen=True)
class Archive:
    root: Path
    report: dict
    config: dict
    data: FrozenInputs
    macro: Path | None


def open_archive(source_run_id: str) -> Archive:
    """Verify completed model and input digests before opening the archived panel."""
    if source_run_id != slugify(source_run_id, limit=120):
        raise ValueError("归档运行编号无效")
    root = artifact_run_dir(source_run_id, create=False)
    key = research._json(root / "config/study.json")["prediction_key"]
    root, report, config, _ = research._reuse_source(source_run_id, key)
    store = ExpertArtifactStore()
    references = store.read_manifest(source_run_id)["files"]
    if research._reference_values(references) != config["sources"]:
        raise ValueError("模型归档的输入清单不匹配")
    data = research._data_by_reference(store, references, report["input_coverage"])
    data, sector = research._verified_sector(
        data, config["model_spec"].get("sector_view", False), config["model_spec"].get("sector_provider", "capco"),
    )
    expected_sector = report["input_coverage"].get("sector_view", {})
    if any(sector.get(key) != value for key, value in expected_sector.items()):
        raise ValueError("恢复的点时板块证据与原研究归档不一致")
    macro = store.verify(config["sources"]["overseas"]) if "overseas" in config["sources"] else None
    return Archive(root, report, config, data, macro)


def _settled(panel: pd.DataFrame, proof: dict) -> pd.DataFrame:
    rows = panel.loc[panel["date"].between(pd.Timestamp(proof["start"]), pd.Timestamp(proof["end"]))
                     & panel["label_end"].le(pd.Timestamp(proof["last_label_end"]))
                     & panel["label_end"].gt(panel["date"]) & np.isfinite(panel["target"])
                     & panel["signal_eligible"].eq(True)]
    actual = models._sample_evidence(rows) if not rows.empty else {}
    expected = {key: proof.get(key) for key in actual}
    if rows.empty or actual != expected:
        raise ValueError("无法从固定面板重建归档的成熟标签样本")
    return rows


def _date_equal(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Weighted residual quantiles, with total weight one per session."""
    weights = 1 / dates.groupby(dates).transform("size").to_numpy()
    bounds = []
    for index, alpha in enumerate(models.QUANTILE_LEVELS):
        order = np.argsort(values[:, index], kind="stable")
        total = np.cumsum(weights[order])
        selected = np.searchsorted(total, alpha * total[-1], side="left")
        bounds.append(values[order[selected], index])
    return np.asarray(bounds)


def _apply(raw: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.sort(np.sort(raw, axis=1) + offsets, axis=1)


def _knn_features(rows: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    values = pd.DataFrame(np.sort(raw, axis=1), columns=Q, index=rows.index)
    features = values.copy()
    features["width_80"] = values["q90"] - values["q10"]
    features["daily_q50_median"] = values["q50"].groupby(rows["date"]).transform("median")
    features["daily_q50_rank"] = values["q50"].groupby(rows["date"]).rank(pct=True)
    result = features.loc[:, KNN_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("KNN 校准输入含非有限分位数或横截面特征")
    return result


def _knn_calibration_sample(rows: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Select an equal-count q50-stratified sample per date without reading labels."""
    selected = []
    q50 = pd.Series(raw[:, 1], index=rows.index)
    positions_by_date = pd.Series(np.arange(len(rows)), index=rows.index).groupby(rows["date"], sort=True)
    for _, positions in positions_by_date:
        positions = positions.to_numpy()
        ordered = positions[np.argsort(q50.iloc[positions].to_numpy(), kind="stable")]
        if len(ordered) > KNN_PER_DATE:
            chosen = np.linspace(0, len(ordered) - 1, KNN_PER_DATE).round().astype(int)
            ordered = ordered[chosen]
        selected.extend(ordered.tolist())
    return np.asarray(selected, dtype=int)


def _knn_residual_quantiles(
    cal_rows: pd.DataFrame, cal_raw: np.ndarray, rows: pd.DataFrame, raw: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit conditional residual quantiles only on labels matured before evaluation."""
    from sklearn.neighbors import NearestNeighbors

    if cal_rows.empty or rows.empty or cal_rows["target"].isna().any():
        raise ValueError("KNN 校准需要非空且标签成熟的校准段")
    if not cal_rows["label_end"].max() < rows["date"].min():
        raise ValueError("KNN 校准标签越过评估起点")
    cal_raw, raw = np.sort(cal_raw, axis=1), np.sort(raw, axis=1)
    sample = _knn_calibration_sample(cal_rows, cal_raw)
    sampled_rows = cal_rows.iloc[sample]
    center = _knn_features(sampled_rows, cal_raw[sample])
    location, scale = center.mean(axis=0), center.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    residuals = sampled_rows["target"].to_numpy(dtype=float)[:, None] - cal_raw[sample]
    model = NearestNeighbors(n_neighbors=min(KNN_NEIGHBORS, len(sample)), algorithm="auto")
    model.fit((center - location) / scale)
    features = (_knn_features(rows, raw) - location) / scale
    calibrated = np.empty_like(raw)
    for start in range(0, len(rows), KNN_CHUNK):
        stop = min(start + KNN_CHUNK, len(rows))
        neighbors = model.kneighbors(features[start:stop], return_distance=False)
        local = residuals[neighbors]
        for index, alpha in enumerate(models.QUANTILE_LEVELS):
            correction = np.quantile(local[:, :, index], alpha, axis=1)
            calibrated[start:stop, index] = raw[start:stop, index] + correction
    calibrated = np.sort(calibrated, axis=1)
    if not np.isfinite(calibrated).all():
        raise ValueError("KNN 校准产生非有限分位数")
    evidence = {
        "method": "knn_conditional_residual_quantiles", "features": list(KNN_FEATURES),
        "neighbors": int(model.n_neighbors), "per_date_sample_cap": KNN_PER_DATE,
        "sample_rows": len(sample), "calibration_rows": len(cal_rows),
        "sample_rule": "equal_count_q50_stratified_per_session_without_labels",
        "distance": "euclidean_standardized_on_calibration_sample",
        "label_boundary": "all_calibration_label_ends_strictly_before_first_evaluation_date",
    }
    return calibrated, evidence


def _invert(archived: np.ndarray, offsets: np.ndarray, crossing_rate: float, replay=None) -> tuple[np.ndarray, str]:
    if crossing_rate == 0:
        raw = archived - offsets
        method = "inverse_no_crossing"
    elif replay is not None:
        raw = np.sort(replay(), axis=1)
        method = "verified_native_replay"
    else:
        raise ValueError("分位数校准后曾交叉，必须使用原生推理核验")
    if (not np.isfinite(raw).all() or (np.diff(raw, axis=1) < -1e-12).any()
            or not np.allclose(_apply(raw, offsets), archived, rtol=0, atol=REPLAY_TOLERANCE)):
        raise ValueError("无法复算归档分位数")
    return raw, method


def _raw_predict(archive: Archive, fold: dict, weights: dict, train: pd.DataFrame,
                 rows: pd.DataFrame, audit: dict) -> np.ndarray:
    from catboost import CatBoostRegressor

    raw = research._rules(train, rows) if weights.keys() & {"supertrend_atr", "historical_quantiles"} else {}
    fix = {"domestic_gpu", "sector_gpu", "overseas_gpu", "domestic_fixed"} & weights.keys()
    if "overseas_gpu" in fix:
        if archive.macro is None:
            raise ValueError("缺少归档海外输入")
        rows = expert_stock_inference._with_macro(rows, archive.macro)
        fix.add("domestic_gpu")
    for expert in ("domestic_gpu", "sector_gpu", "overseas_gpu", "domestic_fixed"):
        if expert not in fix:
            continue
        info = fold["models"]["arms"][expert]
        columns = tuple(info["features"])
        native = []
        for quantile in Q:
            relative = f"models/{fold['period']}/{expert}/{quantile}.cbm"
            path = archive.root / relative
            if info["model_files"][quantile] != relative or info["model_file_sha256"][quantile] != _sha(path):
                raise ValueError("原生权重指纹与归档不匹配")
            model = CatBoostRegressor()
            model.load_model(str(path), format="cbm")
            if tuple(model.feature_names_) != columns:
                raise ValueError("原生模型与归档特征顺序不一致")
            native.append(model)
        audit.setdefault(expert, research._inference_audit())
        raw[expert] = research._expert_raw(
            native, rows, columns, audit[expert], fallback=raw.get("domestic_gpu") if expert == "overseas_gpu" else None,
        )
    return research._fuse(weights, raw)


def recover_fold(archive: Archive, fold: dict, *, arm: str = "fusion") -> tuple[pd.DataFrame, dict, dict]:
    """Recover the archived forecast and causal, fit-free calibration alternatives."""
    period = fold["period"]
    model = research._json(archive.root / "models" / period / "fusion.json")
    if arm == "fusion":
        weights = model["winner"]["weights"]
    elif arm in model["ablations"]:
        weights = model["ablations"][arm]["weights"]
    elif arm in {"historical_quantiles", "supertrend_atr", "domestic_gpu", "overseas_gpu"}:
        weights = {arm: 1.0}
    else:
        raise ValueError("该研究组缺少长期收缩所需的折内预测")
    train, cal = (_settled(archive.data.frame, fold[stage]) for stage in ("train", "calibration"))
    if not train["label_end"].max() < cal["date"].min() or not cal["label_end"].max() < pd.Timestamp(fold["evaluation_start"]):
        raise ValueError("归档校准标签与上游训练或未来样本外观测交叉")
    folder = archive.root / "backtests" / period
    out = pd.read_parquet(folder / f"{arm}_predictions.parquet")
    allowed = archive.data.frame.loc[archive.data.frame["date"].between(pd.Timestamp(fold["evaluation_start"]), pd.Timestamp(fold["evaluation_end"]))
                                    & archive.data.frame["signal_eligible"].eq(True)]
    on = ["date", "stock"]
    if out.duplicated(on).any() or not out[on].reset_index(drop=True).equals(allowed[on].reset_index(drop=True)):
        raise ValueError("预测股票日期键与冻结信号样本不匹配")
    pd.testing.assert_series_equal(
        out["target"].reset_index(drop=True),
        allowed["target"].where(allowed["label_end"].le(pd.Timestamp(fold["evaluation_end"]))).reset_index(drop=True),
    )
    original = model["calibration"][arm]
    if original["method"] != "additive_residual_quantiles" or original["rows"] != len(cal):
        raise ValueError("归档校准口径或样本数不受支持")
    offsets = np.array(original["offsets"])
    audit = {}
    cal_raw = _raw_predict(archive, fold, weights, train, cal, audit)
    residuals = cal["target"].to_numpy()[:, None] - np.sort(cal_raw, axis=1)
    pooled = np.array([np.quantile(residuals[:, index], alpha) for index, alpha in enumerate(models.QUANTILE_LEVELS)])
    if not np.allclose(pooled, offsets, rtol=0, atol=REPLAY_TOLERANCE):
        raise ValueError(f"{period} 校准偏移无法从归档原生模型复算")
    current = out[Q].to_numpy()
    raw, proof = _invert(current, offsets, original["crossing_rate_after_offsets"],
                         lambda: _raw_predict(archive, fold, weights, train, allowed, audit))
    oof = pd.read_parquet(folder / "inner_oof.parquet")
    if oof.empty or oof["label_end"].ge(pd.Timestamp(fold["validation"]["start"])).any():
        raise ValueError("历史收缩锚点必须来自调参段前已成熟的内折预测")
    inner_raw = research._fuse(weights, {name: oof[[f"{name}_{q}" for q in Q]].to_numpy() for name in weights})
    inner_offset = _date_equal(oof["target"].to_numpy()[:, None] - inner_raw, oof["date"])
    balanced = _date_equal(residuals, cal["date"])
    shrunk = (1 - SHRINK) * balanced + SHRINK * inner_offset
    knn, knn_evidence = _knn_residual_quantiles(cal, cal_raw, allowed, raw)
    variants = {"current": current, "raw": raw, "date_equal": _apply(raw, balanced),
                "shrinkage": _apply(raw, shrunk), "knn_residual": knn}
    evidence = {
        "calibration": models._sample_evidence(cal), "inner_oof": models._sample_evidence(oof), "weights": weights,
        "knn": knn_evidence,
        "offsets": {"current": offsets.tolist(), "raw": [0.0] * 3, "date_equal": balanced.tolist(),
                    "shrinkage": shrunk.tolist(), "inner_oof_anchor": inner_offset.tolist()},
        "shrinkage_to_inner_oof": SHRINK, "recovery": proof, "inference": audit,
        "calibration_offset_replay_max_error": float(np.abs(pooled - offsets).max()),
        "archive_predictions_sha256": _sha(folder / f"{arm}_predictions.parquet"),
    }
    return out, variants, evidence


def _daily(rows: pd.DataFrame, values: np.ndarray, policy: str) -> tuple[pd.DataFrame, dict]:
    columns = ["date", "stock", "target"]
    table = rows[columns].reset_index(drop=True).join(pd.DataFrame(values, columns=Q))
    table["positive_score"] = table.q50.gt(0)
    all_days = table.groupby("date").agg(scored_rows=("q50", "size"), positive_scores=("positive_score", "sum"),
                                        max_score=("q50", "max"), median_score=("q50", "median"))
    settled = table.loc[table["target"].notna()].copy()
    for quantile, alpha in zip(Q, models.QUANTILE_LEVELS, strict=True):
        error = settled.target - settled[quantile]
        settled[f"below_{quantile}"] = error.le(0)
        settled[f"pinball_{quantile}"] = np.maximum(alpha * error, (alpha - 1) * error)
    settled["coverage_80"] = settled.target.between(settled.q10, settled.q90)
    settled["width_80"] = settled.q90 - settled.q10
    settled["positive_target"] = settled.target.gt(0)
    fields = ["positive_target", "coverage_80", "width_80", *[f"below_{q}" for q in Q], *[f"pinball_{q}" for q in Q]]
    by_day = settled.groupby("date")[fields].mean()
    by_day["matured_rows"] = settled.groupby("date").size()
    ranks = settled.groupby("date")[["q50", "target"]].rank()
    ranks -= ranks.groupby(settled.date).transform("mean")
    parts = ranks.assign(xy=ranks.q50 * ranks.target, x2=ranks.q50 ** 2, y2=ranks.target ** 2)
    sums = parts.groupby(settled.date)[["xy", "x2", "y2"]].sum()
    by_day["rank_ic"] = sums.xy / (sums.x2 * sums.y2).pow(0.5).replace(0, np.nan)
    top = table.sort_values(["date", "q50", "stock"], ascending=[True, False, True]).groupby("date").head(10)
    top_gate = top.groupby("date")["target"].count().eq(10)
    by_day["top10_mean_matured_excess"] = top.groupby("date")["target"].mean().where(top_gate)
    daily = all_days.join(by_day).reset_index().assign(policy=policy)
    daily["matured_rows"] = daily["matured_rows"].fillna(0).astype(int)
    scored = score_quantiles(table)
    scored.update({field: float(value) if pd.notna(value) else None for field, value in by_day.drop(columns="matured_rows").mean().items()})
    return daily, scored


def _market_days(archive: Archive, fold: dict, session_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    frame = archive.data.frame
    fields = {name: column for name, column in REGIMES.items() if column in frame.columns}
    train = _settled(frame, fold["train"])
    available = frame.loc[frame["date"].isin(session_days)]
    days = available.groupby("date").agg(universe_rows=("stock", "size"), eligible_rows=("signal_eligible", "sum"))
    limits = {}
    if fields:
        daily = available.groupby("date")[list(fields.values())].median()
        breaks = train.groupby("date")[list(fields.values())].median().median()
        for name, column in fields.items():
            if pd.notna(breaks[column]):
                limits[name] = float(breaks[column])
                days[name] = daily[column]
                days[f"{name}_regime"] = daily[column].map(lambda v: "missing" if pd.isna(v) else "high" if v > breaks[column] else "low")
    return days, limits


def run(*, source_run_id: str, run_id: str) -> Path:
    """Write daily and fold-level evidence from archived signals and mature labels."""
    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号无效")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    status = root / "status.json"
    write_json(status, {"status": "running", "phase": "verify_archive"})
    try:
        archive = open_archive(source_run_id)
        policy = archive.config["portfolio"]
        configuration = {"source_run_id": source_run_id, "prediction_key": archive.config["prediction_key"],
                         "prediction_evidence_sha256": archive.report["prediction_evidence_sha256"],
                         "shrinkage": SHRINK, "arm": "fusion", "rebalance_days": policy["rebalance_days"],
                         "threshold": policy["min_score"], "implementation_sha256": _sha(Path(__file__)),
                         "knn": {"neighbors": KNN_NEIGHBORS, "per_date_sample_cap": KNN_PER_DATE,
                                 "features": list(KNN_FEATURES), "selection": "fixed_before_outer_evaluation"}}
        context = create_run_context(run_id=run_id, task="expert-calibration-audit", as_of_date=archive.config["end"],
                                     config=configuration, data_version=stable_hash(archive.config["sources"]),
                                     seed=archive.config["seed"])
        write_json(context.subdir("config") / "study.json", configuration)
        folds = research._archived_folds(archive.report)
        begin = pd.Timestamp(folds[0]["evaluation_start"])
        end = pd.Timestamp(folds[-1]["evaluation_end"])
        continuous = archive.data.sessions[(archive.data.sessions >= begin) & (archive.data.sessions <= end)][:-1:policy["rebalance_days"]]
        parts, summary, detail = [], [], []
        for fold in folds:
            period = fold["period"]
            write_json(status, {"status": "running", "phase": period})
            rows, variants, evidence = recover_fold(archive, fold)
            calendar = pd.DatetimeIndex(rows.date.drop_duplicates())
            session_days = archive.data.sessions[(archive.data.sessions >= pd.Timestamp(fold["evaluation_start"]))
                                                 & (archive.data.sessions <= pd.Timestamp(fold["evaluation_end"]))]
            daily_market, breaks = _market_days(archive, fold, session_days)
            stats = {}
            for name, values in variants.items():
                daily, measures = _daily(rows, values, name)
                daily = daily.set_index("date").reindex(session_days).join(daily_market)
                daily.index.name = "date"
                daily["policy"] = name
                daily["fold"] = period
                daily["continuous_rebalance"] = daily.index.isin(continuous)
                daily["fold_rebalance"] = daily.index.isin(session_days[:-1:policy["rebalance_days"]])
                daily["positive_score_day"] = daily["positive_scores"].fillna(0).gt(0)
                daily["max_score_beats_entry"] = daily["max_score"].gt(policy["min_score"])
                daily["scoring_gap"] = np.select([daily["universe_rows"].isna(), daily["scored_rows"].isna(),
                                                  daily["matured_rows"].fillna(0).eq(0)],
                                                 ["missing_stock_rows", "no_eligible_forecast", "no_mature_target"], default="")
                measures.update(signal_rows=len(rows), signal_dates=len(calendar),
                                continuous_rebalances=int(daily["continuous_rebalance"].sum()),
                                continuous_rebalances_above_entry=int((daily["continuous_rebalance"] & daily["max_score_beats_entry"]).sum()),
                                fold_rebalances_above_entry=int((daily["fold_rebalance"] & daily["max_score_beats_entry"]).sum()),
                                positive_rows=int(daily["positive_scores"].sum()), positive_dates=int(daily["positive_score_day"].sum()))
                stats[name] = measures
                parts.append(daily.reset_index())
                summary.append({"fold": period, "policy": name, **{key: value for key, value in measures.items() if not isinstance(value, dict)}})
            detail.append({"fold": period, "retrospective_review": period == "historical_replay", "evidence": evidence,
                           "training_market_thresholds": breaks, "policies": stats})
        reports = context.subdir("reports")
        _write_table(reports / "calibration_daily.parquet", pd.concat(parts, ignore_index=True))
        _write_table(reports / "calibration_folds.csv", pd.DataFrame(summary))
        destination = reports / REPORT
        write_json(destination, {"run": context.to_dict(), "selection_scope": "historical_diagnostic", "registry_mutated": False,
                                 "configuration": configuration, "folds": detail})
        write_json(status, {"status": "completed", "phase": "complete"})
        return destination
    except Exception as error:
        write_json(status, {"status": "failed", "phase": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=SOURCE)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(source_run_id=args.source_run_id, run_id=args.run_id or make_run_id("expert-calibration")))


if __name__ == "__main__":
    main()
