"""Freeze the original v2 baseline before the prospective signal window opens."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, slugify
from my_strategy.research_workbench.runner import Window, _write_table, write_json

from . import models


SOURCE_RUN = "latent-quant-20261122-cpu-v2"
INPUT_CUTOFF = "2026-11-20"
FIRST_SIGNAL = "2026-11-23"
MANIFEST = "baseline_models.json"
_GROUPS = {
    "stock_quantiles": ["stock"],
    "trend_quantiles": ["direction", "atr_band"],
    "direction_quantiles": ["direction"],
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _forward_window(sessions: pd.DatetimeIndex) -> Window:
    cutoff = pd.Timestamp(INPUT_CUTOFF)
    first = pd.Timestamp(FIRST_SIGNAL)
    if len(sessions) < 315 or sessions[-1] != cutoff or not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError("The v2 calendar must contain sorted unique history through 2026-11-20")
    return Window("prospective_holdout", sessions[-315], sessions[-70], sessions[-35], pd.DatetimeIndex([first]))


def _rule_tables(train: pd.DataFrame, model_dir: Path) -> dict:
    atr = train.loc[train["atr_pct"].gt(0) & np.isfinite(train["atr_pct"]), "atr_pct"]
    cuts = np.unique(atr.quantile([1 / 3, 2 / 3]).to_numpy()) if len(atr) else np.empty(0)
    classified = models._trend_groups(train, cuts)
    result = {}
    for name, columns in _GROUPS.items():
        sample = train if name == "stock_quantiles" else classified
        table = models._group_quantiles(sample, columns).set_axis(models.QUANTILE_COLUMNS, axis=1)
        path = model_dir / f"{name}.parquet"
        _write_table(path, table.reset_index())
        result[name] = {"file": path.name, "groups": len(table)}
    result["atr_tertile_boundaries"] = cuts.tolist()
    result["min_group_rows"] = models._MIN_GROUP
    return result


def _replay_rules(rows: pd.DataFrame, model_dir: Path, manifest: dict) -> dict[str, pd.DataFrame]:
    tables = manifest["rule_tables"]
    values = {}
    for name, keys in _GROUPS.items():
        table = pd.read_parquet(model_dir / tables[name]["file"]).set_index(keys)
        if name == "trend_quantiles":
            groups = models._trend_groups(rows, np.asarray(tables["atr_tertile_boundaries"]))
            index = pd.MultiIndex.from_frame(groups[keys])
        else:
            index = rows["st_direction"] if name == "direction_quantiles" else rows["stock"]
        values[name] = table.reindex(index).to_numpy(dtype=float)
    pooled = np.asarray(manifest["pooled_quantiles"])
    history = np.where(np.isfinite(values["stock_quantiles"]), values["stock_quantiles"], pooled)
    trend = np.where(np.isfinite(values["trend_quantiles"]), values["trend_quantiles"], values["direction_quantiles"])
    trend = np.where(np.isfinite(trend), trend, pooled)
    result = {}
    for arm, raw in (("historical_quantiles", history), ("supertrend_atr", trend)):
        offsets = manifest["arms"][arm]["calibration"]["offsets"]
        result[arm] = models._prediction_table(np.sort(np.sort(raw, axis=1) + offsets, axis=1), rows.index)
    result["supertrend_atr"]["score"] = models._trend_score(rows)
    return result


def replay_frozen_baseline(frame: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Replay all four frozen forecasts for the supplied eligible rows, without targets."""
    manifest = _read(output_dir / MANIFEST)
    model_dir = output_dir / "models"
    for filename, digest in manifest["model_file_sha256"].items():
        if _hash(model_dir / filename) != digest:
            raise ValueError(f"Frozen baseline artifact changed: {filename}")
    rows = frame.loc[frame["signal_eligible"].fillna(False).astype(bool)]
    if rows.empty:
        return {arm: models._prediction_table(np.empty((0, 3)), rows.index) for arm in models.ARMS}
    result = _replay_rules(rows, model_dir, manifest)
    for arm in models.ARMS[2:]:
        result[arm] = models.replay_catboost(rows, model_dir / manifest["arms"][arm]["model_file"])
    return result


def fit_frozen_baseline(source_run: Path, output_dir: Path) -> dict:
    """Fit the fixed v2 rule and model families from the 11/20 snapshot only."""
    import catboost

    files = {
        "config": source_run / "config" / "study.json",
        "panel": source_run / "data_snapshots" / "panel.parquet",
        "benchmark": source_run / "data_snapshots" / "benchmark.parquet",
        "roster": source_run / "data_snapshots" / "roster.parquet",
        "coverage": source_run / "data_snapshots" / "coverage.json",
    }
    if _read(source_run / "status.json")["status"] != "completed":
        raise ValueError("The source baseline run must be complete")
    config = _read(files["config"])
    code_hashes = {key.replace("\\", "/"): value for key, value in config["code_sha256"].items()}
    if (config["end"] != INPUT_CUTOFF or config["prospective"]["start_at_or_after"] != FIRST_SIGNAL
            or tuple(config["arms"]) != models.ARMS or config["catboost_version"] != catboost.__version__
            or code_hashes["latent_quant/models.py"] != _hash(Path(models.__file__))):
        raise ValueError("The cutoff, original models, configuration and CatBoost version must match v2")
    sessions = pd.DatetimeIndex(pd.read_parquet(files["benchmark"], columns=["date"])["date"])
    window = _forward_window(sessions)
    frame = pd.read_parquet(files["panel"])
    labeled = frame["target"].notna()
    if frame["date"].max() != sessions[-1] or not frame.loc[labeled, "label_end"].le(sessions[-1]).all():
        raise ValueError("The v2 snapshot contains observations or known labels outside the frozen cutoff")
    eligible = frame["signal_eligible"].fillna(False).astype(bool)
    settled = eligible & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])
    train, valid, calibration = (frame.loc[mask & settled] for mask in window.masks(frame))
    if any(rows.empty for rows in (train, valid, calibration)):
        raise ValueError("The v2 fit requires nonempty purged training, validation and calibration samples")

    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    replay_sample = frame.loc[eligible].tail(64)
    evidence_rows = pd.concat([calibration, replay_sample])
    labels = calibration["target"].to_numpy(dtype=float)
    pooled = train["target"].quantile(list(models.QUANTILE_LEVELS)).to_numpy()
    manifest = {
        "source_run": source_run.name,
        "source_files": {key: {"path": str(path), "sha256": _hash(path)} for key, path in files.items()},
        "input_cutoff": INPUT_CUTOFF, "first_signal": FIRST_SIGNAL,
        "walk_forward": {"lookback": 315, "validation_sessions": 35, "calibration_sessions": 35},
        "train": models._sample_evidence(train), "validation": models._sample_evidence(valid),
        "calibration": models._sample_evidence(calibration), "pooled_quantiles": pooled.tolist(),
        "point_in_time": _read(files["coverage"])["point_in_time"],
        "rule_tables": _rule_tables(train, model_dir), "arms": {},
        "fitter_sha256": _hash(Path(__file__)),
        "scope": "The archived pre-period v2 fit. No prospective inputs, labels or returns were read.",
    }
    expected = {}
    hist, hist_details = models._historical_quantiles(train, evidence_rows, pooled)
    trend, trend_details = models._trend_quantiles(train, evidence_rows, pooled)
    for arm, raw, details in zip(models.ARMS[:2], (hist, trend), (hist_details, trend_details), strict=True):
        predicted, details["calibration"] = models._calibrate(raw, labels)
        expected[arm] = models._prediction_table(predicted, replay_sample.index)
        manifest["arms"][arm] = details
    expected["supertrend_atr"]["score"] = models._trend_score(replay_sample)
    feature_sets = (config["base_features"], config["base_features"] + config["added_features"])
    for arm, features in zip(models.ARMS[2:], feature_sets, strict=True):
        raw, model, details = models._fit_catboost(
            train, valid, evidence_rows, features, seed=config["seed"], iterations=config["iterations"],
        )
        predicted, details["calibration"] = models._calibrate(raw, labels)
        expected[arm] = models._prediction_table(predicted, replay_sample.index)
        model_path = model_dir / f"{arm}.cbm"
        temporary = model_path.with_suffix(".cbm.tmp")
        model.save_model(str(temporary), format="cbm")
        temporary.replace(model_path)
        metadata_path = model_path.with_suffix(".json")
        details.update(model_file=model_path.name, metadata_file=metadata_path.name)
        write_json(metadata_path, details)
        manifest["arms"][arm] = details
    manifest["model_file_sha256"] = {path.name: _hash(path) for path in sorted(model_dir.iterdir())}
    write_json(output_dir / MANIFEST, manifest)
    for arm, replayed in replay_frozen_baseline(replay_sample.drop(columns=["target"]), output_dir).items():
        pd.testing.assert_frame_equal(replayed, expected[arm])
    manifest["verification"] = {"four_arms_exact_replay": True, "historical_replay_rows_per_arm": len(replay_sample)}
    manifest["sealed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    write_json(output_dir / MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.run_id != slugify(args.run_id, limit=120):
        parser.error("Run ID must be a simple directory name of at most 120 characters")
    source = artifact_run_dir(SOURCE_RUN, create=False)
    output = artifact_run_dir(args.run_id, create=False)
    manifest = fit_frozen_baseline(source, output)
    config = _read(source / "config" / "study.json")
    create_run_context(
        task="latent-quant-forward-models", as_of_date=INPUT_CUTOFF, config=config,
        data_version=manifest["source_files"]["panel"]["sha256"], seed=config["seed"], run_id=args.run_id,
    )
    write_json(output / "status.json", {"status": "completed", "verified": True, "manifest": MANIFEST})
    print(output / MANIFEST)


if __name__ == "__main__":
    main()
