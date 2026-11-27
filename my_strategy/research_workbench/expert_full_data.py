"""Seal and read one reusable historical full-universe panel snapshot."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.latent_quant.domestic_runner import FrozenInputs, _sha
from my_strategy.research_workbench import expert_config


DEFAULT_FULL_FREEZE = ARTIFACT_RUNS_ROOT.parent / "expert_content" / "verified-full-20261120-v1"
SHSZ_FULL_FREEZE = ARTIFACT_RUNS_ROOT.parent / "expert_content" / "verified-full-shsz-20261120-v1"
_SCHEMA = 1
_NAMES = ("panel.parquet", "benchmark.parquet", "roster.parquet", "market_status.parquet", "coverage.json")


def _full_spec(spec: Mapping | None) -> dict:
    normalized = expert_config.normalize_expert_spec({"pool": "full"} if spec is None else spec)
    if normalized["pool"] != "full":
        raise expert_config.ExpertSpecError("只能封存全量核验股票池。")
    scope = {key: normalized[key] for key in ("pool", "start", "end")}
    if expert_config.universe_policy(normalized) != expert_config.DEFAULT_UNIVERSE_POLICY:
        scope["universe_policy"] = expert_config.universe_policy(normalized)
    return scope


def full_freeze_destination(spec: Mapping | None = None) -> Path:
    """Keep the legacy main_star path and version SHSZ independently."""
    return SHSZ_FULL_FREEZE if expert_config.universe_policy(_full_spec(spec)) == "shsz" else DEFAULT_FULL_FREEZE


def _fingerprint(path: Path) -> dict:
    return {"sha256": _sha(path), "size_bytes": path.stat().st_size}


def read_expert_full_manifest(spec: Mapping | None, destination: Path) -> dict:
    """Read scope and source hashes without loading the large input panel."""
    scope = _full_spec(spec)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("input_identity", {})
    if (manifest.get("schema_version") != _SCHEMA or manifest.get("spec") != scope
            or set(manifest.get("files", {})) != set(_NAMES) or stable_hash(identity) != manifest.get("input_version")
            or identity.get("pool") != "full" or identity.get("membership") != expert_config.FULL_MEMBERSHIP
            or expert_config.universe_policy(identity) != expert_config.universe_policy(scope)
            or not identity.get("input_cutoff") or identity["input_cutoff"] > scope["end"]):
        raise expert_config.ExpertSpecError("封存快照与请求的全量研究范围或来源不一致。")
    return manifest


def _verify_files(destination: Path, manifest: dict) -> None:
    for name in _NAMES:
        if _fingerprint(destination / name) != manifest["files"][name]:
            raise expert_config.ExpertSpecError(f"全量封存输入无法通过 SHA-256 校验：{name}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze_expert_full_inputs(spec: Mapping | None, destination: Path | None = None) -> dict:
    """Build once from pinned sources, or verify and return the existing manifest."""
    request = _full_spec(spec)
    destination = full_freeze_destination(request) if destination is None else destination
    if destination.exists():
        manifest = read_expert_full_manifest(request, destination)
        _verify_files(destination, manifest)
        return manifest
    if expert_config.universe_policy(request) != expert_config.DEFAULT_UNIVERSE_POLICY:
        expert_config.require_full_universe_evidence(request)
    data = expert_config.prepare_expert_inputs(request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scope = expert_config._scope(request)
    benchmark, _ = expert_config._archive_benchmark(scope)
    benchmark = benchmark.reindex(data.sessions)[["open", "close", "daily_return"]]
    pd.testing.assert_series_equal(benchmark["daily_return"], data.benchmark_returns)
    with TemporaryDirectory(prefix=".full-inputs-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        for name, frame in (
            ("panel.parquet", data.frame), ("benchmark.parquet", benchmark.reset_index()),
            ("roster.parquet", data.roster.reset_index()), ("market_status.parquet", data.market_status),
        ):
            frame.to_parquet(stage / name, index=False, compression="zstd")
        _write_json(stage / "coverage.json", data.coverage)
        manifest = {
            "schema_version": _SCHEMA, "spec": request, "input_identity": data.coverage["input_identity"],
            "input_version": data.coverage["input_version"], "rows": len(data.frame), "sessions": len(data.sessions),
            "files": {name: _fingerprint(stage / name) for name in _NAMES},
        }
        _write_json(stage / "manifest.json", manifest)
        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            existing = read_expert_full_manifest(request, destination)
            _verify_files(destination, existing)
            if existing["input_version"] != manifest["input_version"]:
                raise expert_config.ExpertSpecError("并行封存使用了不同输入。")
            return existing
    return manifest


def read_expert_full_freeze(spec: Mapping | None, destination: Path | None = None) -> FrozenInputs:
    """Restore the exact staged files after independent byte-level verification."""
    destination = full_freeze_destination(spec) if destination is None else destination
    manifest = read_expert_full_manifest(spec, destination)
    _verify_files(destination, manifest)
    frame = pd.read_parquet(destination / "panel.parquet")
    benchmark = pd.read_parquet(destination / "benchmark.parquet").set_index("date")
    roster = pd.read_parquet(destination / "roster.parquet").set_index("stock")
    status = pd.read_parquet(destination / "market_status.parquet")
    coverage = json.loads((destination / "coverage.json").read_text(encoding="utf-8"))
    sessions = pd.DatetimeIndex(benchmark.index)
    if (len(frame) != manifest["rows"] or len(sessions) != manifest["sessions"]
            or sessions.empty or not sessions.is_monotonic_increasing or not sessions.is_unique
            or expert_config._day(sessions[-1]) != manifest["input_identity"]["input_cutoff"]
            or coverage.get("input_identity") != manifest["input_identity"] or coverage.get("input_version") != manifest["input_version"]
            or expert_config.universe_policy(coverage) != expert_config.universe_policy(manifest["spec"])):
        raise expert_config.ExpertSpecError("全量封存输入的日期、行数或来源不一致。")
    return FrozenInputs(frame, sessions, benchmark["daily_return"], status, roster, coverage, benchmark["close"])
