"""Validated ownership contract and isolated paths for storage migration.

The module is deliberately side-effect free: loading or validating a contract
does not create directories, open SQLite databases, or alter runtime storage
mode.  The migration preparer is the only component allowed to create its
empty staging layout, and it requires an explicit ``--execute`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from my_strategy.core.paths import PROJECT_ROOT


DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "my_strategy" / "configs" / "storage_contract.yaml"
REQUIRED_LOCATIONS = frozenset({"raw_market", "processed_warehouse", "screen_cache", "artifact_runs", "cold_archive"})
REQUIRED_STAGE_TARGETS = frozenset({"warehouse_stage", "screen_cache_stage", "detail_archive_stage", "artifact_stage"})


@dataclass(frozen=True)
class StorageLocation:
    """A single owned storage surface resolved under the project root."""

    name: str
    path: Path
    role: str
    storage_format: str
    retention: str


@dataclass(frozen=True)
class StorageContract:
    """Machine-readable storage ownership and parallel migration contract."""

    version: int
    name: str
    project_root: Path
    config_path: Path
    runtime_default_mode: str
    parallel_migration_enabled: bool
    locations: dict[str, StorageLocation]
    stage_root: Path
    workspace_root: Path
    stage_targets: dict[str, Path]
    table_contracts: dict[str, dict[str, str]]
    cutover_guards: tuple[str, ...]

    def location(self, name: str) -> StorageLocation:
        try:
            return self.locations[name]
        except KeyError as error:
            raise KeyError(f"Unknown storage location: {name}") from error

    @property
    def raw_db(self) -> Path:
        return self.location("raw_market").path

    @property
    def processed_db(self) -> Path:
        return self.location("processed_warehouse").path

    @property
    def screen_db(self) -> Path:
        return self.location("screen_cache").path

    def stage_path(self, target: str) -> Path:
        try:
            return self.stage_targets[target]
        except KeyError as error:
            raise KeyError(f"Unknown migration target: {target}") from error


def _resolve_under_root(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    root = project_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Storage contract path must stay under project root: {resolved}")
    return resolved


def load_storage_contract(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> StorageContract:
    """Load and validate the YAML contract without touching any data surface."""
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT.resolve()
    path = Path(config_path).resolve() if config_path else root / DEFAULT_CONTRACT_PATH.relative_to(PROJECT_ROOT)
    if not path.exists():
        raise FileNotFoundError(f"Storage contract not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    locations_payload = payload.get("locations") or {}
    missing = REQUIRED_LOCATIONS - set(locations_payload)
    if missing:
        raise ValueError(f"Storage contract missing locations: {sorted(missing)}")
    locations: dict[str, StorageLocation] = {}
    for name, item in locations_payload.items():
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"Storage location {name} requires a path")
        locations[str(name)] = StorageLocation(
            name=str(name),
            path=_resolve_under_root(str(item["path"]), root),
            role=str(item.get("role") or ""),
            storage_format=str(item.get("format") or ""),
            retention=str(item.get("retention") or ""),
        )
    migration = payload.get("parallel_migration") or {}
    targets_payload = migration.get("targets") or {}
    missing_targets = REQUIRED_STAGE_TARGETS - set(targets_payload)
    if missing_targets:
        raise ValueError(f"Storage contract missing migration targets: {sorted(missing_targets)}")
    stage_root = _resolve_under_root(str(migration.get("stage_root") or ""), root)
    stage_targets = {str(name): (stage_root / str(value)).resolve() for name, value in targets_payload.items()}
    for name, target in stage_targets.items():
        if stage_root != target and stage_root not in target.parents:
            raise ValueError(f"Migration target {name} escapes stage root: {target}")
    runtime = payload.get("runtime") or {}
    table_contracts = {
        str(name): {str(key): str(value) for key, value in dict(item).items()}
        for name, item in (payload.get("table_contracts") or {}).items()
    }
    return StorageContract(
        version=int(payload.get("version") or 0),
        name=str(payload.get("name") or ""),
        project_root=root,
        config_path=path,
        runtime_default_mode=str(runtime.get("default_mode") or "db"),
        parallel_migration_enabled=bool(runtime.get("parallel_migration_enabled", False)),
        locations=locations,
        stage_root=stage_root,
        workspace_root=_resolve_under_root(str(migration.get("workspace_root") or ""), root),
        stage_targets=stage_targets,
        table_contracts=table_contracts,
        cutover_guards=tuple(str(item) for item in (payload.get("cutover_guards") or [])),
    )


def validate_storage_contract(contract: StorageContract, *, require_existing_sources: bool = False) -> dict[str, Any]:
    """Return a serializable validation report and never create/modify paths."""
    source_names = ("raw_market", "processed_warehouse", "screen_cache")
    source_paths = {name: contract.location(name).path for name in source_names}
    missing_sources = [name for name, path in source_paths.items() if not path.exists()]
    errors: list[str] = []
    if contract.version < 2:
        errors.append("contract_version_must_be_at_least_2")
    if contract.runtime_default_mode not in {"db", "dual", "file"}:
        errors.append("runtime_default_mode_must_be_db_dual_or_file")
    if require_existing_sources and missing_sources:
        errors.append(f"missing_source_paths:{','.join(missing_sources)}")
    return {
        "valid": not errors,
        "errors": errors,
        "contract_path": str(contract.config_path),
        "version": contract.version,
        "runtime_default_mode": contract.runtime_default_mode,
        "parallel_migration_enabled": contract.parallel_migration_enabled,
        "source_paths": {name: {"path": str(path), "exists": path.exists()} for name, path in source_paths.items()},
        "stage_root": str(contract.stage_root),
        "stage_root_exists": contract.stage_root.exists(),
        "stage_targets": {name: str(path) for name, path in contract.stage_targets.items()},
        "cutover_guards": list(contract.cutover_guards),
    }
