#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filesystem retention helpers for generated AI-Agent artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed"
DEFAULT_SCREEN_ROOT = PROCESSED_ROOT / "ai_agent_screen"


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def cleanup_generated_files(
    processed_root: str | Path = PROCESSED_ROOT,
    screen_root: str | Path = DEFAULT_SCREEN_ROOT,
    keep_run_logs: int = 10,
    keep_figures: int = 80,
    include_legacy_backtests: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or remove stale generated files. Deletion requires execute=True."""
    processed = resolve_path(processed_root)
    screen = resolve_path(screen_root)
    candidates: list[Path] = []

    candidates.extend(_old_run_logs(processed / "run_logs", keep_run_logs))
    candidates.extend(_partial_reports(screen / "reports"))
    candidates.extend(_old_pngs(screen / "figures", keep_figures))
    candidates.extend(_stale_smoke_roots(processed))
    if include_legacy_backtests:
        candidates.extend(_legacy_backtest_roots(processed))

    unique = _dedupe_existing(candidates)
    actions = []
    for path in unique:
        size = _path_size(path)
        kind = "directory" if path.is_dir() else "file"
        if execute:
            _remove_path(path)
        actions.append(
            {
                "path": str(path),
                "kind": kind,
                "size_bytes": size,
                "status": "deleted" if execute else "dry_run",
            }
        )

    return {
        "execute": execute,
        "processed_root": str(processed),
        "screen_root": str(screen),
        "include_legacy_backtests": include_legacy_backtests,
        "candidate_count": len(actions),
        "candidate_size_bytes": sum(int(item["size_bytes"]) for item in actions),
        "actions": actions,
    }


def _old_run_logs(log_dir: Path, keep_run_logs: int) -> list[Path]:
    if not log_dir.exists():
        return []
    logs = sorted(
        [p for p in log_dir.glob("*.log") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[max(0, keep_run_logs) :]


def _partial_reports(report_dir: Path) -> list[Path]:
    if not report_dir.exists():
        return []
    patterns = ["*_partial.csv", "*_partial.json", "*.tmp"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in report_dir.glob(pattern) if path.is_file())
    return paths


def _old_pngs(directory: Path, keep: int) -> list[Path]:
    if not directory.exists():
        return []
    pngs = sorted([p for p in directory.glob("*.png") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return pngs[max(0, keep) :]


def _stale_smoke_roots(processed: Path) -> list[Path]:
    if not processed.exists():
        return []
    candidates = [p for p in processed.glob("ai_agent_rl_smoke*") if p.is_dir()]
    screen_root = processed / "ai_agent_screen"
    if screen_root.exists():
        candidates.extend(p for p in screen_root.glob("warehouse_smoke*") if p.is_dir())
    return candidates


def _legacy_backtest_roots(processed: Path) -> list[Path]:
    if not processed.exists():
        return []
    legacy_names = [
        "ai_agent_backtest",
        "ai_agent_screen_auto",
        "ai_agent_screen_100b",
        "strategy_screen",
    ]
    return [processed / name for name in legacy_names if (processed / name).exists()]


def _dedupe_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
