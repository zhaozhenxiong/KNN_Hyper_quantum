#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small parallel execution helpers for KHQuant compute workloads."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

ExecutorKind = Literal["serial", "thread", "process"]
ExecutorOption = Literal["auto", "serial", "thread", "process"]


@dataclass(frozen=True)
class ParallelTaskResult:
    index: int
    key: str
    ok: bool
    value: Any = None
    error: str = ""
    elapsed_seconds: float = 0.0


def resolve_worker_count(workers: str | int, total_tasks: int, *, cap: int = 8) -> int:
    """Resolve auto/N worker configuration with conservative defaults."""
    if total_tasks <= 1:
        return 1
    if isinstance(workers, int):
        requested = workers
    else:
        text = str(workers or "1").strip().lower()
        if text == "auto":
            cpu_count = os.cpu_count() or 2
            requested = max(1, min(cpu_count - 1, cap))
        else:
            requested = int(text)
    return max(1, min(int(requested), int(total_tasks), int(cap)))


def resolve_executor_kind(executor: ExecutorOption | str, total_tasks: int) -> ExecutorKind:
    """Choose the fastest safe executor for the current runtime when set to auto."""
    selected = str(executor or "auto").strip().lower()
    if selected in {"serial", "thread", "process"}:
        return selected  # type: ignore[return-value]
    if total_tasks <= 1:
        return "serial"
    if Path("/.dockerenv").exists():
        return "thread"
    if os.name == "nt":
        return "process"
    return "thread"


def run_parallel_tasks(
    tasks: Iterable[tuple[str, Any]],
    worker: Callable[[Any], Any],
    *,
    workers: str | int = 1,
    executor: ExecutorOption | str = "auto",
    worker_cap: int = 8,
) -> list[ParallelTaskResult]:
    """Run keyed tasks and return results in input order."""
    task_list = list(tasks)
    worker_count = resolve_worker_count(workers, len(task_list), cap=worker_cap)
    executor_kind = resolve_executor_kind(executor, len(task_list))
    if worker_count <= 1 or executor_kind == "serial":
        return [_run_one(index, key, payload, worker) for index, (key, payload) in enumerate(task_list)]

    pool_cls = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    results: list[ParallelTaskResult] = []
    with pool_cls(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(_run_one, index, key, payload, worker): index
            for index, (key, payload) in enumerate(task_list)
        }
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: item.index)
    return results


def _run_one(index: int, key: str, payload: Any, worker: Callable[[Any], Any]) -> ParallelTaskResult:
    started = time.perf_counter()
    try:
        value = worker(payload)
        return ParallelTaskResult(
            index=index,
            key=key,
            ok=True,
            value=value,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
    except Exception as exc:  # pragma: no cover - exercised by integration paths
        return ParallelTaskResult(
            index=index,
            key=key,
            ok=False,
            error=str(exc),
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
