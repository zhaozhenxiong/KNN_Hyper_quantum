"""Bound GPU training concurrency and record measured resource use."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import ctypes
from ctypes import wintypes
from itertools import product
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import perf_counter

from my_strategy.latent_quant import models


GRID = tuple(product((4, 6), (0.03, 0.07), (3.0, 10.0)))
ITERATIONS = 160
PATIENCE = 20
FIXED = (4, 0.05, 3.0)
MEMORY_LIMIT = 0.90
MEMORY_RESERVE_MIB = 768


def _gpu_memory() -> list[dict]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=6,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    return [{"device": int(index), "name": name.strip(), "total_mib": int(total),
             "used_mib": int(used), "driver": driver.strip()}
            for index, name, total, used, driver in csv.reader(output.splitlines())]


def _host_mib() -> float | None:
    if os.name != "nt":
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in (
                        "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
                    )]]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    return counters.working / 2**20 if get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb) else None


class DualGPU:
    """Serial queue on each of the two native CatBoost devices."""

    def __init__(self):
        from catboost.utils import get_gpu_device_count

        self.hardware = _gpu_memory()
        if get_gpu_device_count() < 2 or [item["device"] for item in self.hardware[:2]] != [0, 1]:
            raise RuntimeError("全池训练要求两张可核验的 CUDA 显卡")
        self.hardware = self.hardware[:2]
        self.ram_parts = {
            item["device"]: min(0.80, MEMORY_LIMIT - (item["used_mib"] + MEMORY_RESERVE_MIB) / item["total_mib"])
            for item in self.hardware
        }
        if min(self.ram_parts.values()) <= 0:
            raise RuntimeError("当前显卡余量无法在训练显存上限内运行资源试测")
        self.executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-{device}") for device in range(2)]
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for executor in self.executors:
            executor.shutdown(cancel_futures=True)

    @contextmanager
    def measure(self):
        started = perf_counter()
        stop = Event()
        report = {"gpu_peak_used_mib": {}, "host_peak_working_mib": None, "samples": 0, "sample_errors": []}

        def sample():
            try:
                for item in _gpu_memory()[:2]:
                    peaks = report["gpu_peak_used_mib"]
                    key = str(item["device"])
                    peaks[key] = max(peaks.get(key, 0), item["used_mib"])
                host = _host_mib()
                if host is not None:
                    report["host_peak_working_mib"] = max(report["host_peak_working_mib"] or 0, host)
                report["samples"] += 1
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report["sample_errors"].append(str(error))

        def monitor():
            while not stop.wait(0.5):
                sample()

        sample()
        thread = Thread(target=monitor, name="expert-gpu-telemetry", daemon=True)
        thread.start()
        try:
            yield report
        finally:
            stop.set()
            thread.join()
            sample()
            report["elapsed_seconds"] = perf_counter() - started

    def fit(self, train, valid, features, params, seed, *, iterations=ITERATIONS, patience=PATIENCE):
        from catboost import CatBoostRegressor, Pool

        train_x = models._matrix(train, features)
        train_y = train["target"].to_numpy()
        validation = None
        if valid is not None:
            weights = 1.0 / valid.groupby("date")["date"].transform("size")
            validation = (models._matrix(valid, features), valid["target"].to_numpy(), weights.to_numpy())

        def fit_one(device, alpha):
            began = perf_counter()
            estimator = CatBoostRegressor(
                loss_function=f"Quantile:alpha={alpha}", depth=params[0], learning_rate=params[1], l2_leaf_reg=params[2],
                iterations=iterations, random_seed=seed, task_type="GPU", devices=str(device),
                gpu_ram_part=self.ram_parts[device], thread_count=4, allow_writing_files=False, verbose=False,
            )
            fit_options = {}
            if validation is not None:
                x, y, weight = validation
                fit_options.update(eval_set=Pool(x, y, weight=weight, thread_count=4),
                                   use_best_model=True, early_stopping_rounds=patience)
            estimator.fit(train_x, train_y, **fit_options)
            return estimator, {"alpha": alpha, "device": device, "gpu_ram_part": self.ram_parts[device],
                               "seconds": perf_counter() - began,
                               "trees": int(estimator.tree_count_), "best_iteration": estimator.get_best_iteration()}

        tasks = []
        for alpha in models.QUANTILE_LEVELS:
            device = self._offset % len(self.executors)
            tasks.append(self.executors[device].submit(fit_one, device, alpha))
            self._offset += 1
        pairs = [job.result() for job in tasks]
        return [model for model, _ in pairs], [info for _, info in pairs]

    def pilot(self, train, features, seed):
        universe = train["stock"].drop_duplicates().sort_values().sample(frac=1, random_state=seed)
        stages = []
        for count in (128, 512, 1024):
            if len(universe) < count:
                raise ValueError(f"全池资源试测缺少 {count} 个训练期在册标的")
            sample = train.loc[train["stock"].isin(universe.iloc[:count])]
            with self.measure() as resource:
                fitted, jobs = self.fit(sample, None, features, (6, 0.03, 10.0), seed, iterations=12)
            del fitted
            stage = {"stocks": count, "train_rows": len(sample), "jobs": jobs, **resource}
            capacity = {str(item["device"]): item["total_mib"] for item in self.hardware}
            if set(resource["gpu_peak_used_mib"]) != set(capacity) or resource["samples"] < 2:
                raise RuntimeError("试测未能取得两张显卡的显存读数")
            if any(used / capacity[key] > MEMORY_LIMIT for key, used in resource["gpu_peak_used_mib"].items()):
                raise RuntimeError(f"{count} 股试测的总显存使用超过 {MEMORY_LIMIT:.0%}：{stage}")
            stages.append(stage)
        return {"hardware": self.hardware, "stages": stages, "memory_limit": MEMORY_LIMIT,
                "memory_reserve_mib": MEMORY_RESERVE_MIB, "jobs_per_device": 1}
