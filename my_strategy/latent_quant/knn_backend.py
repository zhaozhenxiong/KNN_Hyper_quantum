"""Deterministic CPU/CUDA nearest-neighbor quantiles for the online KNN expert."""

from __future__ import annotations

import logging
import os
from time import perf_counter

import numpy as np


LOG = logging.getLogger(__name__)

QUANTILE_LEVELS = (0.10, 0.50, 0.90)
DEFAULT_QUERY_CHUNK = 256


def _load_torch():
    import torch

    return torch


class KNNBackend:
    """Select one device for an entire fold and record measured execution evidence."""

    def __init__(self, requested: str = "cpu", *, query_chunk: int = DEFAULT_QUERY_CHUNK):
        requested = str(requested or "cpu").strip().lower()
        if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError("KNN 设备仅支持 auto、cpu 或 cuda")
        if not isinstance(query_chunk, int) or query_chunk < 1:
            raise ValueError("KNN 查询分块必须为正整数")
        self.requested = requested
        self.query_chunk = query_chunk
        self.fallback_reason = None
        self.torch = None
        self.device = "cpu"
        if requested != "cpu":
            try:
                torch = _load_torch()
                if not torch.cuda.is_available():
                    raise RuntimeError("PyTorch 未检测到可用 CUDA 设备")
                if requested.startswith("cuda:"):
                    device = requested
                    index = int(device.split(":", 1)[1])
                    if index < 0 or index >= torch.cuda.device_count():
                        raise RuntimeError(f"KNN CUDA 设备不存在：{device}")
                else:
                    configured = os.environ.get("KHQUANT_KNN_CUDA_DEVICE")
                    index = int(configured) if configured is not None else max(0, torch.cuda.device_count() - 1)
                    if index >= torch.cuda.device_count():
                        raise RuntimeError(f"KHQUANT_KNN_CUDA_DEVICE 超出可用设备：{index}")
                    device = f"cuda:{index}"
                torch.empty(1, dtype=torch.float64, device=device)
                self.torch, self.device = torch, device
            except Exception as error:
                if requested != "auto":
                    raise RuntimeError(f"KNN 显式请求 CUDA 但初始化失败：{error}") from error
                self.fallback_reason = str(error)
                self.torch, self.device = None, "cpu"
                LOG.warning("KNN auto 无法初始化 CUDA，整折明确回退 CPU：%s", error)
        self.evidence = {
            "requested_device": requested,
            "actual_device": self.device,
            "backend": "torch_cuda_float64" if self.torch is not None else "numpy_cpu_float64",
            "fallback_reason": self.fallback_reason,
            "query_chunk": query_chunk,
            "query_rows": 0,
            "batches": 0,
            "elapsed_seconds": 0.0,
            "peak_memory_bytes": None,
            "torch_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "tie_rule": "distance_then_original_sample_position",
        }
        if self.torch is not None:
            index = self.torch.cuda.current_device() if self.device == "cuda" else int(self.device.split(":", 1)[1])
            self.evidence.update(
                torch_version=self.torch.__version__, cuda_version=self.torch.version.cuda,
                gpu_name=self.torch.cuda.get_device_name(index), peak_memory_bytes=0,
            )

    def quantiles(
        self,
        sample_features: np.ndarray,
        sample_targets: np.ndarray,
        queries: np.ndarray,
        *,
        neighbors: int = 12,
    ) -> np.ndarray:
        samples = np.asarray(sample_features, dtype=np.float64)
        targets = np.asarray(sample_targets, dtype=np.float64)
        requested = np.asarray(queries, dtype=np.float64)
        if samples.ndim != 2 or requested.ndim != 2 or samples.shape[1] != requested.shape[1]:
            raise ValueError("KNN 样本和查询必须是特征数一致的二维矩阵")
        if targets.shape != (len(samples),) or len(samples) < neighbors or not len(requested):
            raise ValueError("KNN 邻居标签、样本数或查询数无效")
        if not np.isfinite(samples).all() or not np.isfinite(targets).all() or not np.isfinite(requested).all():
            raise ValueError("KNN 后端只接受有限样本、标签和查询")
        started = perf_counter()
        if self.torch is None:
            result = self._cpu(samples, targets, requested, neighbors)
        else:
            result = self._cuda(samples, targets, requested, neighbors)
        self.evidence["elapsed_seconds"] += perf_counter() - started
        self.evidence["query_rows"] += len(requested)
        self.evidence["batches"] += (len(requested) + self.query_chunk - 1) // self.query_chunk
        return result

    def _cpu(self, samples, targets, queries, neighbors):
        output = []
        for start in range(0, len(queries), self.query_chunk):
            query = queries[start:start + self.query_chunk]
            distances = np.sum((query[:, None, :] - samples[None, :, :]) ** 2, axis=2)
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
            output.append(np.quantile(targets[nearest], QUANTILE_LEVELS, axis=1).T)
        return np.vstack(output)

    def _cuda(self, samples, targets, queries, neighbors):
        torch = self.torch
        device_index = torch.cuda.current_device() if self.device == "cuda" else int(self.device.split(":", 1)[1])
        with torch.cuda.device(device_index):
            torch.cuda.reset_peak_memory_stats(device_index)
            sample = torch.as_tensor(samples, dtype=torch.float64, device=self.device)
            target = torch.as_tensor(targets, dtype=torch.float64, device=self.device)
            levels = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float64, device=self.device)
            output = []
            for start in range(0, len(queries), self.query_chunk):
                query = torch.as_tensor(
                    queries[start:start + self.query_chunk], dtype=torch.float64, device=self.device,
                )
                distances = ((query[:, None, :] - sample[None, :, :]) ** 2).sum(dim=2)
                nearest = torch.argsort(distances, dim=1, stable=True)[:, :neighbors]
                output.append(torch.quantile(target[nearest], levels, dim=1).T)
            result = torch.cat(output).cpu().numpy()
            self.evidence["peak_memory_bytes"] = max(
                self.evidence["peak_memory_bytes"] or 0,
                int(torch.cuda.max_memory_allocated(device_index)),
            )
            return result
