"""
Hardware, latency, memory, and throughput profiling utilities for emotion NLP models.
"""

from __future__ import annotations

import gc
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import psutil
import torch


def get_system_info() -> dict[str, Any]:
    """
    Collect host hardware, OS, and environment metadata for reproducible benchmarking.
    """
    mem = psutil.virtual_memory()
    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "x86_64",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters of a PyTorch module.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }


def get_checkpoint_size_mb(path: str | Path) -> float:
    """
    Calculate the total size on disk of a model checkpoint directory or single file in MB.
    """
    target = Path(path)
    if not target.exists():
        return 0.0

    if target.is_file():
        return round(target.stat().st_size / (1024 * 1024), 2)

    total_bytes = sum(f.stat().st_size for f in target.glob("**/*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def measure_peak_ram_mb(fn: Callable[[], Any]) -> tuple[Any, float]:
    """
    Measure the peak Resident Set Size (RSS) memory delta during function execution in MB.
    """
    gc.collect()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss

    result = fn()

    gc.collect()
    end_mem = process.memory_info().rss
    delta_mb = max(0.0, (end_mem - start_mem) / (1024 * 1024))
    return result, round(delta_mb, 2)


def benchmark_latency_ms(
    fn: Callable[[], Any],
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> dict[str, float]:
    """
    Measure execution latency in milliseconds over repeated trials, reporting median and variability.
    """
    for _ in range(warmup_runs):
        fn()

    durations_ns: list[int] = []
    for _ in range(timed_runs):
        start = time.perf_counter_ns()
        fn()
        durations_ns.append(time.perf_counter_ns() - start)

    durations_ms = np.array(durations_ns, dtype=np.float64) / 1e6

    return {
        "median_ms": float(np.median(durations_ms)),
        "mean_ms": float(np.mean(durations_ms)),
        "std_ms": float(np.std(durations_ms)),
        "min_ms": float(np.min(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
        "p90_ms": float(np.percentile(durations_ms, 90)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "runs": timed_runs,
    }


def benchmark_throughput(
    predict_batch_fn: Callable[[list[str]], Any],
    batch_texts: list[str],
    min_seconds: float = 3.0,
) -> float:
    """
    Measure inference throughput (utterances per second).
    """
    predict_batch_fn(batch_texts)  # Warmup

    total_samples = 0
    start_time = time.perf_counter()
    while True:
        predict_batch_fn(batch_texts)
        total_samples += len(batch_texts)
        elapsed = time.perf_counter() - start_time
        if elapsed >= min_seconds:
            break

    return round(total_samples / elapsed, 2)
