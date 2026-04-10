"""
Fine-grained overhead measurement script for COMP597.

Measures overhead broken down into:
  - cuda_sec:        pure GPU compute (between syncs)
  - sync_sec:        torch.cuda.synchronize() time (start + stop per phase)
  - nvml_mem_sec:    nvmlDeviceGetMemoryInfo() per phase
  - nvml_util_sec:   nvmlDeviceGetUtilizationRates() per step (optimizer only)
  - psutil_sec:      cpu_percent() + virtual_memory() per step (optimizer only)
"""

import os
import csv
import time
import logging
import psutil
import pynvml
import torch
from pathlib import Path

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)
trainer_stats_name = "basic_resources_stats_overhead"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        device = torch.get_default_device()
    out_dir = conf.trainer_stats_configs.basic_resources.output_dir
    return OverheadMonitor(device=device, output_dir=out_dir)


class OverheadMonitor(base.TrainerStats):

    def __init__(self, device: torch.device, output_dir: str) -> None:
        super().__init__()

        self.device = device
        self._run_id = int(time.time())

        if torch.cuda.is_available():
            pynvml.nvmlInit()
            gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        else:
            self._gpu_handle = None

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._out = Path(output_dir)
        self._phase_csv = self._out / f"phases_{self._run_id}.csv"
        self._phase_fh  = None
        self._phase_wr  = None

        self._step_idx      = 0
        self._batch_size    = 1
        self._t_train_start = 0.0
        self._phase_t0      = 0.0
        self._sync_start_sec = 0.0

        self._cpu_last_sample_t = 0.0
        self._cpu_last_value    = 0.0

    def _sync(self) -> float:
        """Synchronize and return time taken."""
        if torch.cuda.is_available():
            t0 = time.perf_counter()
            torch.cuda.synchronize(self.device)
            return time.perf_counter() - t0
        return 0.0

    def _gpu_mem_mb(self) -> float:
        if self._gpu_handle:
            return pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle).used / 1024 ** 2
        return 0.0

    def _gpu_util_pct(self) -> float:
        if self._gpu_handle:
            return float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu)
        return 0.0

    def _cpu_util_pct(self) -> float:
        now = time.perf_counter()
        if (now - self._cpu_last_sample_t) >= 0.1:
            self._cpu_last_value    = psutil.cpu_percent(interval=None)
            self._cpu_last_sample_t = now
        return self._cpu_last_value

    def start_train(self) -> None:
        self._t_train_start = time.perf_counter()
        self._phase_fh = open(self._phase_csv, mode="w", newline="")
        print(f"[OverheadMonitor] Logging to {self._phase_csv}")

    def stop_train(self) -> None:
        if self._phase_fh:
            self._phase_fh.close()
        if torch.cuda.is_available():
            pynvml.nvmlShutdown()

        elapsed = time.perf_counter() - self._t_train_start
        total_csv = self._out / f"total_time_{self._run_id}.csv"
        with open(total_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["total_time_sec"])
            w.writeheader()
            w.writerow({"total_time_sec": elapsed})
        print(f"[OverheadMonitor] Done in {elapsed:.3f}s")

    def start_step(self, batch_size: int = None) -> None:
        if batch_size is not None:
            self._batch_size = batch_size

    def stop_step(self) -> None:
        self._step_idx += 1

    def _start_phase(self) -> None:
        """Time the start sync, then start phase timer."""
        self._sync_start_sec = self._sync()
        self._phase_t0 = time.perf_counter()

    def _record_phase(self, name: str) -> None:
        # Time stop sync
        sync_stop_sec = self._sync()
        duration = time.perf_counter() - self._phase_t0

        # Time NVML memory query
        t0 = time.perf_counter()
        gpu_mem = self._gpu_mem_mb()
        nvml_mem_sec = time.perf_counter() - t0

        # Extra queries at optimizer only
        nvml_util_sec = 0.0
        psutil_sec    = 0.0

        if name == "optimizer":
            t0 = time.perf_counter()
            self._gpu_util_pct()
            nvml_util_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            self._cpu_util_pct()
            psutil.virtual_memory()
            psutil_sec = time.perf_counter() - t0

        row = {
            "step":           self._step_idx,
            "phase":          name,
            "cuda_sec":       duration - sync_stop_sec,
            "sync_start_sec": self._sync_start_sec,
            "sync_stop_sec":  sync_stop_sec,
            "sync_total_sec": self._sync_start_sec + sync_stop_sec,
            "nvml_mem_sec":   nvml_mem_sec,
            "nvml_util_sec":  nvml_util_sec,
            "psutil_sec":     psutil_sec,
        }

        if self._phase_wr is None:
            self._phase_wr = csv.DictWriter(self._phase_fh, fieldnames=row.keys())
            self._phase_wr.writeheader()
        self._phase_wr.writerow(row)

    def start_forward(self) -> None:        self._start_phase()
    def stop_forward(self) -> None:         self._record_phase("forward")
    def start_backward(self) -> None:       self._start_phase()
    def stop_backward(self) -> None:        self._record_phase("backward")
    def start_optimizer_step(self) -> None: self._start_phase()
    def stop_optimizer_step(self) -> None:  self._record_phase("optimizer")

    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None:  pass
    def log_step(self) -> None:              pass
    def log_loss(self, loss: torch.Tensor) -> None: pass
    def log_stats(self) -> None:             pass