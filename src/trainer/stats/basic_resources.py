"""
Fine-grained resource monitor for ResNet152 training.

Collects per-step and per-phase metrics:
  - Step-level : wall time, throughput, GPU utilisation, GPU memory, CPU utilisation, RAM
  - Phase-level: forward / backward / optimizer durations and GPU memory snapshots

GPU timing uses torch.cuda.synchronize() before every timestamp so that
asynchronous CUDA kernels are fully accounted for.  CPU utilisation is
rate-limited to one sample per 100 ms to stay within NVML's reporting
resolution (~167 ms – 1 s window).

At the end of training, two figures are saved automatically:
  1. Six-panel timeline  (GPU util, CPU util, GPU mem, RAM, throughput, step time)
  2. Phase bar chart     (mean ± std for forward / backward / optimizer)
"""

import os
import csv
import time
import logging
import psutil
import pynvml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)
trainer_stats_name = "basic_resources_stats"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("[ResourceMonitor] No device kwarg — using default PyTorch device.")
        device = torch.get_default_device()
    out_dir = conf.trainer_stats_configs.basic_resources.output_dir
    return ResourceMonitor(device=device, output_dir=out_dir)


class ResourceMonitor(base.TrainerStats):
    """Per-step and per-phase resource monitor with accurate GPU timing."""

    # How many steps to group when plotting GPU util (accounts for NVML resolution)
    _GPU_AGG_WINDOW = 15

    def __init__(self, device: torch.device, output_dir: str) -> None:
        super().__init__()

        self.device = device
        self._proc = psutil.Process(os.getpid())
        self._run_id = int(time.time())

        # NVML handle for GPU utilisation
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        else:
            self._gpu_handle = None

        # Output paths
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._out = Path(output_dir)
        self._step_csv   = self._out / f"steps_{self._run_id}.csv"
        self._phase_csv  = self._out / f"phases_{self._run_id}.csv"

        self._step_fh    = None
        self._step_wr    = None
        self._phase_fh   = None
        self._phase_wr   = None

        # Counters / accumulators
        self._step_idx   = 0
        self._batch_size = 1
        self._t_train_start: float = 0.0

        # Phase timing state
        self._phase_t0: float = 0.0

        # CPU rate-limiting state
        self._cpu_last_sample_t: float = 0.0
        self._cpu_last_value: float    = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        """Synchronise CPU with GPU so timestamps are accurate."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _gpu_mem_mb(self) -> float:
        if self._gpu_handle:
            return pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle).used / 1024 ** 2
        return 0.0

    def _gpu_util_pct(self) -> float:
        if self._gpu_handle:
            return float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu)
        return 0.0

    def _cpu_util_pct(self) -> float:
        """Return CPU utilisation, but re-sample at most every 100 ms."""
        now = time.perf_counter()
        if (now - self._cpu_last_sample_t) >= 0.1:
            self._cpu_last_value     = psutil.cpu_percent(interval=None)
            self._cpu_last_sample_t  = now
        return self._cpu_last_value

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def start_train(self) -> None:
        self._t_train_start = time.perf_counter()
        self._step_fh  = open(self._step_csv,  mode="w", newline="")
        self._phase_fh = open(self._phase_csv, mode="w", newline="")
        print(f"[ResourceMonitor] Logging to {self._out}")

    def stop_train(self) -> None:
        for fh in (self._step_fh, self._phase_fh):
            if fh:
                fh.close()

        if torch.cuda.is_available():
            pynvml.nvmlShutdown()

        elapsed = time.perf_counter() - self._t_train_start

        # Write total training time to separate CSV
        total_csv = self._out / f"total_time_{self._run_id}.csv"
        with open(total_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["total_time_sec"])
            w.writeheader()
            w.writerow({"total_time_sec": elapsed})

        print(f"[ResourceMonitor] Training finished in {elapsed:.3f}s")
        print(f"[ResourceMonitor] Generating plots …")
        self._plot_timelines()
        self._plot_phase_bars()
        print(f"[ResourceMonitor] All outputs saved to {self._out}")

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def start_step(self, batch_size: int = None) -> None:
        if batch_size is not None:
            self._batch_size = batch_size
        #self._sync()
        self._t_step_start = time.perf_counter()

    def stop_step(self) -> None:
        #self._sync()
        t_end   = time.perf_counter()
        elapsed = t_end - self._t_step_start

        throughput = self._batch_size / elapsed if elapsed > 0 else 0.0

        row = {
            "step":                    self._step_idx,
            "wall_timestamp":          time.time(),
            "duration_sec":            elapsed,
            "throughput_samples_sec":  throughput,
            "gpu_util_pct":            self._gpu_util_pct(),
            "gpu_mem_mb":              self._gpu_mem_mb(),
            "cpu_util_pct":            self._cpu_util_pct(),
            "ram_mb":                  psutil.virtual_memory().used / 1024 ** 2,
        }

        if self._step_wr is None:
            self._step_wr = csv.DictWriter(self._step_fh, fieldnames=row.keys())
            self._step_wr.writeheader()
        self._step_wr.writerow(row)
        self._step_idx += 1

    # ------------------------------------------------------------------
    # Phase hooks  (forward / backward / optimizer)
    # ------------------------------------------------------------------

    def start_forward(self) -> None:
        self._sync()
        self._phase_t0 = time.perf_counter()

    def stop_forward(self) -> None:
        self._record_phase("forward")

    def start_backward(self) -> None:
        self._sync()
        self._phase_t0 = time.perf_counter()

    def stop_backward(self) -> None:
        self._record_phase("backward")

    def start_optimizer_step(self) -> None:
        self._sync()
        self._phase_t0 = time.perf_counter()

    def stop_optimizer_step(self) -> None:
        self._record_phase("optimizer")

    def _record_phase(self, name: str) -> None:
        self._sync()
        duration = time.perf_counter() - self._phase_t0

        row = {
            "step":           self._step_idx,
            "wall_timestamp": time.time(),
            "phase":          name,
            "duration_sec":   duration,
            "gpu_mem_mb":     self._gpu_mem_mb(),
        }

        if self._phase_wr is None:
            self._phase_wr = csv.DictWriter(self._phase_fh, fieldnames=row.keys())
            self._phase_wr.writeheader()
        self._phase_wr.writerow(row)

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _plot_timelines(self) -> None:
        import pandas as pd

        df = pd.read_csv(self._step_csv)
        if df.empty:
            logger.warning("[ResourceMonitor] No step data — skipping timeline plot.")
            return

        # Relative time axis, clipped to first 5 minutes
        df["t"] = df["wall_timestamp"] - df["wall_timestamp"].iloc[0]
        df = df[df["t"] <= 300].copy()

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle("ResNet152 — Resource Timelines (5 min)", fontsize=15, fontweight="bold")

        def _plot(ax, x, y, ylabel, title):
            ax.plot(x, y, linewidth=1.4)
            ax.axhline(np.mean(y), color="red", linestyle="--", linewidth=1,
                       label=f"avg {np.mean(y):.2f}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # GPU util — aggregate to reduce NVML noise
        df["_blk"] = np.arange(len(df)) // self._GPU_AGG_WINDOW
        agg = df.groupby("_blk").agg({"t": "mean", "gpu_util_pct": "mean"}).reset_index(drop=True)
        _plot(axes[0, 0], agg["t"], agg["gpu_util_pct"], "GPU Util (%)", "GPU Utilisation")

        _plot(axes[0, 1], df["t"], df["cpu_util_pct"],  "CPU Util (%)",  "CPU Utilisation")
        _plot(axes[0, 2], df["t"], df["gpu_mem_mb"],    "GPU Mem (MB)",  "GPU Memory")
        _plot(axes[1, 0], df["t"], df["ram_mb"],        "RAM (MB)",      "System RAM")
        _plot(axes[1, 1], df["t"], df["throughput_samples_sec"], "Samples/s", "Throughput")

        # Step time — drop first step (warmup outlier)
        st = df["duration_sec"].iloc[1:]
        axes[1, 2].plot(df["step"].iloc[1:], st, linewidth=1.4)
        axes[1, 2].axhline(st.mean(), color="red", linestyle="--", linewidth=1,
                           label=f"avg {st.mean():.4f}s")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Duration (s)")
        axes[1, 2].set_title("Step Time")
        axes[1, 2].set_ylim(bottom=0)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        out_path = self._out / f"timelines_{self._run_id}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"[ResourceMonitor] Timeline plot → {out_path}")

    def _plot_phase_bars(self) -> None:
        import pandas as pd

        df = pd.read_csv(self._phase_csv)
        if df.empty:
            logger.warning("[ResourceMonitor] No phase data — skipping bar plot.")
            return

        # Clip to first 5 minutes
        df["t"] = df["wall_timestamp"] - df["wall_timestamp"].iloc[0]
        df = df[df["t"] <= 300]

        phases = ["forward", "backward", "optimizer"]
        df = df[df["phase"].isin(phases)]

        grouped = df.groupby("phase")["duration_sec"]
        means = grouped.mean().reindex(phases)
        stds  = grouped.std().reindex(phases)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("ResNet152 — Mean Phase Duration", fontsize=14, fontweight="bold")

        bars = ax.bar(phases, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452", "#55A868"])
        for i, p in enumerate(phases):
            ax.text(i, means[p] + stds[p] * 0.1,
                    f"{means[p]*1000:.2f} ± {stds[p]*1000:.2f} ms",
                    ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Duration (s)")
        ax.set_xlabel("Phase")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = self._out / f"phase_bars_{self._run_id}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"[ResourceMonitor] Phase bar plot → {out_path}")

    # ------------------------------------------------------------------
    # Remaining no-op hooks
    # ------------------------------------------------------------------

    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None: pass
    def log_step(self) -> None: pass
    def log_loss(self, loss: torch.Tensor) -> None: pass
    def log_stats(self) -> None: pass
