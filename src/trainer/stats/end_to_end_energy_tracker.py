"""
End-to-end energy tracker using CodeCarbon.

Takes a single coarse energy measurement over the entire training run.
This minimises CodeCarbon overhead so the result is comparable to the
time-only baseline. Per-step energy can be inferred by dividing total
energy by the number of steps.
"""

import os
import csv
import time
import logging
import torch
import pandas as pd
import codecarbon
import codecarbon.core.cpu
from typing import List
from pathlib import Path
from codecarbon import OfflineEmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData

import src.config as config
import src.trainer.stats.base as base

# Disable psutil so CodeCarbon falls back to constant TDP for CPU estimates.
codecarbon.core.cpu.is_psutil_available = lambda: False

logger = logging.getLogger(__name__)
trainer_stats_name = "end_to_end_energy_stats"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("[EnergyBaseline] No device kwarg found, falling back to default.")
        device = torch.get_default_device()
    cc_conf = conf.trainer_stats_configs.codecarbon
    return EndToEndEnergyStats(
        device=device,
        run_num=cc_conf.run_num,
        project_name=cc_conf.project_name,
        output_dir=cc_conf.output_dir,
    )


class _CSVAppendOutput(BaseOutput):
    """Custom CodeCarbon output handler that appends rows to a CSV file."""

    def __init__(self, filename: str, directory: str) -> None:
        self._filepath = os.path.join(directory, filename)

    def _headers_match(self, data: EmissionsData) -> bool:
        with open(self._filepath) as f:
            reader = csv.DictReader(f)
            on_disk = list(next(reader).keys())
        return on_disk == list(data.values.keys())

    def to_csv(self, total: EmissionsData, delta: EmissionsData) -> None:
        new_row = pd.DataFrame.from_records([dict(total.values)])
        exists = os.path.isfile(self._filepath)

        if exists and not self._headers_match(total):
            logger.warning("[EnergyBaseline] CSV schema changed — overwriting.")
            exists = False

        if not exists:
            new_row.to_csv(self._filepath, index=False)
        else:
            existing = pd.read_csv(self._filepath)
            pd.concat([existing, new_row]).to_csv(self._filepath, index=False)

    def out(self, total: EmissionsData, delta: EmissionsData) -> None:
        self.to_csv(total, delta)

    def live_out(self, total: EmissionsData, delta: EmissionsData) -> None:
        pass

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str) -> None:
        base_path, ext = os.path.splitext(self._filepath)
        task_path = f"{base_path}-{experiment_name}{ext}"
        rows = pd.DataFrame.from_records([dict(d.values) for d in data])
        rows.dropna(axis=1, how="all").to_csv(task_path, index=False)


class EndToEndEnergyStats(base.TrainerStats):
    """Tracks total energy and CO2 emissions for the full training run.

    CodeCarbon is started once at the beginning of training and stopped
    at the end, producing a single aggregate measurement with negligible
    overhead compared to per-step tracking.
    """

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
    ) -> None:
        self.device = device
        self.gpu_id = device.index if device.index is not None else 0
        self._t_start: float = 0.0
        self._t_stop: float = 0.0

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_name = f"run{run_num}_energy_full_gpu{self.gpu_id}.csv"

        self._tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[_CSVAppendOutput(csv_name, output_dir)],
            allow_multiple_runs=True,
            log_level="warning",
            measure_power_secs=0.5,
            gpu_ids=[self.gpu_id],
        )
        print(f"[EnergyBaseline] Initialised. Output → {output_dir}/{csv_name}")

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self._t_start = time.perf_counter()
        self._tracker.start()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self._tracker.stop()
        self._t_stop = time.perf_counter()
        elapsed = self._t_stop - self._t_start
        print(f"[EnergyBaseline] Total training time (with energy measurement): {elapsed:.3f}s")

    # All per-step hooks are no-ops for this baseline.
    def start_step(self, batch_size: int = None) -> None: pass
    def stop_step(self) -> None: pass
    def start_forward(self) -> None: pass
    def stop_forward(self) -> None: pass
    def start_backward(self) -> None: pass
    def stop_backward(self) -> None: pass
    def start_optimizer_step(self) -> None: pass
    def stop_optimizer_step(self) -> None: pass
    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None: pass
    def log_step(self) -> None: pass
    def log_loss(self, loss: torch.Tensor) -> None: pass
    def log_stats(self) -> None: pass
