import src.trainer.stats.base as base
import src.config as config
import torch
import time
import logging
import pynvml
import psutil
import os

logger = logging.getLogger(__name__)

trainer_stats_name="basic_resources_stats"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to basic resource trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return BasicResourcesStats(device=device)

pynvml.nvmlInit()

class BasicResourcesStats(base.TrainerStats):
    """Stats class that tracks GPU utilization, memory consumption, and I/O."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()

        self.process = psutil.Process(os.getpid())        
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
            device.index if device.index is not None else 0
        )

    def _reset_system_counters(self):
        self.mem_before = None
        self.io_before = None
        self.gpu_mem_before = None
        self.gpu_util_before = None

    def _capture_before(self):
        self.mem_before = self.process.memory_info().rss
        self.io_before = self.process.io_counters()
        self.gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.gpu_util_before = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

    def _capture_after(self):
        mem_after = self.process.memory_info().rss
        io_after = self.process.io_counters()
        gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

        return {
            "ram_mb_abs": (mem_after) / 1e6,
            "io_read_mb_abs": (io_after.read_bytes) / 1e6,
            "io_write_mb_abs": (io_after.write_bytes) / 1e6,
            "ram_delta_mb": (mem_after - self.mem_before) / 1e6,
            "io_read_mb": (io_after.read_bytes - self.io_before.read_bytes) / 1e6,
            "io_write_mb": (io_after.write_bytes - self.io_before.write_bytes) / 1e6,
            "gpu_util_before": self.gpu_util_before.gpu,
            "gpu_util_after": gpu_util_after.gpu,
            "gpu_mem_before_mb": self.gpu_mem_before.used / 1e6,
            "gpu_mem_after_mb": gpu_mem_after.used / 1e6,
        }

    def start_train(self) -> None:
        """Start training."""
        pass

    def stop_train(self) -> None:
        """Stop training."""
        pass

    def start_step(self) -> None:
        """Start a training step."""
        self._capture_before()

    def stop_step(self) -> None:
        """Stop a training step."""
        self.last_step_system = self._capture_after()

    def start_forward(self) -> None:
        """Start the forward pass."""
        pass

    def stop_forward(self) -> None:
        """Stop the forward pass."""
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats."""
        pass

    def start_backward(self) -> None:
        """Start the backward pass."""
        pass

    def stop_backward(self) -> None:
        """Stop the backward pass"""
        pass

    def start_optimizer_step(self) -> None:
        """Start the optimizer step."""
        pass

    def stop_optimizer_step(self) -> None:
        """Stop the optimizer step."""
        pass

    def start_save_checkpoint(self) -> None:
        """Start checkpointing."""
        pass

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing."""
        pass

    def log_step(self) -> None:
        """Logs information about the previous step."""
        sys = getattr(self, "last_step_system", {})

        print(
            f"RAM abs {sys.get('ram_mb_abs', 0):.1f} MB -- "
            f"I/O R abs {sys.get('io_read_mb_abs', 0):.1f} MB W {sys.get('io_write_mb_abs', 0):.1f} MB -- "
            f"RAM Δ {sys.get('ram_delta_mb', 0):.1f} MB -- "
            f"I/O R {sys.get('io_read_mb', 0):.1f} MB W {sys.get('io_write_mb', 0):.1f} MB -- "
            f"GPU util {sys.get('gpu_util_after', 0)}% -- "
            f"GPU mem {sys.get('gpu_mem_after_mb', 0):.1f} MB"
        )

    def log_stats(self) -> None:
        """Logs information about the data accumulated so far."""
        pass
