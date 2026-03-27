"""
Minimal end-to-end training timer.

Records total wall-clock time from the first training step to the last,
with no additional instrumentation so the measurement reflects true
baseline performance.
"""

import torch
import time
import src.config as config
import src.trainer.stats.base as base

trainer_stats_name = "end_to_end_time_stats"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    return EndToEndTimeStats()


class EndToEndTimeStats(base.TrainerStats):
    """Lightweight timer that measures total training duration only.

    No per-step or per-phase instrumentation is performed, so this
    represents the lowest possible measurement overhead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._t_start: float = 0.0
        self._t_stop: float = 0.0

    def start_train(self) -> None:
        self._t_start = time.perf_counter()

    def stop_train(self) -> None:
        self._t_stop = time.perf_counter()
        elapsed = self._t_stop - self._t_start
        print(f"[TimeBaseline] Total training time (no instrumentation): {elapsed:.3f}s")

    # All other hooks are no-ops — intentionally left empty.
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
