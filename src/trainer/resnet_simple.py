"""ResNet152 trainer with fixed-step training loop.

Wraps SimpleTrainer and overrides the training loop to run for a fixed
number of steps rather than a fixed duration. The number of steps is
determined from an uninstrumented time baseline run, so that all
experiments (with and without instrumentation) run for exactly the same
number of steps. This makes it easy to measure instrumentation overhead
by comparing wall-clock time across conditions.

Target: approximately 5 minutes of training per configuration.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm
from typing import Any, Dict, Optional, override

import src.config as config
import src.trainer.stats as stats
from src.trainer.simple import SimpleTrainer

# Fixed number of steps per batch size, determined from uninstrumented
# time baseline runs (~5 minutes each).
# BS 128: ~600 ms/step → 300s / 0.600 = 500 steps
# BS 64:  ~270 ms/step → 300s / 0.270 = 1111 steps
# BS 32:  ~121 ms/step → 300s / 0.121 = 2479 steps
_STEPS_PER_BS = {
    128: 500,
    64:  1111,
    32:  2479,
}

# Default fallback if batch size not in the table
_DEFAULT_STEPS = 500


class ResNetSimpleTrainer(SimpleTrainer):
    """ResNet152 trainer that runs for a fixed number of steps.

    Instead of stopping after a wall-clock deadline, the training loop
    runs for exactly `max_steps` steps. This ensures that experiments
    with and without instrumentation perform the same amount of work,
    making overhead measurement straightforward.
    """

    def __init__(
        self,
        loader: data.DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
        stats: stats.TrainerStats,
        conf: Optional[config.Config] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__(
            loader=loader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            stats=stats,
            conf=conf,
        )
        self._criterion = nn.CrossEntropyLoss().to(self.model.device)

        # Determine max_steps from batch size if not explicitly provided
        if max_steps is not None:
            self.max_steps = max_steps
        else:
            bs = loader.batch_size if loader.batch_size is not None else _DEFAULT_STEPS
            self.max_steps = _STEPS_PER_BS.get(bs, _DEFAULT_STEPS)
            print(f"[ResNetTrainer] batch_size={bs} → max_steps={self.max_steps}")

    @override
    def process_batch(self, i: int, batch: Any) -> Any:
        if isinstance(batch, (list, tuple)):
            return [v.to(self.device) for v in batch]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @override
    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Forward pass with cross-entropy loss."""
        self.optimizer.zero_grad()
        inputs, targets = batch
        outputs = self.model(inputs, **model_kwargs)
        return self._criterion(outputs, targets)

    @override
    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    @override
    def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        """Run training for a fixed number of steps.

        The dataloader is iterated repeatedly until `max_steps` steps
        have been completed. Each complete pass through the data is
        counted as one epoch.

        Parameters
        ----------
        model_kwargs
            Extra keyword arguments forwarded to the model's forward pass.
        """
        t_start = time.perf_counter()

        global_step = 0
        epoch = 0

        progress_bar = tqdm.auto.tqdm(
            total=self.max_steps,
            desc=f"Training ({self.max_steps} steps)",
            unit="step"
        )

        self.stats.start_train()

        while global_step < self.max_steps:
            epoch += 1
            for i, batch in enumerate(self.loader):
                if global_step >= self.max_steps:
                    break

                self.stats.start_step(
                    batch_size=len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
                )
                loss, descr = self.step(global_step, batch, model_kwargs)
                self.stats.stop_step()

                if self.enable_checkpointing and self.should_save_checkpoint(global_step):
                    self.stats.start_save_checkpoint()
                    self.save_checkpoint(global_step)
                    self.stats.stop_save_checkpoint()

                self.stats.log_loss(loss)
                self.stats.log_step()

                global_step += 1
                progress_bar.update(1)
                if descr is not None:
                    progress_bar.set_postfix_str(descr)

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()

        elapsed = time.perf_counter() - t_start
        print(f"[ResNetTrainer] Finished: {global_step} steps over {epoch} epoch(s) in {elapsed:.1f}s")
