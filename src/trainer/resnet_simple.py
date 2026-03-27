"""ResNet152 trainer with 5-minute timed training loop.

Wraps SimpleTrainer and overrides the training loop to run for a fixed
duration (default 5 minutes) rather than a fixed number of epochs.
This matches the experimental protocol required by the course project.
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

# Target training duration in seconds (5 minutes as required)
_TRAIN_DURATION_SEC = 300


class ResNetSimpleTrainer(SimpleTrainer):
    """ResNet152 trainer that runs for a fixed wall-clock duration.

    Instead of stopping after one pass through the dataset, the training
    loop repeats over the dataloader until the target duration is reached.
    This ensures all experiments produce comparable 5-minute measurements
    regardless of batch size or dataset size.
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
        train_duration_sec: int = _TRAIN_DURATION_SEC,
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
        self.train_duration_sec = train_duration_sec
        self._criterion = nn.CrossEntropyLoss().to(self.model.device)

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
        """Run training for a fixed duration of 5 minutes.

        The dataloader is iterated repeatedly until the wall-clock time
        exceeds `train_duration_sec`. Each complete pass through the data
        is counted as one epoch.

        Parameters
        ----------
        model_kwargs
            Extra keyword arguments forwarded to the model's forward pass.
        """
        t_start = time.perf_counter()
        deadline = t_start + self.train_duration_sec

        global_step = 0
        epoch = 0

        progress_bar = tqdm.auto.tqdm(desc="Training (5 min)", unit="step")

        self.stats.start_train()

        while time.perf_counter() < deadline:
            epoch += 1
            for i, batch in enumerate(self.loader):
                # Stop cleanly at the 5-minute mark
                if time.perf_counter() >= deadline:
                    break

                self.stats.start_step(batch_size=len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch))
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
