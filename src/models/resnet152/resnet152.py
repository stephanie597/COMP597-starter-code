"""ResNet152 model implementation for the training framework."""

from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as tvmodels

import src.config as config
import src.trainer.stats as stats_mod
from src.trainer.simple import SimpleTrainer
from src.trainer.base import Trainer


class ResNetWithLoss(nn.Module):
    """
    Wrap torchvision ResNet to return an object with .loss attribute.
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor, **kwargs):
        logits = self.backbone(pixel_values)
        loss = self.criterion(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def _make_dataloader(conf: config.Config, dataset: data.Dataset) -> data.DataLoader:
    """Create a DataLoader for ResNet152 training."""
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    def collate_fn(batch):
        """Convert HuggingFace dataset format to PyTorch tensors."""
        images = torch.stack([transform(item["image"]) for item in batch], dim=0)
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        return {"pixel_values": images, "labels": labels}

    batch_size = getattr(conf, "batch_size", 4)
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
    )


def resnet152_init(conf: config.Config, dataset: data.Dataset) -> Tuple[Trainer, Optional[Dict[str, Any]]]:
    """
    Initialize ResNet152 model, optimizer, and trainer.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResNet152] Using device: {device}")

    # 1) Create model
    print("[ResNet152] Loading ResNet152 backbone...")
    backbone = tvmodels.resnet152(weights=None)
    num_classes = backbone.fc.out_features
    
    model = ResNetWithLoss(backbone, num_classes=num_classes)
    model = model.to(device)
    print(f"[ResNet152] Model loaded with {num_classes} output classes")

    # 2) Create data loader
    print(f"[ResNet152] Creating data loader (batch_size={getattr(conf, 'batch_size', 4)})...")
    loader = _make_dataloader(conf, dataset)
    print(f"[ResNet152] DataLoader created with {len(loader)} batches")

    # 3) Create optimizer
    learning_rate = getattr(conf, "learning_rate", 1e-6)
    print(f"[ResNet152] Creating SGD optimizer (lr={learning_rate})...")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 4) Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # 5) Create statistics tracker
    # ✅ DEFAULT to resource_monitoring for ResNet152
    trainer_stats = getattr(conf, 'trainer_stats', 'resource_monitoring')
    
    # Handle legacy 'no-op' -> 'noop' or 'resource_monitoring'
    if trainer_stats == 'no-op':
        print("[ResNet152] Detected 'no-op', using 'resource_monitoring' instead")
        trainer_stats = 'resource_monitoring'
        conf.trainer_stats = 'resource_monitoring'
    elif trainer_stats == 'noop':
        print("[ResNet152] Using 'noop' (no statistics). Consider using 'resource_monitoring' for full stats.")
    else:
        print(f"[ResNet152] Using trainer_stats: {trainer_stats}")
        conf.trainer_stats = trainer_stats
    
    stats = stats_mod.init_from_conf(conf)

    # 6) Create trainer
    print("[ResNet152] Creating SimpleTrainer...")
    trainer = SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        stats=stats,
        conf=conf,
    )

    print("[ResNet152] Initialization complete!")
    
    return trainer, None