"""ResNet152 model implementation for the training framework."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as tvmodels

import src.config as config
import src.trainer.stats as stats_mod
from src.trainer.simple import SimpleTrainer
from src.trainer.resnet_simple import ResNetSimpleTrainer
from src.trainer.base import Trainer


_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    """Convert ImageFolder format to PyTorch tensors."""
    images = torch.stack([_transform(item[0]) for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return (images, labels)


def _make_dataloader(conf: config.Config, dataset: data.Dataset) -> data.DataLoader:
    """Create a DataLoader for ResNet152 training."""

    batch_size = getattr(conf, "batch_size", 4)
    num_workers = getattr(conf.data_configs.dataset, 'load_num_proc', 1)
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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
    backbone = tvmodels.resnet152(weights="DEFAULT")
    # num_classes = len(dataset.features["label"].names)
    # backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    model = backbone.to(device)
    model.device = device
    print(f"[ResNet152] Model loaded with pretrained weights")

    # 2) Create data loader
    print(f"[ResNet152] Creating data loader (batch_size={getattr(conf, 'batch_size', 4)})...")
    loader = _make_dataloader(conf, dataset)
    print(f"[ResNet152] DataLoader created with {len(loader)} batches")

    # 3) Create optimizer
    learning_rate = getattr(conf, "learning_rate", 1e-6)
    print(f"[ResNet152] Creating SGD optimizer (lr={learning_rate})...")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 4) Create statistics tracker
    trainer_stats = getattr(conf, 'trainer_stats', 'basic_resources_stats')

    if trainer_stats in ('no-op', 'noop'):
        print("[ResNet152] Detected 'no-op', using 'basic_resources_stats' instead")
        trainer_stats = 'basic_resources_stats'
        conf.trainer_stats = 'basic_resources_stats'
    else:
        print(f"[ResNet152] Using trainer_stats: {trainer_stats}")
        conf.trainer_stats = trainer_stats

    # Carbon tracker
    # if trainer_stats == 'no-op':
    #     print("[ResNet152] Detected 'no-op', using 'carbon_tracker' instead")
    #     trainer_stats = 'carbon_tracker'
    #     conf.trainer_stats = 'carbon_tracker'
    # elif trainer_stats == 'noop':
    #     print("[ResNet152] Using 'carbon_tracker' for carbon monitoring")
    #     trainer_stats = 'carbon_tracker'
    #     conf.trainer_stats = 'carbon_tracker'
    # else:
    #     print(f"[ResNet152] Using trainer_stats: {trainer_stats}")
    #     conf.trainer_stats = trainer_stats

    #  Combined tracker
    # if trainer_stats == 'no-op':
    #     print("[ResNet152] Detected 'no-op', using 'combined_stats' instead")
    #     trainer_stats = 'combined_stats'
    #     conf.trainer_stats = 'combined_stats'
    # elif trainer_stats == 'noop':
    #     print("[ResNet152] Using 'combined_stats' for full monitoring (GPU/RAM + Carbon)")
    #     trainer_stats = 'combined_stats'
    #     conf.trainer_stats = 'combined_stats'
    # else:
    #     print(f"[ResNet152] Using trainer_stats: {trainer_stats}")
    #     conf.trainer_stats = trainer_stats

    # marlena script
    # if trainer_stats == 'no-op':
    #     print("[ResNet152] Detected 'no-op', using 'basic_resources_stats' instead")
    #     trainer_stats = 'basic_resources_stats'
    #     conf.trainer_stats = 'basic_resources_stats'
    # elif trainer_stats == 'noop':
    #     print("[ResNet152] Using 'noop' (no statistics). Consider using 'resource_monitoring' for full stats.")
    # else:
    #     print(f"[ResNet152] Using trainer_stats: {trainer_stats}")
    #     conf.trainer_stats = trainer_stats

    stats = stats_mod.init_from_conf(conf, device=device)

    # 5) Create trainer
    print("[ResNet152] Creating ResNetSimpleTrainer...")
    trainer = ResNetSimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        device=device,
        stats=stats,
        conf=conf,
    )

    print("[ResNet152] Initialization complete!")
    
    return trainer, None