import src.config as config
import torch.utils.data
import os
from pathlib import Path
from torchvision.datasets import ImageFolder


data_load_name = "resnet152_data"

def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    """
    Load dataset for ResNet152 training.
    
    This function loads data using torchvision's ImageFolder format.
    It automatically finds FakeImageNet data from environment variables.
    """
    dconf = conf.data_configs.dataset
    
    print("[resnet152/data] Loading data for ResNet152...")
    
    # Get base directory from environment
    base = (
        os.getenv("MILABENCH_DIR_DATA")
        or os.getenv("COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR")
        or os.getenv("COMP597_JOB_STUDENT_STORAGE_DIR")
    )
    
    if not base:
        raise ValueError(
            "No base directory found in environment.\n"
            "Please set one of: MILABENCH_DIR_DATA, COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR"
        )
    
    # Construct data directory path
    data_dir = getattr(dconf, 'data_dir', None)
    if not data_dir or str(data_dir).strip() == "":
        data_dir = str(Path(base) / "FakeImageNet")
    
    root = Path(data_dir).expanduser().resolve()
    print(f"[resnet152/data] Using data directory: {root}")
    
    # Handle nested FakeImageNet directory
    if not (root / "train").exists() and (root / "FakeImageNet").exists():
        root = root / "FakeImageNet"
        print(f"[resnet152/data] Adjusted to nested path: {root}")
    
    # Check if train directory exists
    train_dir = root / "train"
    if not train_dir.exists():
        raise ValueError(
            f"Train directory not found: {train_dir}\n"
            f"Expected structure: {root}/train/<class_id>/*.JPEG\n"
            f"Please ensure FakeImageNet data exists."
        )
    
    print(f"[resnet152/data] Loading ImageFolder from {train_dir}")
    return ImageFolder(root=str(train_dir))