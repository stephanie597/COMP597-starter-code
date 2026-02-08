import datasets
import src.config as config
import torch.utils.data
import os
from pathlib import Path


data_load_name = "resnet152_data"

def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    """
    Load dataset for ResNet152 training.
    
    This function loads data using HuggingFace's imagefolder format.
    It automatically finds FakeImageNet data from environment variables.
    """
    dconf = conf.data_configs.dataset
    
    print("[resnet152/data] Loading data for ResNet152...")
    
    # Get dataset name from config
    dataset_name = getattr(dconf, 'name', None)
    
    # If no dataset name specified, use imagefolder with FakeImageNet
    if dataset_name is None or str(dataset_name).strip() == "":
        print("[resnet152/data] No dataset.name specified, using imagefolder")
        
        # Get base directory from environment
        base = (
            os.getenv("MILABENCH_DIR_DATA")
            or os.getenv("COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR")
            or os.getenv("COMP597_JOB_STUDENT_STORAGE_DIR")
        )
        
        if not base:
            raise ValueError(
                "dataset.name is empty and no base directory found in environment.\n"
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
        
        # Get split with proper handling of empty values
        split = getattr(dconf, 'split', 'train')
        if split is None or str(split).strip() == "":
            split = 'train'
        else:
            split = str(split).strip()
        
        print(f"[resnet152/data] Loading imagefolder from {root} with split={split}")
        
        # Load using imagefolder
        return datasets.load_dataset(
            "imagefolder",
            data_dir=str(root),
            split=split,
        )
    
    # If dataset name is specified, use it (like original dataset/data.py)
    print(f"[resnet152/data] Using dataset: {dataset_name}")
    
    train_files = None
    if getattr(dconf, 'train_files', None) is not None and dconf.train_files != "":
        train_files = {"train": dconf.train_files}
        print(f"[resnet152/data] Using train_files: {train_files}")
    
    # Same fix for split
    split = getattr(dconf, 'split', 'train')
    if split is None or str(split).strip() == "":
        split = 'train'
    else:
        split = str(split).strip()
    
    num_proc = getattr(dconf, 'load_num_proc', None)
    
    print(f"[resnet152/data] Loading with split={split}, num_proc={num_proc}")
    
    return datasets.load_dataset(
        dataset_name,
        data_files=train_files,
        split=split,
        num_proc=num_proc
    )