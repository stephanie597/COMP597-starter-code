# === import necessary modules ===
import sys

# CRITICAL: Monkey patch BEFORE anything else
# This replaces src.data's load_data with our version
def _patch_data_loader():
    """Replace the default data loader before it's called"""
    try:
        import src.data
        import src.data.resnet152.data
        
        # Save original (if needed for debugging)
        if not hasattr(src.data, '_original_load_data'):
            src.data._original_load_data = src.data.load_data
        
        # Replace with our version
        src.data.load_data = src.data.resnet152.data.load_data
        
        print("[resnet152/__init__] Successfully patched src.data.load_data")
        return True
    except Exception as e:
        print(f"[resnet152/__init__] Failed to patch: {e}")
        return False

# Execute patch immediately when module is imported
_patch_data_loader()

# Now import the rest normally
from src.models.resnet152.resnet152 import resnet152_init
import src.config as config
import src.trainer as trainer

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "resnet152"

def init_model(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return resnet152_init(conf, dataset)