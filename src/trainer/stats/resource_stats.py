"""Resource monitoring stats for ResNet152 training.

This module records GPU utilization, memory consumption, and I/O statistics.
"""

import time
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("[ResourceStats] Warning: pynvml not available, GPU stats will be limited")

from src.trainer.stats.base import TrainerStats


class ResourceMonitoringStats(TrainerStats):
    """
    Collect and record resource utilization statistics during training.
    
    Records:
    - GPU utilization (%)
    - GPU memory usage (MB)
    - System memory usage (MB)
    - Training throughput (samples/sec)
    - I/O time
    """
    
    def __init__(self, output_dir: str = None):
        super().__init__()
        
        # Smart default: use scratch space if available
        if output_dir is None:
            scratch = (
                os.getenv("MILABENCH_DIR_DATA")
                or os.getenv("COMP597_JOB_STUDENT_SCRATCH_STORAGE_DIR")
                or os.getenv("COMP597_JOB_STUDENT_STORAGE_DIR")
            )
            if scratch:
                output_dir = str(Path(scratch) / "training_stats")
            else:
                output_dir = "./training_stats"
        
        self.output_dir = Path(output_dir)
        
        # Try to create directory with error handling
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = self.output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            print(f"[ResourceStats] Initialized. Output directory: {self.output_dir}")
        except Exception as e:
            print(f"[ResourceStats] Warning: Cannot create directory {self.output_dir}: {e}")
            print(f"[ResourceStats] Falling back to local directory")
            # Fallback to local directory
            self.output_dir = Path("./training_stats")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"[ResourceStats] Using fallback directory: {self.output_dir}")
            except Exception as e2:
                print(f"[ResourceStats] Error: Cannot create fallback directory: {e2}")
                print(f"[ResourceStats] Stats will not be saved to files!")
                self.output_dir = None
        
        # Statistics storage
        self.stats_history = []
        self.current_stats = {}
        
        # Timing
        self.step_start_time = None
        self.epoch_start_time = None
        self.training_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None
        self.optimizer_start_time = None
        
        # Counters
        self.step_count = 0
        self.epoch_count = 0
        self.total_samples = 0
        
        # GPU handle
        self.gpu_handle = None
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                print("[ResourceStats] Could not get GPU handle")
        
        print(f"[ResourceStats] Initialized. Output directory: {self.output_dir}")
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get current GPU statistics."""
        stats = {
            "gpu_utilization": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
        }
        
        if torch.cuda.is_available():
            # PyTorch GPU memory
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            stats["gpu_memory_percent"] = (stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100
            
            # NVML stats if available
            if self.gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    stats["gpu_utilization"] = util.gpu
                except:
                    pass
        
        return stats
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get system memory statistics."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "system_memory_used_mb": mem.used / 1024**2,
                "system_memory_total_mb": mem.total / 1024**2,
                "system_memory_percent": mem.percent,
            }
        except ImportError:
            return {
                "system_memory_used_mb": 0.0,
                "system_memory_total_mb": 0.0,
                "system_memory_percent": 0.0,
            }
    
    def _save_stats(self):
        """Save statistics to file."""
        if not self.current_stats:
            return
        
        # Check if output directory is valid
        if self.output_dir is None:
            return  # Skip saving if directory creation failed
        
        try:
            # Convert tensors to Python primitives for JSON serialization
            serializable_history = []
            for stats_dict in self.stats_history[-50:]:
                clean_dict = {}
                for key, value in stats_dict.items():
                    if torch.is_tensor(value):
                        clean_dict[key] = float(value.item())
                    else:
                        clean_dict[key] = value
                serializable_history.append(clean_dict)
                
            stats_file = self.output_dir / f"stats_step_{self.step_count}.json"
            with open(stats_file, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            # Also save a running CSV
            csv_file = self.output_dir / "stats.csv"
            
            if not csv_file.exists():
                with open(csv_file, 'w') as f:
                    f.write("step,epoch,step_time_sec,samples_per_sec,loss,"
                           "gpu_utilization,gpu_memory_used_mb,gpu_memory_percent,"
                           "system_memory_used_mb,system_memory_percent\n")
            
            with open(csv_file, 'a') as f:
                s = self.current_stats
                loss_val = s.get('loss', 0)
                if torch.is_tensor(loss_val):
                    loss_val = float(loss_val.item())
                f.write(f"{s['step']},{s['epoch']},{s.get('step_time_sec', 0):.4f},"
                       f"{s.get('samples_per_sec', 0):.2f},{loss_val:.6f},"
                       f"{s.get('gpu_utilization', 0):.2f},{s.get('gpu_memory_used_mb', 0):.2f},"
                       f"{s.get('gpu_memory_percent', 0):.2f},{s.get('system_memory_used_mb', 0):.2f},"
                       f"{s.get('system_memory_percent', 0):.2f}\n")
        except Exception as e:
            # Only print error once every 100 steps to avoid spam
            if self.step_count % 100 == 50:
                try:
                    from tqdm import tqdm
                    tqdm.write(f"[ResourceStats] Warning: Failed to save stats: {e}")
                except:
                    print(f"[ResourceStats] Warning: Failed to save stats: {e}")
    
    # ========== Required abstract methods ==========
    
    def start_train(self):
        """Called when training starts."""
        self.training_start_time = time.time()
        try:
            from tqdm import tqdm
            tqdm.write("[ResourceStats] Training started")
        except:
            print("[ResourceStats] Training started")
    
    def stop_train(self):
        """Called when training ends."""
        if self.training_start_time is None:
            return
            
        total_time = time.time() - self.training_start_time
        
        summary = {
            "total_training_time_sec": total_time,
            "total_epochs": self.epoch_count,
            "total_steps": self.step_count,
            "total_samples": self.total_samples,
            "avg_samples_per_sec": self.total_samples / total_time if total_time > 0 else 0,
        }
        
        # Try to save summary
        saved = False
        if self.output_dir is not None:
            try:
                summary_file = self.output_dir / "training_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                saved = True
            except Exception as e:
                summary_file = f"<failed: {e}>"
        else:
            summary_file = "<no output directory>"
        
        try:
            from tqdm import tqdm
            if saved:
                tqdm.write(f"[ResourceStats] Training ended. Summary saved to {summary_file}")
            else:
                tqdm.write(f"[ResourceStats] Training ended. Warning: Could not save summary to {summary_file}")
            tqdm.write(f"  Total time: {total_time:.2f}s")
            tqdm.write(f"  Total epochs: {self.epoch_count}")
            tqdm.write(f"  Total steps: {self.step_count}")
            tqdm.write(f"  Throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
            if self.output_dir and saved:
                csv_file = self.output_dir / "stats.csv"
                if csv_file.exists():
                    tqdm.write(f"  CSV stats: {csv_file}")
                else:
                    tqdm.write(f"  Warning: CSV file was not created")
        except:
            if saved:
                print(f"[ResourceStats] Training ended. Summary saved to {summary_file}")
            else:
                print(f"[ResourceStats] Training ended. Warning: Could not save summary to {summary_file}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Total epochs: {self.epoch_count}")
            print(f"  Total steps: {self.step_count}")
            print(f"  Throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
    
    def start_step(self):
        """Called at the start of each training step."""
        self.step_start_time = time.time()
        self.step_count += 1
    
    def stop_step(self):
        """Called at the end of each training step."""
        if self.step_start_time is None:
            return
            
        step_time = time.time() - self.step_start_time
        
        # Collect resource stats
        gpu_stats = self._get_gpu_stats()
        mem_stats = self._get_memory_stats()
        
        # Update current stats
        self.current_stats.update({
            "step": self.step_count,
            "epoch": self.epoch_count,
            "timestamp": time.time(),
            "step_time_sec": step_time,
            **gpu_stats,
            **mem_stats,
        })
        
        # Store
        self.stats_history.append(dict(self.current_stats))
        
        # Print every 10 steps - use tqdm.write to avoid being overwritten by progress bar
        if self.step_count % 10 == 0:
            loss_str = f"{self.current_stats.get('loss', 0):.4f}" if 'loss' in self.current_stats else 'N/A'
            msg = (f"[ResourceStats] Step {self.step_count}: "
                   f"GPU {gpu_stats['gpu_utilization']:.1f}%, "
                   f"GPU Mem {gpu_stats['gpu_memory_used_mb']:.0f}MB, "
                   f"Loss {loss_str}")
            
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except:
                print(msg)
        
        # Save periodically
        if self.step_count % 50 == 0:
            self._save_stats()
    
    def start_forward(self):
        """Called before forward pass."""
        self.forward_start_time = time.time()
    
    def stop_forward(self):
        """Called after forward pass."""
        if self.forward_start_time:
            forward_time = time.time() - self.forward_start_time
            self.current_stats['forward_time_sec'] = forward_time
    
    def start_backward(self):
        """Called before backward pass."""
        self.backward_start_time = time.time()
    
    def stop_backward(self):
        """Called after backward pass."""
        if self.backward_start_time:
            backward_time = time.time() - self.backward_start_time
            self.current_stats['backward_time_sec'] = backward_time
    
    def start_optimizer_step(self):
        """Called before optimizer step."""
        self.optimizer_start_time = time.time()
    
    def stop_optimizer_step(self):
        """Called after optimizer step."""
        if self.optimizer_start_time:
            optimizer_time = time.time() - self.optimizer_start_time
            self.current_stats['optimizer_time_sec'] = optimizer_time
    
    def start_save_checkpoint(self):
        """Called before saving checkpoint."""
        pass
    
    def stop_save_checkpoint(self):
        """Called after saving checkpoint."""
        pass
    
    def log_step(self, batch_size: int = 1, **kwargs):
        """Log information about a training step."""
        self.total_samples += batch_size
        if batch_size > 0 and 'step_time_sec' in self.current_stats:
            self.current_stats['samples_per_sec'] = batch_size / self.current_stats['step_time_sec']
    
    def log_loss(self, loss: float, **kwargs):
        """Log the loss value."""
        # Convert tensor to float if needed
        if torch.is_tensor(loss):
            self.current_stats['loss'] = float(loss.item())
        else:
            self.current_stats['loss'] = loss
    
    def log_stats(self, **kwargs):
        """Log arbitrary statistics."""
        self.current_stats.update(kwargs)


# Registration name for auto-discovery
trainer_stats_name = "resource_monitoring"


def construct_trainer_stats(conf, **kwargs):
    """Initialize resource monitoring stats."""
    output_dir = getattr(conf, 'stats_output_dir', None)
    return ResourceMonitoringStats(output_dir=output_dir)
