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
        
        # Primary directory (try scratch first)
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
        
        # Always create local backup directory
        self.backup_dir = Path("./training_stats_backup")
        
        # Try to create primary directory
        primary_created = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            primary_created = True
            
            # Clean up old files in primary directory
            for old_file in self.output_dir.glob("stats*.csv"):
                old_file.unlink()
            for old_file in self.output_dir.glob("training_*.png"):
                old_file.unlink()
            
            print(f"[ResourceStats] Primary output: {self.output_dir}")
        except Exception as e:
            print(f"[ResourceStats] Warning: Cannot create primary directory {self.output_dir}: {e}")
        
        # Always create backup directory
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean up old files in backup directory
            for old_file in self.backup_dir.glob("stats*.csv"):
                old_file.unlink()
            for old_file in self.backup_dir.glob("training_*.png"):
                old_file.unlink()
            
            print(f"[ResourceStats] Backup output: {self.backup_dir}")
        except Exception as e:
            print(f"[ResourceStats] Warning: Cannot create backup directory: {e}")
            self.backup_dir = None
        
        # If primary failed, use backup as primary
        if not primary_created and self.backup_dir:
            self.output_dir = self.backup_dir
            self.backup_dir = None
            print(f"[ResourceStats] Using backup as primary: {self.output_dir}")
        
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
        """Save statistics to CSV file (both primary and backup)."""
        if not self.current_stats:
            return
        
        # Prepare CSV row
        s = self.current_stats
        loss_val = s.get('loss', 0)
        if torch.is_tensor(loss_val):
            loss_val = float(loss_val.item())
        csv_row = (f"{s['step']},{s['epoch']},{s.get('step_time_sec', 0):.4f},"
                  f"{s.get('io_time_sec', 0):.4f},"
                  f"{s.get('samples_per_sec', 0):.2f},{loss_val:.6f},"
                  f"{s.get('gpu_utilization', 0):.2f},{s.get('gpu_memory_used_mb', 0):.2f},"
                  f"{s.get('gpu_memory_percent', 0):.2f},{s.get('system_memory_used_mb', 0):.2f},"
                  f"{s.get('system_memory_percent', 0):.2f}\n")
        
        # Save to both locations
        for directory in [self.output_dir, self.backup_dir]:
            if directory is None:
                continue
            
            try:
                # Save CSV only (no JSON)
                csv_file = directory / "stats.csv"
                if not csv_file.exists():
                    with open(csv_file, 'w') as f:
                        f.write("step,epoch,step_time_sec,io_time_sec,samples_per_sec,loss,"
                               "gpu_utilization,gpu_memory_used_mb,gpu_memory_percent,"
                               "system_memory_used_mb,system_memory_percent\n")
                
                with open(csv_file, 'a') as f:
                    f.write(csv_row)
                    
            except Exception as e:
                if self.step_count % 100 == 50:
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"[ResourceStats] Warning: Failed to save to {directory}: {e}")
                    except:
                        print(f"[ResourceStats] Warning: Failed to save to {directory}: {e}")
    
    def _generate_plots(self):
        """Generate visualization plots of training metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Extract data from history
            steps = [s.get('step', 0) for s in self.stats_history]
            gpu_util = [s.get('gpu_utilization', 0) for s in self.stats_history]
            gpu_mem = [s.get('gpu_memory_used_mb', 0) for s in self.stats_history]
            sys_mem = [s.get('system_memory_used_mb', 0) for s in self.stats_history]
            io_time = [s.get('io_time_sec', 0) * 1000 for s in self.stats_history]  # Convert to ms
            loss = [s.get('loss', 0) for s in self.stats_history if 'loss' in s]
            loss_steps = [s.get('step', 0) for s in self.stats_history if 'loss' in s]
            throughput = [s.get('samples_per_sec', 0) for s in self.stats_history]
            
            
            # Calculate X-axis range
            max_step = max(steps) if steps else 8000
            x_min, x_max = 0, max_step
            # Create figure with 6 subplots (2x3)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('ResNet152 Training Metrics', fontsize=16, fontweight='bold')
            
            # Plot 1: GPU Utilization
            axes[0, 0].plot(steps, gpu_util, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('GPU Utilization (%)')
            axes[0, 0].set_title('GPU Utilization Over Time')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(x_min, x_max)  # Set X-axis range
            axes[0, 0].set_ylim(0, 100)
            if gpu_util:
                avg_util = sum(gpu_util) / len(gpu_util)
                axes[0, 0].axhline(y=avg_util, color='r', linestyle='--', 
                                   label=f'Average: {avg_util:.1f}%', alpha=0.7)
                axes[0, 0].legend()
            
            # Plot 2: GPU Memory Usage
            axes[0, 1].plot(steps, gpu_mem, 'g-', linewidth=1.5, alpha=0.7)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('GPU Memory (MB)')
            axes[0, 1].set_title('GPU Memory Usage')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(bottom=0)  # Start from 0, auto-adjust top
            if gpu_mem:
                avg_mem = sum(gpu_mem) / len(gpu_mem)
                max_mem = max(gpu_mem)
                # Set top limit with some padding for better visibility
                axes[0, 1].set_ylim(0, max_mem * 1.1)  # 0 to max + 10% padding
                axes[0, 1].axhline(y=avg_mem, color='r', linestyle='--',
                                   label=f'Average: {avg_mem:.0f} MB', alpha=0.7)
                axes[0, 1].legend()
            
            # Plot 3: System Memory (RAM) Usage
            axes[0, 2].plot(steps, sys_mem, 'orange', linewidth=1.5, alpha=0.7)
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('System Memory (MB)')
            axes[0, 2].set_title('System Memory (RAM) Usage')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_xlim(x_min, x_max)  # Set X-axis range
            # Auto-adjust Y-axis for better visibility
            if sys_mem:
                avg_sys_mem = sum(sys_mem) / len(sys_mem)
                axes[0, 2].axhline(y=avg_sys_mem, color='r', linestyle='--',
                                   label=f'Average: {avg_sys_mem:.0f} MB', alpha=0.7)
                axes[0, 2].legend()
            
            # Plot 4: Training Loss
            if loss:
                axes[1, 0].plot(loss_steps, loss, 'r-', linewidth=1.5, alpha=0.7)
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_xlim(x_min, x_max)  # Set X-axis range
                # Auto-adjust Y-axis for better visibility
            else:
                axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Training Loss (No Data)')
            
            # Plot 5: Throughput
            axes[1, 1].plot(steps, throughput, 'm-', linewidth=1.5, alpha=0.7)
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Throughput (samples/sec)')
            axes[1, 1].set_title('Training Throughput')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(x_min, x_max)  # Set X-axis range
            # Auto-adjust Y-axis for better visibility
            if throughput:
                avg_throughput = sum(throughput) / len(throughput)
                axes[1, 1].axhline(y=avg_throughput, color='r', linestyle='--',
                                   label=f'Average: {avg_throughput:.1f} samples/s', alpha=0.7)
                axes[1, 1].legend()
            
            # Plot 6: Empty (I/O data available in CSV and console output)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_file = self.output_dir / "training_metrics.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[ResourceStats] Visualization saved to {output_file}")
            
        except Exception as e:
            print(f"[ResourceStats] Warning: Could not generate plots: {e}")
    
    def _copy_to_local_backup(self):
        """Copy important files to local backup directory."""
        try:
            import shutil
            
            # Local backup directory
            local_backup = Path("./training_stats_backup")
            
            # Check if source and destination are the same
            if self.output_dir.resolve() == local_backup.resolve():
                # Already in the local directory, no need to copy
                return
            
            local_backup.mkdir(parents=True, exist_ok=True)
            
            # Files to copy
            files_to_copy = [
                "stats.csv",
                "training_metrics.png",
                "training_summary.json",
            ]
            
            # Copy each file if it exists
            copied_files = []
            for filename in files_to_copy:
                src = self.output_dir / filename
                dst = local_backup / filename
                
                if src.exists():
                    shutil.copy2(src, dst)
                    copied_files.append(filename)
            
            if copied_files:
                try:
                    from tqdm import tqdm
                    tqdm.write(f"[ResourceStats] Copied {len(copied_files)} files to ./training_stats_backup/")
                except:
                    print(f"[ResourceStats] Copied {len(copied_files)} files to ./training_stats_backup/")
            
        except Exception as e:
            print(f"[ResourceStats] Warning: Could not copy to local backup: {e}")
    
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
        
        # Generate visualization plots
        if self.output_dir is not None and len(self.stats_history) > 0:
            self._generate_plots()
        
        # Copy files to local backup
        self._copy_to_local_backup()
        
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
                plots_file = self.output_dir / "training_metrics.png"
                if plots_file.exists():
                    tqdm.write(f"  Visualization: {plots_file}")
                tqdm.write(f"  Local backup: ./training_stats_backup/")
        except:
            if saved:
                print(f"[ResourceStats] Training ended. Summary saved to {summary_file}")
            else:
                print(f"[ResourceStats] Training ended. Warning: Could not save summary to {summary_file}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Total epochs: {self.epoch_count}")
            print(f"  Total steps: {self.step_count}")
            print(f"  Throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
            print(f"  Local backup: ./training_stats_backup/")
    
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
        
        # Calculate I/O time (data loading + overhead)
        forward_time = self.current_stats.get('forward_time_sec', 0)
        backward_time = self.current_stats.get('backward_time_sec', 0)
        optimizer_time = self.current_stats.get('optimizer_time_sec', 0)
        
        # I/O time = total time - (forward + backward + optimizer)
        io_time = step_time - (forward_time + backward_time + optimizer_time)
        io_time = max(0, io_time)  # Prevent negative values
        
        # Update current stats
        self.current_stats.update({
            "step": self.step_count,
            "epoch": self.epoch_count,
            "timestamp": time.time(),
            "step_time_sec": step_time,
            "io_time_sec": io_time,  # ← New: I/O time
            **gpu_stats,
            **mem_stats,
        })
        
        # Store
        self.stats_history.append(dict(self.current_stats))
        
        # Memory management: limit history size to prevent memory bloat
        if len(self.stats_history) > 1000:
            self.stats_history = self.stats_history[-1000:]
        
        # Periodic cleanup to prevent memory accumulation
        if self.step_count % 500 == 0:
            import gc
            gc.collect()  # Trigger Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear PyTorch GPU cache
        
        # Print every 10 steps - use tqdm.write to avoid being overwritten by progress bar
        if self.step_count % 10 == 0:
            loss_str = f"{self.current_stats.get('loss', 0):.4f}" if 'loss' in self.current_stats else 'N/A'
            msg = (f"[ResourceStats] Step {self.step_count}: "
                   f"GPU {gpu_stats['gpu_utilization']:.1f}%, "
                   f"GPU Mem {gpu_stats['gpu_memory_used_mb']:.0f}MB, "
                   f"I/O {io_time*1000:.1f}ms, "  # ← New: show I/O time
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