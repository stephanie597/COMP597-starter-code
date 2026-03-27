"""Combined Stats: Resource Monitoring + Carbon Tracking.

Tracks both system resources (GPU, RAM, I/O) and carbon emissions
in a single unified stats class.
"""

import time
import json
import os
import gc
from pathlib import Path
from typing import Dict
import torch
import pandas as pd

# Resource monitoring imports
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

# Carbon tracking imports
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

from src.trainer.stats.base import TrainerStats

# Registration name
trainer_stats_name = "combined_stats"


def construct_trainer_stats(conf=None, **kwargs):
    """Initialize combined stats."""
    # Get device
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get config or use defaults
    if conf and hasattr(conf, 'trainer_stats_configs') and hasattr(conf.trainer_stats_configs, 'combined_stats'):
        output_dir = conf.trainer_stats_configs.combined_stats.output_dir
        run_num = conf.trainer_stats_configs.combined_stats.run_num
        project_name = conf.trainer_stats_configs.combined_stats.project_name
    else:
        # Default: use SLURM directory for main output
        output_dir = "/mnt/teaching/slurm/yxia2/training_stats"
        run_num = 1
        project_name = "resnet152_combined"
    
    return CombinedStats(
        device=device,
        output_dir=output_dir,
        run_num=run_num,
        project_name=project_name
    )


class CombinedStats(TrainerStats):
    """Combined resource monitoring and carbon tracking.
    
    Tracks:
    - GPU utilization and memory
    - System RAM usage
    - Training throughput and I/O time
    - Carbon emissions and energy consumption
    - Training loss
    
    Outputs:
    - stats.csv (GPU/RAM/Throughput/I/O)
    - combined_metrics.png (visualization)
    - carbon_total.csv (total emissions)
    - carbon_steps.csv (per-step emissions)
    """
    
    def __init__(
        self,
        device: torch.device,
        output_dir: str = "./training_stats_backup",
        run_num: int = 1,
        project_name: str = "resnet152_combined"
    ):
        super().__init__()
        
        self.device = device
        self.output_dir = Path(output_dir)
        self.run_num = run_num
        self.project_name = project_name
        
        # Create output directory and clean old files
        self._setup_output_dir()
        
        # Resource monitoring state
        self.stats_history = []
        self.current_stats = {}
        self.step_count = 0
        self.epoch_count = 0
        self.total_samples = 0
        self.losses = []
        
        # Timing
        self.step_start_time = None
        self.training_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None
        self.optimizer_start_time = None
        
        # GPU monitoring
        self.gpu_handle = None
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_id = device.index if device.index is not None else 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except:
                print("[CombinedStats] Could not get GPU handle")
        
        # Carbon tracking
        self.carbon_enabled = CODECARBON_AVAILABLE
        if self.carbon_enabled:
            self._setup_carbon_tracking()
        else:
            print("[CombinedStats] CodeCarbon not available - carbon tracking disabled")
        
        print(f"[CombinedStats] Initialized successfully")
        print(f"[CombinedStats] Output directory: {self.output_dir}")
        print(f"[CombinedStats] Resource monitoring: ✓")
        print(f"[CombinedStats] Carbon tracking: {'✓' if self.carbon_enabled else '✗'}")
    
    def _setup_output_dir(self):
        """Create output directory and clean old files."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean old files
            for pattern in ["stats*.csv", "training_*.png", "run_*_carbon_*.csv"]:
                for old_file in self.output_dir.glob(pattern):
                    old_file.unlink()
        except Exception as e:
            print(f"[CombinedStats] Warning: Could not clean directory: {e}")
    
    def _setup_carbon_tracking(self):
        """Initialize carbon tracking."""
        try:
            gpu_id = self.device.index if self.device.type == "cuda" and self.device.index is not None else None
            run_prefix = f"run_{self.run_num}_"
            device_suffix = f"_gpu{gpu_id}" if gpu_id is not None else "_cpu"
            
            self.carbon_total_file = self.output_dir / f"{run_prefix}carbon_total{device_suffix}.csv"
            self.carbon_steps_file = self.output_dir / f"{run_prefix}carbon_steps{device_suffix}.csv"
            
            self.carbon_total_tracker = OfflineEmissionsTracker(
                project_name=self.project_name,
                country_iso_code="CAN",
                region="quebec",
                save_to_file=True,
                save_to_api=False,
                output_file=str(self.carbon_total_file),
                gpu_ids=[gpu_id] if gpu_id is not None else None,
                log_level="error"
            )
            
            self.carbon_step_tracker = OfflineEmissionsTracker(
                project_name=self.project_name,
                country_iso_code="CAN",
                region="quebec",
                save_to_file=True,
                save_to_api=False,
                output_file=str(self.carbon_steps_file),
                gpu_ids=[gpu_id] if gpu_id is not None else None,
                log_level="error"
            )
        except Exception as e:
            print(f"[CombinedStats] Error setting up carbon tracking: {e}")
            self.carbon_enabled = False
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics."""
        stats = {
            "gpu_utilization": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            stats["gpu_memory_percent"] = (stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100
            
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
        """Save resource statistics to files."""
        if not self.current_stats or self.output_dir is None:
            return
        
        try:
            # Prepare data
            clean_dict = {}
            for key, value in self.current_stats.items():
                if torch.is_tensor(value):
                    clean_dict[key] = float(value.item())
                else:
                    clean_dict[key] = value
            
            # Save CSV
            csv_file = self.output_dir / "stats.csv"
            if not csv_file.exists():
                with open(csv_file, 'w') as f:
                    f.write("step,epoch,step_time_sec,io_time_sec,samples_per_sec,loss,"
                           "gpu_utilization,gpu_memory_used_mb,gpu_memory_percent,"
                           "system_memory_used_mb,system_memory_percent\n")
            
            with open(csv_file, 'a') as f:
                s = clean_dict
                loss_val = s.get('loss', 0)
                f.write(f"{s['step']},{s['epoch']},{s.get('step_time_sec', 0):.4f},"
                       f"{s.get('io_time_sec', 0):.4f},{s.get('samples_per_sec', 0):.2f},"
                       f"{loss_val:.6f},"
                       f"{s.get('gpu_utilization', 0):.2f},{s.get('gpu_memory_used_mb', 0):.2f},"
                       f"{s.get('gpu_memory_percent', 0):.2f},{s.get('system_memory_used_mb', 0):.2f},"
                       f"{s.get('system_memory_percent', 0):.2f}\n")
        except Exception as e:
            if self.step_count % 100 == 50:
                print(f"[CombinedStats] Warning: Failed to save stats: {e}")
    
    def _generate_plots(self):
        """Generate visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not self.stats_history:
                return
            
            # Extract data
            steps = [s.get('step', 0) for s in self.stats_history]
            gpu_util = [s.get('gpu_utilization', 0) for s in self.stats_history]
            gpu_mem = [s.get('gpu_memory_used_mb', 0) for s in self.stats_history]
            sys_mem = [s.get('system_memory_used_mb', 0) for s in self.stats_history]
            loss = [s.get('loss', 0) for s in self.stats_history if 'loss' in s]
            loss_steps = [s.get('step', 0) for s in self.stats_history if 'loss' in s]
            throughput = [s.get('samples_per_sec', 0) for s in self.stats_history]
            
            # Calculate X-axis range
            max_step = max(steps) if steps else 8000
            x_min, x_max = 0, max_step
            
            # Create 2x3 figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('ResNet152 Training Metrics (Combined)', fontsize=16, fontweight='bold')
            
            # GPU Utilization
            axes[0, 0].plot(steps, gpu_util, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('GPU Utilization (%)')
            axes[0, 0].set_title('GPU Utilization')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(x_min, x_max)  # Set X-axis range
            axes[0, 0].set_ylim(0, 100)  # Already 0-100
            if gpu_util:
                avg_util = sum(gpu_util) / len(gpu_util)
                axes[0, 0].axhline(y=avg_util, color='r', linestyle='--', 
                                   label=f'Avg: {avg_util:.1f}%', alpha=0.7)
                axes[0, 0].legend()
            
            # GPU Memory
            axes[0, 1].plot(steps, gpu_mem, 'g-', linewidth=1.5, alpha=0.7)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('GPU Memory (MB)')
            axes[0, 1].set_title('GPU Memory Usage')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(x_min, x_max)  # Set X-axis range
            # Auto-adjust Y-axis for better visibility
            if gpu_mem:
                avg_mem = sum(gpu_mem) / len(gpu_mem)
                axes[0, 1].axhline(y=avg_mem, color='r', linestyle='--',
                                   label=f'Avg: {avg_mem:.0f} MB', alpha=0.7)
                axes[0, 1].legend()
            
            # System Memory
            axes[0, 2].plot(steps, sys_mem, 'orange', linewidth=1.5, alpha=0.7)
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('System Memory (MB)')
            axes[0, 2].set_title('System Memory (RAM)')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_xlim(x_min, x_max)  # Set X-axis range
            # Auto-adjust Y-axis for better visibility
            if sys_mem:
                avg_sys_mem = sum(sys_mem) / len(sys_mem)
                axes[0, 2].axhline(y=avg_sys_mem, color='r', linestyle='--',
                                   label=f'Avg: {avg_sys_mem:.0f} MB', alpha=0.7)
                axes[0, 2].legend()
            
            # Loss
            if loss:
                axes[1, 0].plot(loss_steps, loss, 'r-', linewidth=1.5, alpha=0.7)
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_xlim(x_min, x_max)  # Set X-axis range
                # Auto-adjust Y-axis for better visibility
            else:
                axes[1, 0].text(0.5, 0.5, 'No loss data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Throughput
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
                                   label=f'Avg: {avg_throughput:.1f} samples/s', alpha=0.7)
                axes[1, 1].legend()
            
            # Empty space
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            output_file = self.output_dir / "combined_metrics.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[CombinedStats] Visualization saved to {output_file}")
        except Exception as e:
            print(f"[CombinedStats] Could not generate plots: {e}")
    
    def _copy_to_local_backup(self):
        """Copy important files to local backup directory."""
        try:
            import shutil
            
            # Local backup directory (relative to where script is run)
            local_backup = Path("./training_stats_backup")
            
            # Check if source and destination are the same
            if self.output_dir.resolve() == local_backup.resolve():
                # Already in the local directory, no need to copy
                return
            
            local_backup.mkdir(parents=True, exist_ok=True)
            
            # Files to copy
            files_to_copy = [
                "stats.csv",
                "combined_metrics.png",
            ]
            
            # Copy each file
            copied_files = []
            for filename in files_to_copy:
                src = self.output_dir / filename
                dst = local_backup / filename
                
                if src.exists():
                    shutil.copy2(src, dst)
                    copied_files.append(filename)
            
            # Copy carbon files if they exist
            for carbon_file in self.output_dir.glob("run_*_carbon_*.csv"):
                dst = local_backup / carbon_file.name
                shutil.copy2(carbon_file, dst)
                copied_files.append(carbon_file.name)
            
            if copied_files:
                print(f"[CombinedStats] Copied {len(copied_files)} files to ./training_stats_backup/")
            
        except Exception as e:
            print(f"[CombinedStats] Warning: Could not copy to local backup: {e}")
    
    # ===== TrainerStats interface methods =====
    
    def start_train(self):
        """Start training tracking."""
        self.training_start_time = time.time()
        
        if self.carbon_enabled:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                self.carbon_total_tracker.start()
            except Exception as e:
                print(f"[CombinedStats] Error starting carbon tracker: {e}")
        
        print("[CombinedStats] Training started")
    
    def stop_train(self):
        """Stop training and save results."""
        if self.training_start_time is None:
            return
        
        total_time = time.time() - self.training_start_time
        
        # Calculate summary (but don't save to JSON)
        summary = {
            "total_training_time_sec": total_time,
            "total_epochs": self.epoch_count,
            "total_steps": self.step_count,
            "total_samples": self.total_samples,
            "avg_samples_per_sec": self.total_samples / total_time if total_time > 0 else 0,
        }
        
        # Stop carbon tracking
        carbon_emissions = None
        if self.carbon_enabled:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                carbon_emissions = self.carbon_total_tracker.stop()
            except Exception as e:
                print(f"[CombinedStats] Error stopping carbon tracker: {e}")
        
        # Generate plots
        if len(self.stats_history) > 0:
            self._generate_plots()
        
        # Copy files to local backup directory
        self._copy_to_local_backup()
        
        # Print summary
        print(f"\n{'='*70}")
        print("[CombinedStats] Training Completed - Summary")
        print(f"{'='*70}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total steps: {self.step_count}")
        print(f"  Throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
        if carbon_emissions:
            print(f"  CO2 emissions: {carbon_emissions:.6f} kg CO2eq")
        print(f"  Results saved to: {self.output_dir}")
        print(f"  Local backup: ./training_stats_backup/")
        print(f"{'='*70}\n")
    
    def start_step(self, **kwargs):
        """Start tracking step."""
        self.step_start_time = time.time()
        self.step_count += 1
        
        if self.carbon_enabled:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                self.carbon_step_tracker.start()
            except:
                pass
    
    def stop_step(self):
        """Stop tracking step."""
        if self.step_start_time is None:
            return
        
        step_time = time.time() - self.step_start_time
        
        # Calculate I/O time
        forward_time = self.current_stats.get('forward_time_sec', 0)
        backward_time = self.current_stats.get('backward_time_sec', 0)
        optimizer_time = self.current_stats.get('optimizer_time_sec', 0)
        io_time = max(0, step_time - (forward_time + backward_time + optimizer_time))
        
        # Collect stats
        gpu_stats = self._get_gpu_stats()
        mem_stats = self._get_memory_stats()
        
        self.current_stats.update({
            "step": self.step_count,
            "epoch": self.epoch_count,
            "timestamp": time.time(),
            "step_time_sec": step_time,
            "io_time_sec": io_time,
            **gpu_stats,
            **mem_stats,
        })
        
        self.stats_history.append(dict(self.current_stats))
        
        # Memory management
        if len(self.stats_history) > 1000:
            self.stats_history = self.stats_history[-1000:]
        
        if self.step_count % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print progress
        if self.step_count % 10 == 0:
            loss_str = f"{self.current_stats.get('loss', 0):.4f}" if 'loss' in self.current_stats else 'N/A'
            msg = (f"[CombinedStats] Step {self.step_count}: "
                   f"GPU {gpu_stats['gpu_utilization']:.1f}%, "
                   f"GPU Mem {gpu_stats['gpu_memory_used_mb']:.0f}MB, "
                   f"I/O {io_time*1000:.1f}ms, "
                   f"Loss {loss_str}")
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except:
                print(msg)
        
        # Save every step (not just every 50)
        self._save_stats()
        
        # Stop carbon step tracker
        if self.carbon_enabled:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                self.carbon_step_tracker.stop()
            except:
                pass
    
    def start_forward(self):
        self.forward_start_time = time.time()
    
    def stop_forward(self):
        if self.forward_start_time:
            self.current_stats['forward_time_sec'] = time.time() - self.forward_start_time
    
    def start_backward(self):
        self.backward_start_time = time.time()
    
    def stop_backward(self):
        if self.backward_start_time:
            self.current_stats['backward_time_sec'] = time.time() - self.backward_start_time
    
    def start_optimizer_step(self):
        self.optimizer_start_time = time.time()
    
    def stop_optimizer_step(self):
        if self.optimizer_start_time:
            self.current_stats['optimizer_time_sec'] = time.time() - self.optimizer_start_time
    
    def start_save_checkpoint(self):
        pass
    
    def stop_save_checkpoint(self):
        pass
    
    def log_step(self, batch_size: int = 1, **kwargs):
        self.total_samples += batch_size
        if batch_size > 0 and 'step_time_sec' in self.current_stats:
            self.current_stats['samples_per_sec'] = batch_size / self.current_stats['step_time_sec']
    
    def log_loss(self, loss, **kwargs):
        """Log the loss value. Disabled to avoid GPU-CPU sync overhead."""
        pass
    
    def log_stats(self, **kwargs):
        self.current_stats.update(kwargs)