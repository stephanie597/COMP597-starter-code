"""Carbon Emissions Tracking for ResNet152 Training.

Tracks energy consumption and CO2 emissions using CodeCarbon library.
Simplified implementation with minimal configuration required.
"""

import logging
import os
import torch
import pandas as pd
from pathlib import Path

# Try to import CodeCarbon
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("[CarbonTracker] Warning: codecarbon not installed")
    print("[CarbonTracker] Install with: pip install codecarbon --break-system-packages")

import src.config as config
from src.trainer.stats.base import TrainerStats

logger = logging.getLogger(__name__)

# Registration name for auto-discovery
trainer_stats_name = "carbon_tracker"


def construct_trainer_stats(conf: config.Config = None, **kwargs):
    """Initialize carbon tracking stats.
    
    Args:
        conf: Configuration object (optional)
        **kwargs: Additional arguments including 'device'
    
    Returns:
        CarbonTracker instance
    """
    # Get device
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided. Using default PyTorch device")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get configuration or use defaults
    if conf and hasattr(conf, 'trainer_stats_configs') and hasattr(conf.trainer_stats_configs, 'carbon_tracker'):
        run_num = conf.trainer_stats_configs.carbon_tracker.run_num
        project_name = conf.trainer_stats_configs.carbon_tracker.project_name
        output_dir = conf.trainer_stats_configs.carbon_tracker.output_dir
    else:
        # Default configuration - use SLURM directory
        run_num = 1
        project_name = "resnet152_carbon"
        output_dir = "/mnt/teaching/slurm/yxia2/training_stats"
        logger.info(f"Using default carbon tracker config: run={run_num}, project={project_name}")
    
    return CarbonTracker(
        device=device,
        run_num=run_num,
        project_name=project_name,
        output_dir=output_dir
    )


class CarbonTracker(TrainerStats):
    """Track carbon emissions and energy consumption during training.
    
    Measures:
    - Total training emissions (kg CO2)
    - Energy consumption (kWh)
    - Per-step emissions
    - Forward/backward/optimizer emissions
    
    Outputs CSV files with detailed emission data.
    
    Parameters
    ----------
    device : torch.device
        PyTorch device (GPU or CPU)
    run_num : int
        Run number for tracking different experiments
    project_name : str
        Project name for CodeCarbon
    output_dir : str
        Directory for saving CSV files
    """
    
    def __init__(
        self,
        device: torch.device,
        run_num: int = 1,
        project_name: str = "resnet152_carbon",
        output_dir: str = "./training_stats_backup"
    ):
        super().__init__()
        
        if not CODECARBON_AVAILABLE:
            print("[CarbonTracker] CodeCarbon not available - tracking disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.device = device
        self.run_num = run_num
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        
        # Tracking state
        self.step_count = 0
        self.losses = []
        
        # Create output directory and clean old files
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clean_old_files()
        
        # Get GPU ID
        if device.type == "cuda":
            self.gpu_id = device.index if device.index is not None else 0
        else:
            self.gpu_id = None
        
        # File naming
        run_prefix = f"run_{run_num}_"
        device_suffix = f"_gpu{self.gpu_id}" if self.gpu_id is not None else "_cpu"
        
        self.total_emissions_file = self.output_dir / f"{run_prefix}carbon_total{device_suffix}.csv"
        self.step_emissions_file = self.output_dir / f"{run_prefix}carbon_steps{device_suffix}.csv"
        
        # Initialize trackers
        try:
            # Total training emissions tracker
            self.total_tracker = OfflineEmissionsTracker(
                project_name=project_name,
                country_iso_code="CAN",
                region="quebec",
                save_to_file=True,
                save_to_api=False,
                output_file=str(self.total_emissions_file),
                gpu_ids=[self.gpu_id] if self.gpu_id is not None else None,
                log_level="error",
                tracking_mode="machine"
            )
            
            # Per-step emissions tracker
            self.step_tracker = OfflineEmissionsTracker(
                project_name=project_name,
                country_iso_code="CAN",
                region="quebec",
                save_to_file=True,
                save_to_api=False,
                output_file=str(self.step_emissions_file),
                gpu_ids=[self.gpu_id] if self.gpu_id is not None else None,
                log_level="error",
                tracking_mode="machine"
            )
            
            print(f"[CarbonTracker] Initialized successfully")
            print(f"[CarbonTracker] Output directory: {self.output_dir}")
            print(f"[CarbonTracker] Device: {device}")
            print(f"[CarbonTracker] Files will be saved as:")
            print(f"  - {self.total_emissions_file.name}")
            print(f"  - {self.step_emissions_file.name}")
            
        except Exception as e:
            print(f"[CarbonTracker] Error initializing trackers: {e}")
            self.enabled = False
    
    def start_train(self):
        """Start tracking total training emissions."""
        if not self.enabled:
            return
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            
            self.total_tracker.start()
            print("[CarbonTracker] Started tracking training emissions")
            
        except Exception as e:
            print(f"[CarbonTracker] Error in start_train: {e}")
    
    def stop_train(self):
        """Stop tracking and save final results."""
        if not self.enabled:
            return
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            
            # Stop total tracker
            total_emissions = self.total_tracker.stop()
            
            # Save losses
            if self.losses:
                self._save_losses()
            
            # Generate plots from CSV data
            self._generate_plots()
            
            # Copy files to local backup
            self._copy_to_local_backup()
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"[CarbonTracker] Training Completed - Emission Summary")
            print(f"{'='*60}")
            print(f"  Total CO2 emissions: {total_emissions:.6f} kg CO2eq")
            print(f"  Total steps tracked: {self.step_count}")
            print(f"  Results saved to: {self.output_dir}")
            print(f"  Local backup: ./training_stats_backup/")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[CarbonTracker] Error in stop_train: {e}")
    
    def start_step(self, **kwargs):
        """Start tracking emissions for this training step."""
        if not self.enabled:
            return
        
        try:
            self.step_count += 1
            
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            
            self.step_tracker.start()
            
        except Exception as e:
            if self.step_count == 1:
                print(f"[CarbonTracker] Error in start_step: {e}")
    
    def stop_step(self):
        """Stop tracking emissions for this training step."""
        if not self.enabled:
            return
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            
            self.step_tracker.stop()
            
            # Print progress every 500 steps
            if self.step_count % 500 == 0:
                print(f"[CarbonTracker] Tracked {self.step_count} steps")
                
        except Exception as e:
            pass  # Silent to avoid spam
    
    def start_forward(self):
        """Called before forward pass."""
        pass  # Not tracking sub-steps by default for simplicity
    
    def stop_forward(self):
        """Called after forward pass."""
        pass
    
    def start_backward(self):
        """Called before backward pass."""
        pass
    
    def stop_backward(self):
        """Called after backward pass."""
        pass
    
    def start_optimizer_step(self):
        """Called before optimizer step."""
        pass
    
    def stop_optimizer_step(self):
        """Called after optimizer step."""
        pass
    
    def start_save_checkpoint(self):
        """Called before saving checkpoint."""
        pass
    
    def stop_save_checkpoint(self):
        """Called after saving checkpoint."""
        pass
    
    def log_step(self, batch_size: int = 1, **kwargs):
        """Log step information."""
        pass
    
    def log_loss(self, loss: torch.Tensor, **kwargs):
        """Log loss value for this step."""
        if not self.enabled:
            return
        
        try:
            loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            self.losses.append({
                "step": self.step_count,
                "loss": loss_value
            })
        except Exception as e:
            pass
    
    def log_stats(self, **kwargs):
        """Log additional statistics."""
        pass
    
    def _save_losses(self):
        """Save loss data to CSV file."""
        try:
            losses_file = self.output_dir / f"run_{self.run_num}_carbon_losses.csv"
            df = pd.DataFrame(self.losses)
            df.to_csv(losses_file, index=False)
            print(f"[CarbonTracker] Losses saved to {losses_file}")
        except Exception as e:
            print(f"[CarbonTracker] Error saving losses: {e}")
    
    def _generate_plots(self):
        """Generate visualization plots from carbon emission data."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Read step emissions CSV
            if not self.step_emissions_file.exists():
                print("[CarbonTracker] No step emissions data found, skipping plots")
                return
            
            df = pd.read_csv(self.step_emissions_file)
            
            if len(df) == 0:
                print("[CarbonTracker] Empty emissions data, skipping plots")
                return
            
            # Extract data - use diff() to get per-step values from cumulative data
            steps = list(range(1, len(df) + 1))
            import numpy as np
            
            # Per-step values (diff of cumulative)
            emissions_cumul = df['emissions'].values * 1000  # kg -> g CO2
            energy_cumul = df['energy_consumed'].values * 1e6  # kWh -> mWh
            
            emissions_per_step = np.diff(emissions_cumul, prepend=0)
            energy_per_step = np.diff(energy_cumul, prepend=0)
            
            # Cumulative values for cumulative plots
            emissions_cumul_g = emissions_cumul
            energy_cumul_mwh = energy_cumul
            
            # Calculate X-axis range
            max_step = max(steps) if steps else len(df)
            x_min, x_max = 0, max_step
            
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Carbon Emissions Tracking', fontsize=16, fontweight='bold')
            
            # Plot 1: CO2 Emissions per step
            axes[0, 0].plot(steps, emissions_per_step, 'g-', linewidth=1, alpha=0.7)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('CO2 Emissions (g CO2eq/step)')
            axes[0, 0].set_title('Carbon Emissions per Step')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(x_min, x_max)
            avg_emissions = emissions_per_step.mean()
            axes[0, 0].axhline(y=avg_emissions, color='r', linestyle='--',
                               label=f'Avg: {avg_emissions:.6f} g/step', alpha=0.7)
            axes[0, 0].legend()

            # Plot 2: Energy per step
            axes[0, 1].plot(steps, energy_per_step, 'b-', linewidth=1, alpha=0.7)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Energy Consumed (mWh/step)')
            axes[0, 1].set_title('Energy Consumption per Step')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(x_min, x_max)
            avg_energy = energy_per_step.mean()
            axes[0, 1].axhline(y=avg_energy, color='r', linestyle='--',
                               label=f'Avg: {avg_energy:.4f} mWh/step', alpha=0.7)
            axes[0, 1].legend()

            # Plot 3: Cumulative CO2
            axes[1, 0].plot(steps, emissions_cumul_g, 'g-', linewidth=1.5, alpha=0.7)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Cumulative CO2 (g CO2eq)')
            axes[1, 0].set_title('Cumulative Carbon Emissions')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(x_min, x_max)
            total_emissions = emissions_cumul_g[-1]
            axes[1, 0].text(0.98, 0.02, f'Total: {total_emissions:.4f} g CO2eq',
                           transform=axes[1, 0].transAxes,
                           ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Plot 4: Cumulative Energy
            axes[1, 1].plot(steps, energy_cumul_mwh, 'b-', linewidth=1.5, alpha=0.7)
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Cumulative Energy (mWh)')
            axes[1, 1].set_title('Cumulative Energy Consumption')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(x_min, x_max)
            total_energy = energy_cumul_mwh[-1]
            axes[1, 1].text(0.98, 0.02, f'Total: {total_energy:.2f} mWh',
                           transform=axes[1, 1].transAxes,
                           ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            
            # Save figure
            output_file = self.output_dir / "carbon_emissions_plot.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[CarbonTracker] Visualization saved to {output_file}")
            
        except Exception as e:
            print(f"[CarbonTracker] Could not generate plots: {e}")
    
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
            files_to_copy = []
            
            # Add carbon CSV files
            if self.total_emissions_file.exists():
                files_to_copy.append(self.total_emissions_file)
            if self.step_emissions_file.exists():
                files_to_copy.append(self.step_emissions_file)
            
            # Add losses file
            losses_file = self.output_dir / f"run_{self.run_num}_carbon_losses.csv"
            if losses_file.exists():
                files_to_copy.append(losses_file)
            
            # Add plot
            plot_file = self.output_dir / "carbon_emissions_plot.png"
            if plot_file.exists():
                files_to_copy.append(plot_file)
            
            # Copy files
            copied_count = 0
            for src_file in files_to_copy:
                dst = local_backup / src_file.name
                shutil.copy2(src_file, dst)
                copied_count += 1
            
            if copied_count > 0:
                print(f"[CarbonTracker] Copied {copied_count} files to ./training_stats_backup/")
            
        except Exception as e:
            print(f"[CarbonTracker] Warning: Could not copy to local backup: {e}")
    
    def _clean_old_files(self):
        """Clean old carbon tracking files from output directory."""
        try:
            # Patterns to clean
            patterns = [
                "run_*_carbon_*.csv",
                "carbon_emissions_plot.png",
                "*_carbon_losses.csv"
            ]
            
            cleaned_count = 0
            for pattern in patterns:
                for old_file in self.output_dir.glob(pattern):
                    old_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                print(f"[CarbonTracker] Cleaned {cleaned_count} old files")
                
        except Exception as e:
            print(f"[CarbonTracker] Warning: Could not clean old files: {e}")