#!/usr/bin/env python3
"""
Generate training metrics plots from stats.csv

This script reads the stats.csv file and generates visualization plots.
Can be run independently without re-training.

Usage:
    python generate_plots.py [csv_file] [output_png]
    
Examples:
    python generate_plots.py stats.csv training_metrics_l.png
    python generate_plots.py training_stats_backup/stats.csv training_metrics_l.png
    python generate_plots.py  # Uses default paths
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def generate_plots(csv_file="training_stats_backup/stats.csv", 
                   output_file="training_stats_backup/training_metrics_l.png"):
    """Generate training metrics plots from CSV file.
    
    Args:
        csv_file: Path to the stats.csv file
        output_file: Path where to save the PNG file
    """
    
    print(f"Reading data from: {csv_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        print("\nTry specifying the correct path:")
        print(f"  python {sys.argv[0]} /path/to/stats.csv")
        sys.exit(1)
    
    print(f"Found {len(df)} data points")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    # Extract data
    steps = df['step'].values
    gpu_util = df['gpu_utilization'].values
    gpu_mem = df['gpu_memory_used_mb'].values
    sys_mem = df['system_memory_used_mb'].values
    throughput = df['samples_per_sec'].values
    
    # Calculate X-axis range
    max_step = int(steps.max())
    x_min, x_max = 0, max_step
    
    print(f"X-axis range: {x_min} to {x_max}")
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ResNet152 Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: GPU Utilization
    axes[0, 0].plot(steps, gpu_util, 'b-', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('GPU Utilization (%)')
    axes[0, 0].set_title('GPU Utilization Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(0, 100)
    avg_util = gpu_util.mean()
    axes[0, 0].axhline(y=avg_util, color='r', linestyle='--', 
                       label=f'Average: {avg_util:.1f}%', alpha=0.7)
    axes[0, 0].legend()
    
    # Plot 2: GPU Memory
    axes[0, 1].plot(steps, gpu_mem, 'g-', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('GPU Memory (MB)')
    axes[0, 1].set_title('GPU Memory Usage')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(x_min, x_max)
    avg_mem = gpu_mem.mean()
    axes[0, 1].axhline(y=avg_mem, color='r', linestyle='--',
                       label=f'Average: {avg_mem:.0f} MB', alpha=0.7)
    axes[0, 1].legend()
    
    # Plot 3: System Memory (RAM)
    axes[1, 0].plot(steps, sys_mem, 'orange', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('System Memory (MB)')
    axes[1, 0].set_title('System Memory (RAM) Usage')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(x_min, x_max)
    avg_sys_mem = sys_mem.mean()
    axes[1, 0].axhline(y=avg_sys_mem, color='r', linestyle='--',
                       label=f'Average: {avg_sys_mem:.0f} MB', alpha=0.7)
    axes[1, 0].legend()
    
    # Plot 4: Throughput
    axes[1, 1].plot(steps, throughput, 'm-', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Throughput (samples/sec)')
    axes[1, 1].set_title('Training Throughput')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(x_min, x_max)
    avg_throughput = throughput.mean()
    axes[1, 1].axhline(y=avg_throughput, color='r', linestyle='--',
                       label=f'Average: {avg_throughput:.1f} samples/s', alpha=0.7)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  Steps: {len(steps)}")
    print(f"  GPU Util: {avg_util:.1f}%")
    print(f"  GPU Mem: {avg_mem:.0f} MB")
    print(f"  RAM: {avg_sys_mem:.0f} MB")
    print(f"  Throughput: {avg_throughput:.1f} samples/s")


def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) == 1:
        # No arguments - use defaults
        csv_file = "training_stats_backup/stats.csv"
        output_file = "training_stats_backup/training_metrics_l.png"
    elif len(sys.argv) == 2:
        # Only CSV file provided
        csv_file = sys.argv[1]
        output_file = str(Path(csv_file).parent / "training_metrics_l.png")
    elif len(sys.argv) == 3:
        # Both CSV and output file provided
        csv_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Usage:")
        print(f"  {sys.argv[0]} [csv_file] [output_png]")
        print("\nExamples:")
        print(f"  {sys.argv[0]}")
        print(f"  {sys.argv[0]} stats.csv")
        print(f"  {sys.argv[0]} training_stats_backup/stats.csv my_plot.png")
        sys.exit(1)
    
    generate_plots(csv_file, output_file)


if __name__ == "__main__":
    main()