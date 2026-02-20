#!/usr/bin/env python3
"""
Generate carbon emissions plots from CodeCarbon CSV files

This script reads CodeCarbon step emissions CSV and generates plots.

Usage:
    python generate_carbon_plots.py [csv_file] [output_png]
    
Examples:
    python generate_carbon_plots.py
    python generate_carbon_plots.py run_1_carbon_steps_gpu0.csv
    python generate_carbon_plots.py run_1_carbon_steps_gpu0.csv carbon_plot.png
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def generate_carbon_plots(csv_file="training_stats_backup/run_1_carbon_steps_gpu0.csv", 
                          output_file="training_stats_backup/carbon_emissions_plot.png"):
    """Generate carbon emissions plots from CodeCarbon CSV file.
    
    Args:
        csv_file: Path to the CodeCarbon steps CSV file
        output_file: Path where to save the PNG file
    """
    
    print(f"Reading carbon data from: {csv_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        print("\nTry specifying the correct path:")
        print(f"  python {sys.argv[0]} /path/to/run_1_carbon_steps_gpu0.csv")
        sys.exit(1)
    
    print(f"Found {len(df)} data points")
    
    # Extract data
    steps = list(range(1, len(df) + 1))
    emissions = df['emissions'].values * 1000  # Convert to grams CO2
    energy = df['energy_consumed'].values * 1000  # Convert to Wh
    
    # Calculate X-axis range
    max_step = max(steps) if steps else len(df)
    x_min, x_max = 0, max_step
    
    print(f"Step range: 1 to {max_step}")
    print(f"X-axis range: {x_min} to {x_max}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Carbon Emissions Tracking', fontsize=16, fontweight='bold')
    
    # Plot 1: CO2 Emissions
    axes[0].plot(steps, emissions, 'g-', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('CO2 Emissions (g CO2eq)')
    axes[0].set_title('Carbon Emissions per Step')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(x_min, x_max)
    # Auto-adjust Y-axis for better visibility
    
    if len(emissions) > 0:
        total_emissions = emissions.sum()
        avg_emissions = emissions.mean()
        axes[0].axhline(y=avg_emissions, color='r', linestyle='--',
                       label=f'Avg: {avg_emissions:.4f} g/step', alpha=0.7)
        axes[0].text(0.98, 0.98, f'Total: {total_emissions:.2f} g CO2eq',
                    transform=axes[0].transAxes,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0].legend()
    
    # Plot 2: Energy Consumption
    axes[1].plot(steps, energy, 'b-', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Energy Consumed (Wh)')
    axes[1].set_title('Energy Consumption per Step')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(x_min, x_max)
    # Auto-adjust Y-axis for better visibility
    
    if len(energy) > 0:
        total_energy = energy.sum()
        avg_energy = energy.mean()
        axes[1].axhline(y=avg_energy, color='r', linestyle='--',
                       label=f'Avg: {avg_energy:.4f} Wh/step', alpha=0.7)
        axes[1].text(0.98, 0.98, f'Total: {total_energy:.2f} Wh',
                    transform=axes[1].transAxes,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plot saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  Steps: {len(steps)}")
    print(f"  Total CO2: {total_emissions:.4f} g CO2eq")
    print(f"  Total Energy: {total_energy:.4f} Wh")
    print(f"  Avg CO2/step: {avg_emissions:.6f} g")
    print(f"  Avg Energy/step: {avg_energy:.6f} Wh")


def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) == 1:
        # No arguments - use defaults
        csv_file = "training_stats_backup/run_1_carbon_steps_gpu0.csv"
        output_file = "training_stats_backup/carbon_emissions_plot.png"
    elif len(sys.argv) == 2:
        # Only CSV file provided
        csv_file = sys.argv[1]
        output_file = str(Path(csv_file).parent / "carbon_emissions_plot.png")
    elif len(sys.argv) == 3:
        # Both CSV and output file provided
        csv_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Usage:")
        print(f"  {sys.argv[0]} [csv_file] [output_png]")
        print("\nExamples:")
        print(f"  {sys.argv[0]}")
        print(f"  {sys.argv[0]} run_1_carbon_steps_gpu0.csv")
        print(f"  {sys.argv[0]} run_1_carbon_steps_gpu0.csv my_carbon_plot.png")
        sys.exit(1)
    
    generate_carbon_plots(csv_file, output_file)


if __name__ == "__main__":
    main()