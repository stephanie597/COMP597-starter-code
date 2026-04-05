#!/usr/bin/env python3
"""
Generate all report plots for COMP597 ResNet152 experiments.

Each figure is saved as a SEPARATE file (PNG + PDF).
All Y-axes start from 0.

Usage:
    python generate_report_plots.py [results_dir] [output_dir]
    
Default:
    results_dir = ./results
    output_dir  = ./report_plots
"""

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUTPUT_DIR  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES  = [128, 64, 32]
RUNS         = [1, 2, 3]
COLORS       = {128: "#2196F3", 64: "#4CAF50", 32: "#FF9800"}
PHASE_COLORS = {"forward": "#4C72B0", "backward": "#DD8452", "optimizer": "#55A868"}

# ── Matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

# ── Helpers ─────────────────────────────────────────────────────────────
def latest_csv(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def read_phases(bs, run):
    f = latest_csv(str(RESULTS_DIR / f"basic_resources_logs/bs{bs}_run{run}/phases_*.csv"))
    if not f: return None
    df = pd.read_csv(f)
    return df[df["step"] > 0]

def read_steps(bs, run):
    f = latest_csv(str(RESULTS_DIR / f"basic_resources_logs/bs{bs}_run{run}/steps_*.csv"))
    if not f: return None
    df = pd.read_csv(f)
    return df[df["step"] > 0]

def read_energy(bs, run):
    f = str(RESULTS_DIR / f"energy_baseline_logs/bs{bs}/run{run}_energy_full_gpu0.csv")
    try:
        return pd.read_csv(f).iloc[0]
    except:
        return None

def extract_times_from_log(log_path, keyword):
    times = {bs: [] for bs in BATCH_SIZES}
    current_bs = None
    with open(log_path) as f:
        for line in f:
            for bs in BATCH_SIZES:
                if f"batch_size={bs}" in line:
                    current_bs = bs
            if keyword in line and current_bs:
                try:
                    t = float(line.split(":")[-1].strip().rstrip("s"))
                    times[current_bs].append(t)
                except:
                    pass
    return times

def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}.png / .pdf saved")

# ══════════════════════════════════════════════════════════════════════
# FIG 1 — End-to-end time: baseline vs instrumented
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 1: End-to-end time baseline vs instrumented...")

time_log   = RESULTS_DIR / "time_baseline.log"
energy_log = RESULTS_DIR / "energy_baseline.log"

baseline_times     = extract_times_from_log(time_log,   "no instrumentation")
instrumented_times = extract_times_from_log(energy_log, "with energy measurement")

fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(len(BATCH_SIZES))
w = 0.35

for i, (label, color, times_dict) in enumerate([
    ("Time Baseline",       "#4C72B0", baseline_times),
    ("Energy Instrumented", "#DD8452", instrumented_times),
]):
    means = [np.mean(times_dict[bs]) for bs in BATCH_SIZES]
    stds  = [np.std(times_dict[bs])  for bs in BATCH_SIZES]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=stds,
           label=label, color=color, capsize=4, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"BS {bs}" for bs in BATCH_SIZES])
ax.set_ylabel("Training Time (s)")
ax.set_title("Fig. 1 — End-to-End Time: Baseline vs. Instrumented")
ax.set_ylim(0)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig1_time_baseline_vs_instrumented")

# ══════════════════════════════════════════════════════════════════════
# FIG 2 — End-to-end energy consumption
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 2: Energy consumption...")

fig, ax = plt.subplots(figsize=(4.5, 3.5))
means, stds = [], []
for bs in BATCH_SIZES:
    vals = [read_energy(bs, r)["energy_consumed"] * 1000 for r in RUNS]
    means.append(np.mean(vals))
    stds.append(np.std(vals))

bars = ax.bar([f"BS {bs}" for bs in BATCH_SIZES], means, yerr=stds,
              color=[COLORS[bs] for bs in BATCH_SIZES], capsize=5, alpha=0.85)
ax.set_ylabel("Energy Consumed (mWh)")
ax.set_title("Fig. 2 — End-to-End Energy Consumption")
ax.set_ylim(0)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{m:.2f}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig2_energy_consumption")

# ══════════════════════════════════════════════════════════════════════
# FIG 3 — Per-phase bar chart
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 3: Per-phase bar chart...")

phases = ["forward", "backward", "optimizer"]
phase_data = {bs: {p: [] for p in phases} for bs in BATCH_SIZES}

for bs in BATCH_SIZES:
    for run in RUNS:
        df = read_phases(bs, run)
        if df is None: continue
        for p in phases:
            vals = df[df["phase"] == p]["duration_sec"].values * 1000
            phase_data[bs][p].extend(vals)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(phases))
w = 0.25

for i, bs in enumerate(BATCH_SIZES):
    means = [np.mean(phase_data[bs][p]) for p in phases]
    stds  = [np.std(phase_data[bs][p])  for p in phases]
    ax.bar(x + (i - 1) * w, means, w, yerr=stds,
           label=f"BS {bs}", color=COLORS[bs], capsize=4, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(["Forward", "Backward", "Optimizer"])
ax.set_ylabel("Duration (ms)")
ax.set_title("Fig. 3 — Average Per-Phase Duration (mean ± std)")
ax.set_ylim(0)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig3_phase_bar_chart")

# ══════════════════════════════════════════════════════════════════════
# FIG 4a/b/c — System utilization timelines (ONE figure per BS)
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 4: Utilization timelines (separate per BS)...")

metrics = [
    ("gpu_util_pct",  "GPU Utilization (%)",   0, 100),
    ("cpu_util_pct",  "CPU Utilization (%)",    0, None),
    ("gpu_mem_mb",    "GPU Memory (MB)",         0, None),
]

bs_labels = {128: "a", 64: "b", 32: "c"}

for bs in BATCH_SIZES:
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    fig.suptitle(f"Fig. 4{bs_labels[bs]} — System Utilization Timelines (BS {bs}, Run 1)",
                 fontsize=11, fontweight="bold")

    df = read_steps(bs, 1)
    if df is not None:
        steps = df["step"].values
        for row, (metric, ylabel, ymin, ymax) in enumerate(metrics):
            ax = axes[row]
            ax.plot(steps, df[metric].values,
                    color=COLORS[bs], linewidth=1.0, alpha=0.85)
            avg = df[metric].mean()
            ax.axhline(avg, color="red", linestyle="--", linewidth=0.8,
                       label=f"Avg: {avg:.1f}", alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=ymin)
            if ymax is not None:
                ax.set_ylim(top=ymax)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    save_fig(fig, f"fig4{bs_labels[bs]}_timelines_bs{bs}")

# ══════════════════════════════════════════════════════════════════════
# FIG 4 combined — all 3 BS side by side (for report)
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 4 combined...")

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey="row")
fig.suptitle("Fig. 4 — System Utilization Timelines (Run 1)",
             fontsize=12, fontweight="bold")

for col, bs in enumerate(BATCH_SIZES):
    df = read_steps(bs, 1)
    if df is None: continue
    steps = df["step"].values
    for row, (metric, ylabel, ymin, ymax) in enumerate(metrics):
        ax = axes[row][col]
        ax.plot(steps, df[metric].values,
                color=COLORS[bs], linewidth=1.0, alpha=0.85)
        avg = df[metric].mean()
        ax.axhline(avg, color="red", linestyle="--", linewidth=0.8,
                   label=f"Avg: {avg:.1f}", alpha=0.8)
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_title(f"BS {bs}", fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 2:
            ax.set_xlabel("Step")
        ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "fig4_timelines_combined")

# ══════════════════════════════════════════════════════════════════════
# FIG 5 — Batch size effects
# ══════════════════════════════════════════════════════════════════════
print("Generating Fig 5: Batch size effects...")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
fig.suptitle("Fig. 5 — Batch Size Effects", fontsize=12, fontweight="bold")

# (a) Throughput
ax = axes[0]
tput_means, tput_stds = [], []
for bs in BATCH_SIZES:
    vals = []
    for run in RUNS:
        df = read_steps(bs, run)
        if df is not None:
            vals.append(df["throughput_samples_sec"].mean())
    tput_means.append(np.mean(vals))
    tput_stds.append(np.std(vals))
ax.errorbar(BATCH_SIZES, tput_means, yerr=tput_stds,
            fmt="o-", color="#4C72B0", linewidth=2, markersize=7, capsize=5)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Throughput (samples/s)")
ax.set_title("(a) Throughput")
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(0)
ax.grid(alpha=0.3)

# (b) Energy
ax = axes[1]
e_means, e_stds = [], []
for bs in BATCH_SIZES:
    vals = [read_energy(bs, r)["energy_consumed"] * 1000 for r in RUNS]
    e_means.append(np.mean(vals))
    e_stds.append(np.std(vals))
ax.errorbar(BATCH_SIZES, e_means, yerr=e_stds,
            fmt="o-", color="#DD8452", linewidth=2, markersize=7, capsize=5)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Energy (mWh)")
ax.set_title("(b) Energy Consumption")
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(0)
ax.grid(alpha=0.3)

# (c) Phase duration
ax = axes[2]
for phase, color in PHASE_COLORS.items():
    p_means = []
    for bs in BATCH_SIZES:
        all_vals = []
        for run in RUNS:
            df = read_phases(bs, run)
            if df is not None:
                all_vals.extend(
                    df[df["phase"] == phase]["duration_sec"].values * 1000
                )
        p_means.append(np.mean(all_vals))
    ax.plot(BATCH_SIZES, p_means, "o-", color=color,
            linewidth=2, markersize=7, label=phase.capitalize())
ax.set_xlabel("Batch Size")
ax.set_ylabel("Avg Duration (ms)")
ax.set_title("(c) Phase Duration vs Batch Size")
ax.set_xticks(BATCH_SIZES)
ax.set_ylim(0)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
save_fig(fig, "fig5_batch_size_effects")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY TABLE ===")
print(f"{'BS':>4} | {'Time_base(s)':>13} | {'Time_inst(s)':>13} | {'Energy(mWh)':>12} | {'Throughput':>12} | {'Fwd(ms)':>8} | {'Bwd(ms)':>8} | {'Opt(ms)':>8}")
print("-" * 100)
for bs in BATCH_SIZES:
    t_base = np.mean(baseline_times[bs])
    t_inst = np.mean(instrumented_times[bs])
    energy = np.mean([read_energy(bs, r)["energy_consumed"] * 1000 for r in RUNS])
    tput   = np.mean([read_steps(bs, r)["throughput_samples_sec"].mean() for r in RUNS])
    fwd    = np.mean([v for run in RUNS
                      for v in read_phases(bs, run)[read_phases(bs, run)["phase"] == "forward"]["duration_sec"].values * 1000])
    bwd    = np.mean([v for run in RUNS
                      for v in read_phases(bs, run)[read_phases(bs, run)["phase"] == "backward"]["duration_sec"].values * 1000])
    opt    = np.mean([v for run in RUNS
                      for v in read_phases(bs, run)[read_phases(bs, run)["phase"] == "optimizer"]["duration_sec"].values * 1000])
    print(f"{bs:>4} | {t_base:>13.3f} | {t_inst:>13.3f} | {energy:>12.4f} | {tput:>12.2f} | {fwd:>8.2f} | {bwd:>8.2f} | {opt:>8.2f}")

print(f"\nAll plots saved to: {OUTPUT_DIR}/")