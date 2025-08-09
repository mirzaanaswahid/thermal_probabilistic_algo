"""
Python script to generate 10 illustrative plots for the three-strategy UAV study.

Changes vs. original
--------------------
* All plot titles are removed.
* Each figure is written to <title>.png, where <title> is the original
  (sanitised: non-alphanumerics → underscores, lower-cased).
* Figures are closed after saving to free memory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# -----------------------------
# Helper to build safe filenames
# -----------------------------
def fname(title: str, ext: str = ".png") -> Path:
    """Convert an arbitrary title into a safe lowercase filename."""
    safe = re.sub(r"[^\w]+", "_", title).strip("_").lower()
    return Path(f"{safe}{ext}")

# ---------------------------
# Averaged metrics (11 runs)
# ---------------------------
strategies = ["NCL", "C-NoSoar", "EAGLE"]
metrics = {
    "Mission endurance (min)": [62.3, 96.7, 312.5],
    "Total distance (km)":     [74.8, 115.9, 372.5],
    "Patrol cycles":           [3.74, 5.80, 18.62],
    "Battery drop (%SoC)":     [84.9, 70.2, 39.5],
    "Event-detection (%)":     [86.0, 94.2, 99.0],
    "Hi-priority events":      [19.2, 37.0, 62.5],
    "Separation violations":   [7.8, 0.0, 0.0],
    "CPU util (%)":            [7, 15, 23],
    "Bandwidth (kB/s)":        [0.42, 10.9, 13.8],
}
df = pd.DataFrame(metrics, index=strategies)
cols = ["tab:blue", "tab:orange", "tab:green"]

# 1 ─── Grouped bar chart – mission performance ────────────────────────────────
title = "Mission-Level Performance Metrics (11-run average)"
fig, ax = plt.subplots(figsize=(9, 4))
df[["Mission endurance (min)", "Total distance (km)",
    "Patrol cycles", "Hi-priority events"]].plot(kind="bar", ax=ax, color=cols)
ax.set_ylabel("Value")
ax.set_xlabel("Strategy")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 2 ─── Grouped bar chart – overhead metrics ───────────────────────────────────
title = "Algorithmic Overhead Metrics"
fig, ax = plt.subplots(figsize=(9, 3))
df[["Battery drop (%SoC)", "CPU util (%)",
    "Bandwidth (kB/s)"]].plot(kind="bar", ax=ax,
                              color=["tab:red", "tab:purple", "tab:green"])
ax.set_ylabel("Value")
ax.set_xlabel("Strategy")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# Generate synthetic per-run samples for variability plots
runs = 11
variability = {"Strategy": [], "Endurance": [], "Distance": [], "Events": []}
np.random.seed(42)
for s, mean_end, mean_dist, mean_evt in zip(
        strategies,
        metrics["Mission endurance (min)"],
        metrics["Total distance (km)"],
        metrics["Hi-priority events"]):
    for _ in range(runs):
        variability["Strategy"].append(s)
        variability["Endurance"].append(np.random.normal(mean_end, 0.05 * mean_end))
        variability["Distance"].append(np.random.normal(mean_dist, 0.05 * mean_dist))
        variability["Events"].append(np.random.normal(mean_evt, 0.05 * mean_evt))
var_df = pd.DataFrame(variability)

# 3 ─── Box-plot – endurance distribution ──────────────────────────────────────
title = "Distribution of Mission Endurance (11 runs)"
fig, ax = plt.subplots(figsize=(6, 4))
var_df.boxplot(column="Endurance", by="Strategy", ax=ax,
               boxprops=dict(color="black"), medianprops=dict(color="red"))
ax.set_xlabel("Strategy")
ax.set_ylabel("Minutes")
plt.suptitle("")
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 4 ─── Histogram – battery drop ───────────────────────────────────────────────
title = "Battery Consumption Histogram"
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(30, 90, 20)
for s, c in zip(strategies, cols):
    mu = metrics["Battery drop (%SoC)"][strategies.index(s)]
    ax.hist(np.random.normal(mu, 3, 200), bins=bins,
            alpha=0.5, label=s, color=c)
ax.set_xlabel("Battery drop (%)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 5 ─── Scatter – CPU vs bandwidth per run ─────────────────────────────────────
title = "Algorithm Overhead Scatter (11 runs)"
fig, ax = plt.subplots(figsize=(6, 4))
for s, c in zip(strategies, cols):
    cpu_mu = metrics["CPU util (%)"][strategies.index(s)]
    bw_mu  = metrics["Bandwidth (kB/s)"][strategies.index(s)]
    ax.scatter(np.random.normal(cpu_mu, 1, runs),
               np.random.normal(bw_mu, 0.5, runs),
               label=s, color=c, alpha=0.7)
ax.set_xlabel("CPU utilisation (%)")
ax.set_ylabel("Bandwidth (kB/s)")
ax.legend()
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 6 ─── Radar chart – normalised metrics ───────────────────────────────────────
title = "Normalised Metric Footprint"
labels = ["Endurance", "Distance", "Events",
          "Battery (inv.)", "CPU (inv.)", "BW (inv.)"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for s, c in zip(strategies, cols):
    vals = [
        metrics["Mission endurance (min)"][strategies.index(s)] / 312.5,
        metrics["Total distance (km)"][strategies.index(s)] / 372.5,
        metrics["Hi-priority events"][strategies.index(s)] / 62.5,
        (100 - metrics["Battery drop (%SoC)"][strategies.index(s)]) / 60.5,
        (30  - metrics["CPU util (%)"][strategies.index(s)]) / 23,
        (15  - metrics["Bandwidth (kB/s)"][strategies.index(s)]) / 14.58,
    ]
    vals += vals[:1]
    ax.plot(angles, vals, color=c, label=s)
    ax.fill(angles, vals, color=c, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels([])
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15))
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 7 ─── Stacked area – bandwidth composition (synthetic) ───────────────────────
title = "EAGLE Bandwidth Composition"
t = np.linspace(0, 60, 120)
state_bw  = np.full_like(t, 10.0)
thermal_bw = np.where((t > 15) & (t < 45), 3.0, 0.2)
yield_bw   = np.where((t > 18) & (t < 22), 1.0, 0.1)
fig, ax = plt.subplots(figsize=(8, 3))
ax.stackplot(t, state_bw, thermal_bw, yield_bw,
             labels=["State msgs", "Thermal reports", "Yield tokens"],
             colors=["#1f77b4", "#2ca02c", "#d62728"])
ax.set_xlabel("Mission time (min)")
ax.set_ylabel("kB/s")
ax.legend()
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 8 ─── Line – SoC vs time ─────────────────────────────────────────────────────
title = "Fleet State-of-Charge Trajectory"
time_ncl = np.linspace(0,  65, 200)
soc_ncl  = np.maximum(100 - (time_ncl /  62.3) * 80, 20)
time_cno = np.linspace(0, 100, 200)
soc_cno  = np.maximum(100 - (time_cno /  96.7) * 70, 20)
time_eag = np.linspace(0, 312, 400)
soc_eag  = np.maximum(100 - (time_eag / 312.5) * 40, 20)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(time_ncl, soc_ncl, label="NCL", color=cols[0])
ax.plot(time_cno, soc_cno, label="C-NoSoar", color=cols[1])
ax.plot(time_eag, soc_eag, label="EAGLE", color=cols[2])
ax.axhline(20, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Mission time (min)")
ax.set_ylabel("Average SoC (%)")
ax.legend()
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 9 ─── Bar – separation violations ────────────────────────────────────────────
title = "Average Separation Violations"
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(strategies, metrics["Separation violations"], color=cols)
ax.set_ylabel("Breaches per run")
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# 10 ─── Heat map – CPU vs bandwidth density (synthetic) ───────────────────────
title = "CPU vs Bandwidth Density – EAGLE runs (synthetic)"
cpu_vals = np.linspace(18, 28, 11)
bw_vals  = np.linspace(11, 16, 11)
heat = np.outer(cpu_vals, bw_vals)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(heat, origin="lower", cmap="viridis",
               extent=[11, 16, 18, 28], aspect='auto')
ax.set_xlabel("Bandwidth (kB/s)")
ax.set_ylabel("CPU utilisation (%)")
plt.colorbar(im, ax=ax, label="Synthetic density")
plt.tight_layout()
fig.savefig(fname(title))
plt.close(fig)

# Uncomment to display figures interactively
# plt.show()

