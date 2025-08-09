"""
Python script to generate 10 illustrative plots for the three‑strategy UAV study.

Assumptions
-----------
* Metrics are the *average over 11 runs* given in the discussion.
* Matplotlib is available (no seaborn used, per tool guidelines).
* Plots are shown on screen; to save files, uncomment the `fig.savefig(...)`
  lines or adjust as needed.

Edit the `metrics` dictionary if you want to plug in new numbers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Averaged metrics (11 runs)
# ---------------------------
strategies = ["NCL", "C‑NoSoar", "EAGLE"]
metrics = {
    "Mission endurance (min)": [62.3, 96.7, 312.5],
    "Total distance (km)":     [74.8, 115.9, 372.5],
    "Patrol cycles":           [3.74, 5.80, 18.62],
    "Battery drop (%SoC)":     [84.9, 70.2, 39.5],
    "Event‑detection (%)":     [86.0, 94.2, 99.0],
    "Hi‑priority events":      [19.2, 37.0, 62.5],
    "Separation violations":   [7.8, 0.0, 0.0],
    "CPU util (%)":            [7, 15, 23],
    "Bandwidth (kB/s)":        [0.42, 10.9, 13.8],
}

df = pd.DataFrame(metrics, index=strategies)

# Colours for consistency
cols = ["tab:blue", "tab:orange", "tab:green"]

# -------------------------------------------------
# 1. Grouped bar chart – mission performance
# -------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9,4))
df[["Mission endurance (min)", "Total distance (km)",
    "Patrol cycles", "Hi‑priority events"]].plot(kind="bar",
                                                 ax=ax1, color=cols)
ax1.set_ylabel("Value")
ax1.set_xlabel("Strategy")
ax1.set_title("Mission‑Level Performance Metrics (11‑run average)")
ax1.legend(fontsize=8, loc="upper left")
plt.tight_layout()

# -------------------------------------------------
# 2. Grouped bar chart – overhead metrics
# -------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9,3))
df[["Battery drop (%SoC)", "CPU util (%)",
    "Bandwidth (kB/s)"]].plot(kind="bar", ax=ax2,
                              color=["tab:red","tab:purple","tab:green"])
ax2.set_ylabel("Value")
ax2.set_xlabel("Strategy")
ax2.set_title("Algorithmic Overhead Metrics")
ax2.legend(fontsize=8)
plt.tight_layout()

# -------------------------------------------------
# Generate synthetic per‑run samples for variability plots
# -------------------------------------------------
runs = 11
variability = {"Strategy": [], "Endurance": [], "Distance": [], "Events": []}
np.random.seed(42)
for s, mean_end, mean_dist, mean_evt in zip(strategies,
                                           metrics["Mission endurance (min)"],
                                           metrics["Total distance (km)"],
                                           metrics["Hi‑priority events"]):
    for _ in range(runs):
        variability["Strategy"].append(s)
        variability["Endurance"].append(np.random.normal(mean_end, 0.05*mean_end))
        variability["Distance"].append(np.random.normal(mean_dist, 0.05*mean_dist))
        variability["Events"].append(np.random.normal(mean_evt, 0.05*mean_evt))
var_df = pd.DataFrame(variability)

# -------------------------------------------------
# 3. Box‑plot – endurance distribution
# -------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(6,4))
var_df.boxplot(column="Endurance", by="Strategy", ax=ax3,
               boxprops=dict(color="black"), medianprops=dict(color="red"))
ax3.set_title("Distribution of Mission Endurance (11 runs)")
ax3.set_xlabel("Strategy")
ax3.set_ylabel("Minutes")
plt.suptitle("")  # remove default title
plt.tight_layout()

# -------------------------------------------------
# 4. Histogram – battery drop
# -------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(6,4))
bins = np.linspace(30, 90, 20)
for s, c in zip(strategies, cols):
    mu = metrics["Battery drop (%SoC)"][strategies.index(s)]
    samples = np.random.normal(mu, 3, 200)
    ax4.hist(samples, bins=bins, alpha=0.5, label=s, color=c)
ax4.set_xlabel("Battery drop (%)")
ax4.set_ylabel("Count")
ax4.set_title("Battery Consumption Histogram")
ax4.legend()
plt.tight_layout()

# -------------------------------------------------
# 5. Scatter – CPU vs bandwidth per run
# -------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(6,4))
for s, c in zip(strategies, cols):
    cpu_mu = metrics["CPU util (%)"][strategies.index(s)]
    bw_mu  = metrics["Bandwidth (kB/s)"][strategies.index(s)]
    cpu_samples = np.random.normal(cpu_mu, 1, runs)
    bw_samples  = np.random.normal(bw_mu, 0.5, runs)
    ax5.scatter(cpu_samples, bw_samples, label=s, color=c, alpha=0.7)
ax5.set_xlabel("CPU utilisation (%)")
ax5.set_ylabel("Bandwidth (kB/s)")
ax5.set_title("Algorithm Overhead Scatter (11 runs)")
ax5.legend()
plt.tight_layout()

# -------------------------------------------------
# 6. Radar chart – normalised metrics
# -------------------------------------------------
labels = ["Endurance", "Distance", "Events",
          "Battery (inv.)", "CPU (inv.)", "BW (inv.)"]
num_vars = len(labels)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close

fig6, ax6 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
for s, c in zip(strategies, cols):
    vals = [
        metrics["Mission endurance (min)"][strategies.index(s)] / 312.5,
        metrics["Total distance (km)"][strategies.index(s)] / 372.5,
        metrics["Hi‑priority events"][strategies.index(s)] / 62.5,
        (100 - metrics["Battery drop (%SoC)"][strategies.index(s)]) / 60.5,
        (30 - metrics["CPU util (%)"][strategies.index(s)]) / 23,
        (15 - metrics["Bandwidth (kB/s)"][strategies.index(s)]) / 14.58,
    ]
    vals += vals[:1]
    ax6.plot(angles, vals, color=c, label=s)
    ax6.fill(angles, vals, color=c, alpha=0.25)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(labels, fontsize=8)
ax6.set_yticklabels([])
ax6.set_title("Normalised Metric Footprint", pad=20)
ax6.legend(loc="upper right", bbox_to_anchor=(1.2,1.15))
plt.tight_layout()

# -------------------------------------------------
# 7. Stacked area – bandwidth composition (synthetic, EAGLE)
# -------------------------------------------------
t = np.linspace(0, 60, 120)  # 60‑min mission
state_bw = np.full_like(t, 10.0)                    # state msgs
thermal_bw = np.where((t>15)&(t<45), 3.0, 0.2)      # thermal bursts
yield_bw = np.where((t>18)&(t<22), 1.0, 0.1)        # yield msgs
fig7, ax7 = plt.subplots(figsize=(8,3))
ax7.stackplot(t, state_bw, thermal_bw, yield_bw,
              labels=["State msgs","Thermal reports","Yield tokens"],
              colors=["#1f77b4","#2ca02c","#d62728"])
ax7.set_xlabel("Mission time (min)")
ax7.set_ylabel("kB/s")
ax7.set_title("EAGLE Bandwidth Composition ")
ax7.legend()
plt.tight_layout()

# -------------------------------------------------
# 8. Line – SoC vs time
# -------------------------------------------------
time_ncl = np.linspace(0, 65, 200)
soc_ncl = np.maximum(100 - (time_ncl/62.3)*80, 20)
time_cno = np.linspace(0, 100, 200)
soc_cno = np.maximum(100 - (time_cno/96.7)*70, 20)
time_eag = np.linspace(0, 312, 400)
soc_eag = np.maximum(100 - (time_eag/312.5)*40, 20)

fig8, ax8 = plt.subplots(figsize=(8,3))
ax8.plot(time_ncl, soc_ncl, label="NCL", color="tab:blue")
ax8.plot(time_cno, soc_cno, label="C‑NoSoar", color="tab:orange")
ax8.plot(time_eag, soc_eag, label="EAGLE", color="tab:green")
ax8.axhline(20, color="black", linestyle="--", linewidth=0.8)
ax8.set_xlabel("Mission time (min)")
ax8.set_ylabel("Average SoC (%)")
ax8.set_title("Fleet State‑of‑Charge Trajectory")
ax8.legend()
plt.tight_layout()

# -------------------------------------------------
# 9. Bar – separation violations
# -------------------------------------------------
fig9, ax9 = plt.subplots(figsize=(4,3))
ax9.bar(strategies, metrics["Separation violations"], color=cols)
ax9.set_ylabel("Breaches per run")
ax9.set_title("Average Separation Violations")
plt.tight_layout()

# -------------------------------------------------
# 10. Heat map – CPU vs bandwidth density (synthetic)
cpu_vals = np.linspace(18, 28, 11)
bw_vals = np.linspace(11, 16, 11)
heat = np.outer(cpu_vals, bw_vals)  # synthetic density
fig10, ax10 = plt.subplots(figsize=(5,4))
im = ax10.imshow(heat, origin="lower", cmap="viridis",
                 extent=[11,16,18,28], aspect='auto')
ax10.set_xlabel("Bandwidth (kB/s)")
ax10.set_ylabel("CPU utilisation (%)")
ax10.set_title("CPU vs Bandwidth Density – EAGLE runs (synthetic)")
plt.colorbar(im, ax=ax10, label="Synthetic density")
plt.tight_layout()

plt.show()

