import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1F3864"

# --- SoC-spoofing-only ablation ---
df = pd.read_csv("/home/claude/reb_s_code/results/soc_only_ablation_summary.csv").set_index("method")
order = ["Unaware (baseline)", "Margin only (no reconstruction)", "Reconstruction only (no margin)", "Margin + reconstruction (full REB-S)"]
df = df.reindex(order)
short_labels = ["Unaware", "Margin only", "Reconstruction\nonly", "Margin +\nreconstruction"]
colors = ["#B5482A", "#3E6B9E", "#B08968", "#1F3864"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.bar(range(len(order)), df["completion_mean"], yerr=df["completion_std"], capsize=4,
       color=colors, edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("EV completion rate (%)")
ax.set_ylim(80, 105)
ax.set_title("SoC-spoofing-only attack (no departure-time\nmanipulation): isolating eq. (13)'s contribution")
plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/soc_only_ablation.png", dpi=200, bbox_inches="tight")
print("saved soc_only_ablation.png")

# --- tau_cap sweep ---
df2 = pd.read_csv("/home/claude/reb_s_code/results/tau_cap_sweep_summary.csv")
fig2, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].errorbar(df2.tau_cap_hours, df2.completion_mean, yerr=df2.completion_std,
                  marker="o", color=NAVY, linewidth=2, capsize=4)
axes[0].set_xlabel(r"$\tau_{cap}$ (hours of trusted departure extension)")
axes[0].set_ylabel("EV completion rate (%)")
axes[0].set_title("(a) Reliability vs. trust bound")
axes[0].grid(alpha=0.3)

axes[1].plot(df2.tau_cap_hours, df2.violations_mean, marker="s", color="#B5482A", linewidth=2)
axes[1].set_xlabel(r"$\tau_{cap}$ (hours of trusted departure extension)")
axes[1].set_ylabel("Constraint violations (/10 EVs)")
axes[1].set_title("(b) Violations vs. trust bound")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/tau_cap_sweep.png", dpi=200, bbox_inches="tight")
print("saved tau_cap_sweep.png")
