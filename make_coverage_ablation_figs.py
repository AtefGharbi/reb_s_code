import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1F3864"
GOLD = "#B08968"

# --- coverage sweep figure ---
df = pd.read_csv("/home/claude/reb_s_code/results/table3b_coverage_raw.csv")
fig, ax = plt.subplots(figsize=(6, 4.2))
for method, label, marker, color in [("reactive", "Reactive (CUSUM) + MILP", "o", GOLD),
                                       ("rebs", "REB-S (proposed)", "s", NAVY)]:
    sub = df[df.method == method].sort_values("coverage")
    ax.plot(sub.coverage * 100, sub.f1, marker=marker, label=label, color=color, linewidth=2)
ax.set_xlabel("Attack coverage (% of fleet attacked)")
ax.set_ylabel("Detection F1-score")
ax.set_title("F1-score vs. attack coverage (\u03B4_max = 0.06 fixed)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/coverage_sensitivity.png", dpi=200, bbox_inches="tight")
print("saved coverage sensitivity figure")

# --- ablation bar chart ---
df_ab = pd.read_csv("/home/claude/reb_s_code/results/ablation_summary.csv").set_index("method")
order = ["Full REB-S", "No state reconstruction", "No robust departure", "No risk margin/derate", "Detector only (all off)"]
df_ab = df_ab.reindex(order)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
colors_ab = ["#1F3864", "#3E6B9E", "#B5482A", "#B08968", "#8B8B8B"]

axes[0].bar(range(len(order)), df_ab["completion_mean"], yerr=df_ab["completion_std"],
            capsize=4, color=colors_ab, edgecolor="black", linewidth=0.6)
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels(order, rotation=32, ha="right", fontsize=8)
axes[0].set_ylabel("EV completion rate (%)")
axes[0].set_title("(a) Completion rate by ablated component")

axes[1].bar(range(len(order)), df_ab["violations_mean"], color=colors_ab, edgecolor="black", linewidth=0.6)
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels(order, rotation=32, ha="right", fontsize=8)
axes[1].set_ylabel("Constraint violations (/10 EVs)")
axes[1].set_title("(b) Violations by ablated component")

plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/ablation_bars.png", dpi=200, bbox_inches="tight")
print("saved ablation bar chart")
