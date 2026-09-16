import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/reb_s_code/results/table2_summary.csv").set_index("method")
methods = ["nominal", "unaware", "reactive", "rebs"]
labels_disp = {"nominal": "Nominal", "unaware": "Attack-unaware", "reactive": "Reactive (CUSUM)", "rebs": "REB-S"}
colors = {"nominal": "#8DA0C4", "unaware": "#B5482A", "reactive": "#B08968", "rebs": "#1F3864"}

# Metrics oriented so that LARGER = BETTER after transformation.
metrics = {
    "Low cost": -df["total_cost_mean"],
    "Peak reduction": df["peak_reduction_mean"],
    "Completion rate": df["completion_mean"],
    "Battery health\n(1/EFC)": 1.0 / df["efc_mean"],
    "V2G revenue": df["v2g_mean"],
    "Reliability\n(1 - violations)": 10 - df["violations_mean"],
}
M = pd.DataFrame(metrics)

# min-max normalize each column to [0.15, 1] across the 4 methods (keep a visible floor)
Mn = M.copy()
for c in Mn.columns:
    lo, hi = M[c].min(), M[c].max()
    Mn[c] = 0.15 + 0.85 * (M[c] - lo) / (hi - lo if hi > lo else 1.0)

cats = list(Mn.columns)
n = len(cats)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["", "", "", ""], fontsize=0)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=9.5)

for m in methods:
    vals = Mn.loc[m].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=colors[m], linewidth=2.2, label=labels_disp[m])
    ax.fill(angles, vals, color=colors[m], alpha=0.06)

ax.set_title("Multi-metric comparison (outer = better; each axis min-max\nnormalized across the four methods)", fontsize=10.5, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)

plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/radar_comparison.png", dpi=200, bbox_inches="tight")
print("saved radar chart")
print(Mn.round(2))
