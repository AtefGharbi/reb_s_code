import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/reb_s_code/results/table2_summary.csv").set_index("method")
methods = ["nominal", "unaware", "reactive", "rebs"]
labels_disp = {"nominal": "Nominal", "unaware": "Unaware", "reactive": "Reactive\n(CUSUM)", "rebs": "REB-S"}
colors = {"nominal": "#8DA0C4", "unaware": "#B5482A", "reactive": "#B08968", "rebs": "#1F3864"}

panels = [
    ("completion_mean", "completion_std", "EV completion rate (%)"),
    ("total_cost_mean", "total_cost_std", "Total operating cost (SAR)"),
    ("efc_mean", "efc_std", "BESS degradation (EFC)"),
    ("violations_mean", "violations_std", "Constraint violations (/10 EVs)"),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
x = np.arange(len(methods))
for ax, (mcol, scol, title) in zip(axes, panels):
    vals = [df.loc[m, mcol] for m in methods]
    errs = [df.loc[m, scol] for m in methods]
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=[colors[m] for m in methods],
                   edgecolor="black", linewidth=0.6, error_kw=dict(elinewidth=1.1))
    ax.set_xticks(x); ax.set_xticklabels([labels_disp[m] for m in methods], fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.axhline(0, color="black", lw=0.6)

fig.suptitle("Table 3 metrics as mean \u00B1 1 s.d. across 20 seeds (moderate attack)", fontsize=11, y=1.04)
plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/grouped_bar_ci.png", dpi=200, bbox_inches="tight")
print("saved grouped bar chart with error bars")
