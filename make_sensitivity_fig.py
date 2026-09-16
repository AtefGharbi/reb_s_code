import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/reb_s_code/results/table3_sensitivity_raw.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for method, label, marker, color in [('reactive', 'Reactive (CUSUM) + MILP', 'o', '#B08968'),
                                       ('rebs', 'REB-S (proposed)', 's', '#1F3864')]:
    sub = df[df.method == method].sort_values('delta_max')
    axes[0].plot(sub.delta_max, sub.mean_degr_cost, marker=marker, label=label, color=color, linewidth=2)
    axes[1].plot(sub.delta_max, sub.recall, marker=marker, label=label, color=color, linewidth=2)

axes[0].set_xlabel(r'Attack intensity $\delta_{max}$ (fraction SoC)')
axes[0].set_ylabel('Mean BESS degradation cost (SAR)')
axes[0].set_title('(a) Degradation cost vs. attack intensity')
axes[0].grid(alpha=0.3)

axes[1].set_xlabel(r'Attack intensity $\delta_{max}$ (fraction SoC)')
axes[1].set_ylabel('Detection recall')
axes[1].set_title('(b) Detection recall vs. attack intensity')
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(alpha=0.3)
axes[1].legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/reb_s_code/figures/sensitivity_intensity.png', dpi=200, bbox_inches='tight')
print('saved sensitivity_intensity.png')
