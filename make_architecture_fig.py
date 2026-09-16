import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

NAVY = "#1F3864"
GOLD = "#B08968"
RED = "#B5482A"
GREY = "#616161"
LGREY = "#EDEDED"

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=NAVY, tc="black", fs=10, lw=1.6, bold=False, zorder=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             color=tc, weight="bold" if bold else "normal", zorder=zorder + 1, wrap=True)
    return (x, y, w, h)


def arrow(p0, p1, color=NAVY, style="-|>", lw=1.8, ls="-", rad=0.0, z=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                         color=color, linewidth=lw, linestyle=ls,
                         connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(a)


# --- boxes ---
b_assets = box(0.3, 4.5, 2.2, 1.0, "Physical EV / BESS\nassets", fc=LGREY, ec=GREY, bold=True)
b_bms = box(0.3, 3.0, 2.2, 1.0, "BMS telemetry\n(SoC, departure time)", fc="white", ec=NAVY)
b_attacker = box(3.1, 3.0, 2.0, 1.0, "Adversary\n(SoC spoof / dep.-time\ninjection, eq. 9-10)", fc="#FBEAEA", ec=RED, tc=RED, bold=True)
b_reported = box(0.3, 1.5, 2.2, 1.0, "Reported (possibly\nattacked) stream", fc="white", ec=NAVY)
b_detector = box(3.1, 0.3, 2.6, 1.0, "Transformer detector\n(Sec. 5.2-5.3)", fc="#E8EEF7", ec=NAVY, bold=True)
b_risk = box(6.2, 0.3, 1.7, 1.0, "Risk score\nr(t) \u2208 [0,1]", fc="white", ec=NAVY)
b_milp = box(6.2, 1.9, 2.9, 1.2, "Risk-adaptive MILP\nscheduler (Sec. 3, 5.4-5.5)", fc="#E8EEF7", ec=NAVY, bold=True)
b_cmd = box(6.2, 3.6, 2.9, 1.0, "Charge/discharge\ncommands", fc="white", ec=NAVY)

# --- arrows: main loop ---
arrow((1.4, 4.5), (1.4, 4.0))                       # assets -> bms
arrow((2.5, 3.5), (3.1, 3.5), color=RED, ls="--")   # attacker injects into channel
arrow((1.4, 3.0), (1.4, 2.5))                       # bms -> reported
arrow((2.5, 2.0), (3.1, 1.0), rad=-0.15)             # reported -> detector
arrow((1.4, 1.5), (1.4, 0.8))                        # reported straight down
arrow((1.4, 0.8), (6.2, 2.3), rad=0.12)              # reported -> MILP directly (trusted input path)
arrow((5.7, 0.8), (6.2, 0.8))                        # detector -> risk score
arrow((7.05, 1.3), (7.65, 1.9), rad=-0.1)            # risk score -> MILP (margin, eq 11)
arrow((7.65, 3.1), (7.65, 3.6))                      # MILP -> commands
arrow((6.2, 4.1), (2.5, 4.9), rad=0.18)              # commands back to physical assets (closes loop)

ax.text(4.6, 3.75, "compromised\nchannel", fontsize=8, color=RED, ha="center", style="italic")
ax.text(3.75, 1.62, "reported stream", fontsize=7.5, color=GREY, ha="center", style="italic", rotation=-15)
ax.text(1.85, 0.92, "trusted (uncorrected) path used\nby unaware / reactive baselines", fontsize=7, color=GREY, ha="center", style="italic")
ax.text(7.35, 1.62, "margin\ntightening", fontsize=7.5, color=NAVY, ha="center", style="italic")
ax.text(4.3, 4.55, "commands applied to physical assets\n(closes the control loop)", fontsize=7.5, color=GREY, ha="center", style="italic")

ax.text(5.0, 5.7, "REB-S closed-loop architecture", ha="center", fontsize=13, weight="bold", color=NAVY)

plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/architecture.png", dpi=220, bbox_inches="tight")
print("saved architecture diagram")
