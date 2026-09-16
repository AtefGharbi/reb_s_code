"""
Produces a rich case-study figure for one representative attacked EV:
(a) true vs reported SoC with the attack window shaded and the two
    re-optimization checkpoints marked,
(b) the physics-based residual driving both detectors,
(c) a continuous Transformer risk score (evaluated at every step for
    visualization -- the operational system only queries it at checkpoints,
    but sliding the trained model across the whole trace shows its
    behavior continuously) with the CUSUM alarm state overlaid.

All data below comes from an actual run of simulate.run_day; nothing is
illustrative/hand-drawn.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import ETA_CH, ETA_DIS, DT_HOURS
from detector import TransformerDetector, WINDOW, risk_scores
from simulate import run_day, _feature_window
from cusum import CusumDetector

NAVY = "#1F3864"
GOLD = "#B08968"
RED = "#B5482A"

model = TransformerDetector()
model.load_state_dict(torch.load("trained_detector.pt"))
model.eval()

attack_cfg = dict(delta_max=0.08, ddelta_max=0.02, coverage=0.5, duration=40, tau_max=4)
SEED = 7
res = run_day("rebs", 10, 2, attack_cfg, seed=SEED, torch_model=model)

evs = res["evs"]
atk = res["attack"]
true_soc_ev = res["true_soc_ev"]
committed = res["committed"]

# pick the EV with the largest attack coverage window for a clear illustration
coverage_per_ev = atk["ev_attacked_gt"].sum(axis=1)
target = int(np.argmax(coverage_per_ev))
ev = [e for e in evs if e["id"] == target][0]
lo, hi = ev["arrival"], ev["departure"]

t_range = np.arange(lo, hi)
true_soc = np.array([true_soc_ev[target].get(t, np.nan) for t in t_range])
bias = atk["ev_bias"][target, lo:hi]
rng = np.random.default_rng(1000 + SEED)
noise = rng.normal(0, 0.004, size=len(t_range))
reported = true_soc + bias + noise
attacked_mask = atk["ev_attacked_gt"][target, lo:hi]

pch = np.array([committed["Pch_ev"].get((target, t), 0.0) for t in t_range])
pdis = np.array([committed["Pdis_ev"].get((target, t), 0.0) for t in t_range])
expected_dsoc = np.zeros(len(t_range))
expected_dsoc[1:] = (ETA_CH * pch[:-1] - pdis[:-1] / ETA_DIS) * DT_HOURS / ev["cap_kwh"]
d_reported = np.diff(reported, prepend=reported[0])
residual = d_reported - expected_dsoc

# --- continuous risk score (sliding window, for illustration only) ---
risk_curve = np.full(len(t_range), np.nan)
for k in range(len(t_range)):
    if k < 2:
        continue
    start = max(0, k - WINDOW)
    seg_soc = reported[start:k]
    seg_res = residual[start:k]
    seg_pch = pch[start:k]
    seg_pdis = pdis[start:k]
    feat = np.zeros((WINDOW, 5), dtype=np.float32)
    n_valid = len(seg_soc)
    off = WINDOW - n_valid
    feat[off:, 0] = seg_soc
    feat[off:, 1] = seg_res * 10.0
    feat[off:, 2] = (seg_res ** 2) * 100.0
    feat[off:, 3] = seg_pch / ev["p_max_kw"]
    feat[off:, 4] = seg_pdis / ev["p_max_kw"]
    risk_curve[k] = float(risk_scores(model, feat[None, :, :])[0])

# --- CUSUM alarm state over the same trace ---
cusum = CusumDetector(0.001, 0.020)
cusum_alarm = np.zeros(len(t_range), dtype=bool)
for k in range(len(t_range)):
    alarm, _ = cusum.update(target, float(residual[k]))
    cusum_alarm[k] = alarm

# ---------------- plot ----------------
fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1.4]})

t_hours = (t_range - lo) * DT_HOURS
attack_spans = []
in_span = False
for k, a in enumerate(attacked_mask):
    if a and not in_span:
        start_k = k; in_span = True
    if not a and in_span:
        attack_spans.append((start_k, k)); in_span = False
if in_span:
    attack_spans.append((start_k, len(attacked_mask)))

for ax in axes:
    for (s, e) in attack_spans:
        ax.axvspan(t_hours[s], t_hours[min(e, len(t_hours) - 1)], color=RED, alpha=0.08, zorder=0)
for c in [0, 40]:
    if lo <= c < hi:
        for ax in axes:
            ax.axvline((c - lo) * DT_HOURS, color="grey", linestyle=":", linewidth=1)

axes[0].plot(t_hours, true_soc, color=NAVY, lw=2, label="True SoC")
axes[0].plot(t_hours, reported, color=RED, lw=1.4, ls="--", label="Reported (attacked) SoC")
axes[0].set_ylabel("SoC")
axes[0].set_title(f"Case study: EV {target} (attack coverage window shaded, checkpoints dotted)", fontsize=11)
axes[0].legend(loc="upper right", fontsize=8.5, ncol=2)

axes[1].plot(t_hours, residual, color="#555555", lw=1.2)
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set_ylabel("Residual\n(SoC/step)")

axes[2].plot(t_hours, risk_curve, color=NAVY, lw=1.8, label="REB-S risk score r(t)")
axes[2].axhline(0.5, color=NAVY, lw=0.8, ls=":", label="Detection threshold")
ax2b = axes[2].twinx()
ax2b.step(t_hours, cusum_alarm.astype(float), color=GOLD, lw=1.4, where="post", label="CUSUM alarm", alpha=0.8)
ax2b.set_ylim(-0.1, 1.6)
ax2b.set_yticks([0, 1])
ax2b.set_yticklabels(["off", "ON"], fontsize=8)
axes[2].set_ylim(-0.05, 1.05)
axes[2].set_ylabel("Risk score")
axes[2].set_xlabel("Hours since EV arrival")
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, ncol=1)

plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/case_study_trace.png", dpi=200, bbox_inches="tight")
print("saved case study trace, target EV:", target, "attack coverage steps:", int(coverage_per_ev[target]))
