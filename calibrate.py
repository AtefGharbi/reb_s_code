"""
Threshold calibration on a VALIDATION set, independent of both the
Transformer's training data and the final held-out test set used for
Table 3 / Figure 4 / all operational simulation results.

Split design: every window in generate_dataset() is an independently
sampled synthetic instance (fresh initial SoC, fresh commanded-power
trace, fresh attack draw) rather than a sub-window carved from a shared
pool of longer trajectories, so there is no risk of the same underlying
trajectory appearing in two splits -- the three sets are generated with
three disjoint numpy Generator seeds and share no random state.

  train seed = 500  (used to fit the Transformer's weights, detector.py)
  val   seed = 502  (used ONLY here, to pick decision thresholds)
  test  seed = 501  (used ONLY in detection_benchmark.py / Table 3, never
                      seen during training or threshold selection)
"""
import json
import numpy as np
import torch
from detector import TransformerDetector, generate_dataset, risk_scores
from cusum import CusumDetector

TARGET_FPR = 0.05

model = TransformerDetector()
model.load_state_dict(torch.load("trained_detector.pt"))
model.eval()

rng_val = np.random.default_rng(502)
X_val, y_val = generate_dataset(3000, rng_val)

# --- Transformer: sweep threshold on validation, pick best recall s.t. FPR <= TARGET_FPR ---
scores_val = risk_scores(model, X_val)
thresholds = np.linspace(0.01, 0.99, 197)
best_t, best_recall, best_fpr = 0.5, -1, None
for t in thresholds:
    pred = scores_val >= t
    tp = np.sum(pred & (y_val == 1)); fp = np.sum(pred & (y_val == 0))
    fn = np.sum(~pred & (y_val == 1)); tn = np.sum(~pred & (y_val == 0))
    fpr = fp / (fp + tn) if (fp + tn) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if fpr <= TARGET_FPR and recall > best_recall:
        best_t, best_recall, best_fpr = t, recall, fpr
transformer_threshold = float(best_t)
print(f"Transformer: threshold={transformer_threshold:.3f} -> val recall={best_recall:.3f} val FPR={best_fpr:.3f}")

# --- CUSUM: grid-search (k, h) on validation, same FPR target ---
resid_val = X_val[:, :, 1] / 10.0


def cusum_eval(k, h, X_resid, y):
    preds = np.zeros(len(y))
    for n in range(len(y)):
        det = CusumDetector(k=k, h=h)
        alarmed = False
        for t in range(X_resid.shape[1]):
            alarm, _ = det.update("a", float(X_resid[n, t]))
            if alarm:
                alarmed = True
        preds[n] = 1.0 if alarmed else 0.0
    tp = np.sum((preds == 1) & (y == 1)); fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1)); tn = np.sum((preds == 0) & (y == 0))
    fpr = fp / (fp + tn) if (fp + tn) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return fpr, recall


best_k, best_h, best_c_recall, best_c_fpr = 0.001, 0.02, -1, None
for k in [0.0006, 0.001, 0.0015, 0.002, 0.003]:
    for h in [0.012, 0.016, 0.02, 0.025, 0.03, 0.04]:
        fpr, recall = cusum_eval(k, h, resid_val, y_val)
        if fpr <= TARGET_FPR and recall > best_c_recall:
            best_k, best_h, best_c_recall, best_c_fpr = k, h, recall, fpr
print(f"CUSUM: k={best_k}, h={best_h} -> val recall={best_c_recall:.3f} val FPR={best_c_fpr:.3f}")

config = dict(
    target_fpr=TARGET_FPR,
    transformer_threshold=transformer_threshold,
    transformer_val_recall=float(best_recall), transformer_val_fpr=float(best_fpr),
    cusum_k=best_k, cusum_h=best_h,
    cusum_val_recall=float(best_c_recall), cusum_val_fpr=float(best_c_fpr),
    val_seed=502, val_n=3000,
    note="Thresholds selected on validation (seed=502) only, frozen before any test-set (seed=501) or operational-simulation evaluation."
)
with open("calibration.json", "w") as f:
    json.dump(config, f, indent=2)
print("saved calibration.json:", config)
