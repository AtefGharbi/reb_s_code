"""
Static detection benchmark on the held-out synthetic window distribution
(Section 5.3): compares the Transformer detector against an Isolation
Forest baseline (trained on benign-only windows, standard anomaly-detection
practice) via full ROC/PR curves and AUROC/AUPRC, with the CUSUM baseline's
single tuned operating point overlaid for reference (CUSUM is a stateful/
sequential detector, so a full threshold sweep is not directly comparable
in the same way -- see Section 7.2 discussion).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from detector import (TransformerDetector, LSTMDetector, generate_dataset, risk_scores,
                       lstm_scores, WINDOW, FEATS)
from cusum import CusumDetector
import json

NAVY = "#1F3864"
GOLD = "#B08968"
RED = "#B5482A"
GREEN = "#3D7A4D"

model = TransformerDetector()
model.load_state_dict(torch.load("trained_detector.pt"))
model.eval()

lstm_model = LSTMDetector()
lstm_model.load_state_dict(torch.load("trained_lstm.pt"))
lstm_model.eval()

with open("calibration.json") as f:
    CAL = json.load(f)

# --- benchmark dataset: SAME held-out test seed as calibrate.py's val is
# distinct from (val=502, test=501, train=500) -- test set never seen
# during training OR threshold selection ---
rng_train = np.random.default_rng(500)
rng_test = np.random.default_rng(501)
X_train, y_train = generate_dataset(4000, rng_train)
X_test, y_test = generate_dataset(3000, rng_test)

# --- Transformer & LSTM scores ---
transformer_scores = risk_scores(model, X_test)
lstm_test_scores = lstm_scores(lstm_model, X_test)

# --- Isolation Forest, trained on BENIGN-ONLY windows (standard IF practice) ---
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
benign_only = X_train_flat[y_train == 0]
iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=0)
iso.fit(benign_only)
iso_raw = -iso.score_samples(X_test_flat)   # higher = more anomalous
iso_scores = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())

# --- CUSUM at its validation-calibrated (k, h) operating point ---
CUSUM_K, CUSUM_H = CAL["cusum_k"], CAL["cusum_h"]
resid = X_test[:, :, 1] / 10.0
cusum_pred = np.zeros(len(y_test))
for n in range(len(y_test)):
    det = CusumDetector(CUSUM_K, CUSUM_H)
    alarmed = False
    for t in range(resid.shape[1]):
        alarm, _ = det.update("a", float(resid[n, t]))
        if alarm:
            alarmed = True
    cusum_pred[n] = 1.0 if alarmed else 0.0
cusum_tpr = (cusum_pred[y_test == 1] == 1).mean()
cusum_fpr = (cusum_pred[y_test == 0] == 1).mean()
cusum_prec = (y_test[cusum_pred == 1] == 1).mean() if (cusum_pred == 1).any() else np.nan
cusum_rec = cusum_tpr

auroc_t = roc_auc_score(y_test, transformer_scores)
auroc_l = roc_auc_score(y_test, lstm_test_scores)
auroc_i = roc_auc_score(y_test, iso_scores)
auprc_t = average_precision_score(y_test, transformer_scores)
auprc_l = average_precision_score(y_test, lstm_test_scores)
auprc_i = average_precision_score(y_test, iso_scores)

fpr_t, tpr_t, _ = roc_curve(y_test, transformer_scores)
fpr_l, tpr_l, _ = roc_curve(y_test, lstm_test_scores)
fpr_i, tpr_i, _ = roc_curve(y_test, iso_scores)
prec_t, rec_t, _ = precision_recall_curve(y_test, transformer_scores)
prec_l, rec_l, _ = precision_recall_curve(y_test, lstm_test_scores)
prec_i, rec_i, _ = precision_recall_curve(y_test, iso_scores)

print(f"Transformer:      AUROC={auroc_t:.3f} AUPRC={auprc_t:.3f}")
print(f"LSTM:             AUROC={auroc_l:.3f} AUPRC={auprc_l:.3f}")
print(f"IsolationForest:  AUROC={auroc_i:.3f} AUPRC={auprc_i:.3f}")
print(f"CUSUM operating point (k={CUSUM_K}, h={CUSUM_H}): TPR(recall)={cusum_rec:.3f} FPR={cusum_fpr:.3f} precision={cusum_prec:.3f}")

with open("results/roc_pr_summary.csv", "w") as f:
    f.write("detector,AUROC,AUPRC\n")
    f.write(f"Transformer (REB-S),{auroc_t:.4f},{auprc_t:.4f}\n")
    f.write(f"LSTM,{auroc_l:.4f},{auprc_l:.4f}\n")
    f.write(f"IsolationForest,{auroc_i:.4f},{auprc_i:.4f}\n")
    f.write(f"CUSUM (operating point only),,\n")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
axes[0].plot(fpr_t, tpr_t, color=NAVY, lw=2, label=f"Transformer (AUROC={auroc_t:.2f})")
axes[0].plot(fpr_l, tpr_l, color=GREEN, lw=2, label=f"LSTM (AUROC={auroc_l:.2f})")
axes[0].plot(fpr_i, tpr_i, color=GOLD, lw=2, label=f"Isolation Forest (AUROC={auroc_i:.2f})")
axes[0].scatter([cusum_fpr], [cusum_tpr], color=RED, zorder=5, s=50, label="CUSUM (fixed op. point)")
axes[0].plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("(a) ROC"); axes[0].legend(fontsize=7.5, loc="lower right")

axes[1].plot(rec_t, prec_t, color=NAVY, lw=2, label=f"Transformer (AUPRC={auprc_t:.2f})")
axes[1].plot(rec_l, prec_l, color=GREEN, lw=2, label=f"LSTM (AUPRC={auprc_l:.2f})")
axes[1].plot(rec_i, prec_i, color=GOLD, lw=2, label=f"Isolation Forest (AUPRC={auprc_i:.2f})")
axes[1].scatter([cusum_rec], [cusum_prec], color=RED, zorder=5, s=50, label="CUSUM (fixed op. point)")
axes[1].axhline(y_test.mean(), color="grey", lw=1, ls="--", label="Chance (prevalence)")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-Recall"); axes[1].legend(fontsize=7.5, loc="lower left")

plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=200, bbox_inches="tight")
print("saved ROC/PR figure")
