"""
Interpretability figure: extract the first encoder layer's self-attention
weight matrix (averaged over heads) for one benign and one attacked
evaluation window, and visualize which time steps the model attends to
when flagging an attack.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector import TransformerDetector, generate_dataset, WINDOW

model = TransformerDetector()
model.load_state_dict(torch.load("trained_detector.pt"))
model.eval()

rng = np.random.default_rng(42)
X, y = generate_dataset(400, rng)

# pick a clean benign example and a clearly-attacked example the model gets right
with torch.no_grad():
    scores = model(torch.tensor(X)).numpy()

benign_idx = [i for i in range(len(y)) if y[i] == 0 and scores[i] < 0.3]
attack_idx = [i for i in range(len(y)) if y[i] == 1 and scores[i] > 0.7]
i_benign = benign_idx[0]
i_attack = attack_idx[0]


def get_attention(x_sample):
    x = torch.tensor(x_sample[None, :, :].astype(np.float32))
    with torch.no_grad():
        h = model.embed(x) + model.pos
        layer = model.encoder.layers[0]
        attn_out, attn_w = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
    return attn_w[0].numpy()  # [WINDOW, WINDOW]


A_benign = get_attention(X[i_benign])
A_attack = get_attention(X[i_attack])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
labels = [f"t-{WINDOW-1-k}" for k in range(WINDOW)]

for ax, A, title, score in [
    (axes[0], A_benign, "Benign window", scores[i_benign]),
    (axes[1], A_attack, "Attacked window", scores[i_attack]),
]:
    im = ax.imshow(A, cmap="YlGnBu", vmin=0, vmax=max(A_benign.max(), A_attack.max()))
    ax.set_xticks(range(WINDOW)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(WINDOW)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Attended-to step (key)")
    ax.set_title(f"{title}\n(model risk score = {score:.2f})", fontsize=10)
axes[0].set_ylabel("Query step")
fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label="Attention weight")

plt.savefig("/home/claude/reb_s_code/figures/attention_heatmap.png", dpi=200, bbox_inches="tight")
print("saved attention heatmap; benign score", scores[i_benign], "attack score", scores[i_attack])

# also save a compact "attention received per step" bar comparison (mean over queries)
fig2, ax = plt.subplots(figsize=(7, 3.2))
recv_benign = A_benign.mean(axis=0)
recv_attack = A_attack.mean(axis=0)
xpos = np.arange(WINDOW)
w = 0.38
ax.bar(xpos - w/2, recv_benign, width=w, label="Benign window", color="#8DA0C4")
ax.bar(xpos + w/2, recv_attack, width=w, label="Attacked window", color="#B5482A")
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8, rotation=45)
ax.set_ylabel("Mean attention received")
ax.set_xlabel("Time step within the 6-hour window")
ax.set_title("Which time steps drive the detector's decision")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/reb_s_code/figures/attention_bars.png", dpi=200, bbox_inches="tight")
print("saved attention bar comparison")
