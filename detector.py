"""
Transformer-based stealthy-attack detector (Section 5 of the paper).

Trains on synthetically generated windows of [SoC, dSoC, Pch, Pdis] with
randomly injected stealthy attacks (same threat model as attacks.py),
independent of the operational scheduler simulation, matching the offline
pretraining setup described in Section 5.3.
"""
import numpy as np
import torch
import torch.nn as nn

WINDOW = 12          # 6 hours of 30-min steps
FEATS = 5            # soc, residual, residual^2, pch, pdis
NOISE_STD = 0.004


def _gen_benign_power_trace(n_steps, p_max, rng):
    """Smooth random charge/discharge command trace, roughly realistic."""
    t = np.arange(n_steps)
    n_events = rng.integers(1, 4)
    trace = np.zeros(n_steps)
    for _ in range(n_events):
        c = rng.uniform(0, n_steps)
        w = rng.uniform(3, 10)
        sign = rng.choice([1.0, -1.0])
        amp = rng.uniform(0.2, 0.55) * p_max
        trace += sign * amp * np.exp(-0.5 * ((t - c) / w) ** 2)
    pch = np.clip(trace, 0, p_max)
    pdis = np.clip(-trace, 0, p_max)
    return pch, pdis


def generate_dataset(n_windows, rng, delta_range=(0.02, 0.10),
                      eta_ch=0.95, eta_dis=0.95, dt=0.5):
    """
    Training windows are drawn with the SAME asset parameter combinations
    used in the operational fleet (network.make_ev_fleet / make_bess_fleet)
    so the detector is not evaluated out-of-distribution at deployment time.
    """
    from attacks import soc_spoofing_bias
    X = np.zeros((n_windows, WINDOW, FEATS), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.float32)
    n_steps = WINDOW + 1
    ev_pairs = [(50.0, 7.4), (62.0, 7.4), (75.0, 11.0), (62.0, 11.0)]
    bess_pairs = [(100.0, 50.0), (150.0, 75.0), (200.0, 100.0)]
    for n in range(n_windows):
        cap_kwh, p_max = ev_pairs[rng.integers(len(ev_pairs))] if rng.random() < 0.7 \
            else bess_pairs[rng.integers(len(bess_pairs))]
        pch, pdis = _gen_benign_power_trace(n_steps, p_max, rng)
        true_soc = np.zeros(n_steps)
        true_soc[0] = rng.uniform(0.35, 0.65)
        for t in range(n_steps - 1):
            nxt = true_soc[t] + (eta_ch * pch[t] - pdis[t] / eta_dis) * dt / cap_kwh
            true_soc[t + 1] = float(np.clip(nxt, 0.0, 1.0))  # per-step clip: a real battery saturates and STAYS there, it doesn't overshoot and silently un-clip later

        # Label = "currently under attack at the last (most recent) step",
        # matching the operational semantics in simulate.py. The attack, when
        # present, starts at a random onset and remains active through the
        # end of the window, so the model sees examples ranging from
        # "just started" (hard) to "long-running" (easy) -- not just the
        # fully-attacked-throughout case.
        attacked = rng.random() < 0.5
        if attacked:
            delta_max = rng.uniform(*delta_range)
            ddelta_max = delta_max / rng.uniform(3, 6)
            onset = rng.integers(-WINDOW, WINDOW)   # may start before window (fully visible) or partway through
            active = np.zeros(n_steps, dtype=bool)
            active[max(onset, 0):] = True
            delta = soc_spoofing_bias(n_steps, delta_max, ddelta_max, active, rng)
            frac_attacked = 1.0
        else:
            delta = np.zeros(n_steps)
            frac_attacked = 0.0

        noise = rng.normal(0, NOISE_STD, size=n_steps)
        reported = true_soc + delta + noise
        d_reported = np.diff(reported, prepend=reported[0])

        # Physics-based residual: observed increment minus the increment
        # expected from the commanded power under the SAME saturation-aware
        # dynamics used to generate true_soc above (comparing against an
        # unclipped linear formula would itself produce spurious residuals
        # whenever a trace nears 0%/100% SoC, which is common at BESS-scale
        # C-rates and would unfairly inflate every detector's false-alarm
        # rate on purely benign windows).
        expected_dsoc = np.diff(true_soc, prepend=true_soc[0])
        residual = d_reported - expected_dsoc

        X[n, :, 0] = reported[:WINDOW]
        X[n, :, 1] = residual[:WINDOW] * 10.0        # scaled for numerical friendliness
        X[n, :, 2] = (residual[:WINDOW] ** 2) * 100.0  # magnitude cue (attack raises variance, not mean)
        X[n, :, 3] = pch[:WINDOW] / p_max
        X[n, :, 4] = pdis[:WINDOW] / p_max
        y[n] = frac_attacked

        # Partial-history augmentation: a newly-connected asset has fewer
        # than WINDOW valid past readings at the first re-optimization
        # checkpoint. Zero-pad a random-length prefix (matching
        # simulate._feature_window's padding) so the detector is not only
        # ever trained on full, mature windows.
        if rng.random() < 0.25:
            valid_len = int(rng.integers(5, WINDOW))
            X[n, :WINDOW - valid_len, :] = 0.0
    return X, y


class TransformerDetector(nn.Module):
    def __init__(self, feats=FEATS, d_model=32, nhead=4, num_layers=2, window=WINDOW):
        super().__init__()
        self.embed = nn.Linear(feats, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=64, batch_first=True,
                                            dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.encoder(h)
        h = h.mean(dim=1)
        return torch.sigmoid(self.head(h)).squeeze(-1)


def train_detector(seed=0, n_train=6000, n_val=1000, epochs=12, batch_size=64, verbose=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xtr, ytr = generate_dataset(n_train, rng)
    Xva, yva = generate_dataset(n_val, rng)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)

    model = TransformerDetector()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCELoss()
    n = Xtr_t.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = lossf(pred, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose:
            model.eval()
            with torch.no_grad():
                vpred = model(Xva_t)
                vloss = lossf(vpred, yva_t).item()
                vacc = ((vpred > 0.5).float() == yva_t).float().mean().item()
            print(f"epoch {ep}: train_loss={tot/n:.4f} val_loss={vloss:.4f} val_acc={vacc:.4f}")
    model.eval()
    return model


@torch.no_grad()
def risk_scores(model, window_batch):
    """window_batch: np.array [B, WINDOW, FEATS] -> np.array [B] risk in [0,1]."""
    x = torch.tensor(window_batch.astype(np.float32))
    return model(x).numpy()


class LSTMDetector(nn.Module):
    """Temporal deep-learning baseline: same input features and training
    procedure as TransformerDetector, but a 2-layer LSTM instead of
    self-attention, so the Transformer's advantage (if any) over a strong
    sequential deep model -- not just over a non-sequential Isolation
    Forest -- can be isolated."""
    def __init__(self, feats=FEATS, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feats, hidden_size=hidden, num_layers=num_layers,
                             batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.head(last)).squeeze(-1)


def train_lstm(seed=1, n_train=8000, n_val=1200, epochs=45, batch_size=64, verbose=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xtr, ytr = generate_dataset(n_train, rng)
    Xva, yva = generate_dataset(n_val, rng)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva)

    model = LSTMDetector()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCELoss()
    n = Xtr_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = lossf(pred, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose:
            model.eval()
            with torch.no_grad():
                vpred = model(Xva_t)
                vacc = ((vpred > 0.5).float() == yva_t).float().mean().item()
            print(f"[LSTM] epoch {ep}: train_loss={tot/n:.4f} val_acc={vacc:.4f}")
    model.eval()
    return model


@torch.no_grad()
def lstm_scores(model, window_batch):
    x = torch.tensor(window_batch.astype(np.float32))
    return model(x).numpy()
