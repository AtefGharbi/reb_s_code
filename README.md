# REB-S Simulation Code

Reference implementation used to generate every table and figure in the
REB-S manuscript (Sections 6-7): a resilient MILP scheduler for joint
EV/BESS charging (explicit binary charge/discharge exclusivity), a stealthy
attack threat model, CUSUM/Isolation Forest/LSTM baselines, and a
Transformer-based detector with risk-weighted state reconstruction and
bounded-trust departure-time handling — tied together in a receding-horizon
simulation, with a full statistical validation harness (20+ seeds, paired
significance testing, two ablation studies, a tau_cap sensitivity sweep,
and a properly separated train/validation/test calibration protocol).

## IMPORTANT — synthetic data
`network.py` generates synthetic fleet/feeder parameters representative of
the case study described in the paper. It is NOT real utility data and NOT
a recognized public test feeder (e.g. IEEE 33-bus) or public EV mobility
dataset — this is stated explicitly in the paper (Sections 6.1-6.2, 7.12)
rather than implied. Before treating results as deployment-ready, replace
the generators in `network.py` with real feeder parameters and real fleet
mobility traces — every downstream script works unchanged.

## Files
- `network.py`          - synthetic fleet/BESS/price/feeder parameters
- `attacks.py`           - stealthy SoC-spoofing & departure-time-injection threat model (eqs. 9-10)
- `scheduler.py`         - MILP (PuLP/CBC) implementing eqs. (1)-(8) with explicit binary u_i(t)
- `cusum.py`              - CUSUM reactive baseline detector
- `detector.py`           - Transformer AND LSTM detectors: synthetic training data
                            (domain-matched to the fleet, randomized attack onset,
                            partial-history augmentation, saturation-aware residual)
- `calibrate.py`          - selects Transformer/LSTM threshold and CUSUM (k,h) on a
                            VALIDATION split only (seed 502), frozen before any test-set
                            or operational evaluation; writes `calibration.json`
- `simulate.py`           - receding-horizon simulation driver: scheduler + attacks +
                            detectors + state reconstruction (eq. 13) + bounded-trust
                            departure handling (eq. 14, with tunable tau_cap) + ablation toggles
- `metrics.py`            - operational + detection metrics (pooled confusion-matrix aggregation)
- `run_experiments.py`    - top-level script: Table 3 (20 seeds, mean+/-std, paired stats),
                            two ablation studies (joint-attack and SoC-only), tau_cap sweep,
                            intensity + coverage detection sweeps (30 seeds for coverage),
                            extended scalability (5-200 EVs, measured inference latency)
- `detection_benchmark.py` - static ROC/PR benchmark: Transformer vs. LSTM vs. Isolation
                            Forest vs. CUSUM's fixed calibrated operating point (Figure 6)
- `make_*.py`              - figure-generation scripts, one per paper figure
- `trained_detector.pt` / `trained_lstm.pt` - trained weights used for the paper's results
- `calibration.json`       - frozen validation-selected thresholds (see calibrate.py)
- `results/`               - CSV outputs from the last full run (`run_log.txt` has the
                            full console transcript, including F1 sanity-check assertions)
- `figures/`                - all figures used in the paper, at full resolution

## Reproducing the paper's results
```bash
pip install pulp numpy pandas torch matplotlib scikit-learn scipy --break-system-packages
python3 calibrate.py               # selects thresholds on validation split (seed 502)
python3 run_experiments.py         # ~10-15 minutes on CPU (includes 200-EV scalability point)
python3 detection_benchmark.py     # ROC/PR curves + AUROC/AUPRC (Figure 6)
python3 make_architecture_fig.py
python3 make_case_study_fig.py
python3 make_attention_fig.py
python3 make_grouped_bar_fig.py
python3 make_coverage_ablation_figs.py
python3 make_sensitivity_fig.py
python3 make_soc_only_taucap_figs.py
```
This retrains nothing by default (loads `trained_detector.pt` / `trained_lstm.pt`
and `calibration.json`). To retrain from scratch:
```python
from detector import train_detector, train_lstm
import torch
model = train_detector(seed=1, n_train=8000, n_val=1200, epochs=45, verbose=True)
torch.save(model.state_dict(), "trained_detector.pt")
lstm = train_lstm(seed=1, n_train=8000, n_val=1200, epochs=45, verbose=True)
torch.save(lstm.state_dict(), "trained_lstm.pt")
# then re-run calibrate.py before run_experiments.py / detection_benchmark.py
```
Hardware/software used to produce the paper's numbers: single CPU core,
Python 3.12.3, PyTorch 2.13.0, PuLP 3.3.2 (bundled CBC), Linux x86_64 — no
GPU used or required at these problem sizes (see `results/hardware_info.json`).

## What changed in this (second) revision round
Round 1 fixed a metric-aggregation bug, re-added explicit MILP binaries, and
added the state-reconstruction / robust-departure mechanisms. Round 2,
responding to a second, more detailed review, made these further changes:

- **Found and fixed a second real bug**: the "expected" physics transition
  used for the detection residual didn't account for SoC saturation, which
  inflated benign-window residual variance ~8x whenever a trace neared
  0%/100% SoC (common at BESS-scale C-rates). Fixed by deriving the
  expected transition from the same saturation-aware trajectory used to
  generate the "true" SoC, in both training data and the operational
  simulation. Required retraining both detectors.
- **Proper train/validation/test separation**: `calibrate.py` now selects
  every detector's operating point on a validation split only (seed 502),
  frozen before evaluation on a disjoint test split (seed 501), itself
  disjoint from the training split (seed 500). Documented in the paper
  (Section 6.6) including why window-level independence holds (each window
  is an independently-generated instance, not a sub-window of a shared
  trajectory).
- **Added an LSTM baseline** (`LSTMDetector`, `train_lstm`) — and reported
  honestly that it modestly **outperforms** the Transformer on the static
  ROC/PR benchmark (AUROC 0.939 vs. 0.914). This is not hidden or spun;
  see paper Section 7.6 for the reported explanation and its limits.
- **SoC-spoofing-only ablation** (tau_max=0) added to isolate eq. (13)'s
  standalone contribution from eq. (14)'s. Finding: margin-tightening alone
  already reaches 100% completion under pure SoC attacks at the intensity
  tested; reconstruction alone can *underperform* doing nothing (no margin
  buffer to absorb its own estimation error). Reported as found.
- **tau_cap sensitivity sweep** {0, 0.5, 1, 2} hours — a clean, monotonic
  reliability/flexibility trade-off, replacing the single tau_cap=0
  operating point used in round 1.
- **Coverage sweep redone at 30 seeds** (was 10) — the higher-powered
  result reverses the earlier, noisier finding: CUSUM now exceeds REB-S at
  4 of 5 coverage levels tested. Reported as measured.
- **Scalability extended to 200 EVs** with measured (not estimated)
  detector inference latency at every size, and hardware/software specs
  logged.
- **Full simulation parameter table** added (Table 2 in the paper) so the
  case study is completely specified, not just described qualitatively.
- Two references (TranAD, Li & Li) updated to their published venues
  (VLDB, IEEE Trans. Smart Grid) in place of arXiv preprints.
- Removed manuscript-development language ("first version of this study",
  "original design") per reviewer request; abstract trimmed and its
  detection-comparison claim reworded to be exact rather than favorable.

## Known remaining limitations (see paper Section 7.12 for full discussion)
- Still synthetic data throughout — not a recognized public test feeder or
  public EV mobility dataset, and not real utility data.
- Only 2 re-optimization checkpoints per day, so sub-checkpoint detection
  latency is still not resolved.
- The Transformer, not the LSTM, is used in the operational scheduler
  (Sections 7.1-7.4) despite the LSTM's edge on the static benchmark — an
  LSTM-based or hybrid REB-S variant is flagged as future work, not done.

