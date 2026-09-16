"""
Receding-horizon simulation of one 48-hour scenario (96 half-hour steps) for
a given scheduling method: 'nominal' (no attack), 'unaware' (attack-unaware
MILP), 'reactive' (CUSUM + MILP), or 'rebs' (Transformer detector +
risk-adaptive MILP, Section 5.4).

Two re-optimization checkpoints (t=0, t=40) are used: an initial day-ahead
solve, and one mid-horizon re-solve using updated (possibly attacked)
SoC telemetry -- enough to exercise the detection -> re-optimization loop
described in the paper while keeping the number of MILP solves small.
"""
import numpy as np
import json
from network import (make_ev_fleet, make_bess_fleet, time_of_use_price, v2g_price,
                      base_load_profile, ETA_CH, ETA_DIS, FEEDER_MAX_KW, STATION_MAX_KW,
                      DT_HOURS, GAMMA_RISK, BUFFER_SOC)
import scheduler
from attacks import make_attack_scenario
from cusum import CusumDetector
from detector import WINDOW, risk_scores

N_STEPS = 96
CHECKPOINTS = [0, 40]

with open("calibration.json") as _f:
    _CAL = json.load(_f)
CUSUM_K, CUSUM_H = _CAL["cusum_k"], _CAL["cusum_h"]          # selected on validation (seed=502) only, see calibrate.py
TRANSFORMER_THRESHOLD = _CAL["transformer_threshold"]         # selected on validation (seed=502) only, see calibrate.py
REACTIVE_MARGIN = 0.05                     # fixed SoC margin added on CUSUM alarm
NOISE_STD = 0.004
PROCESS_NOISE_STD = 0.0025                 # model-mismatch noise on the physics-reconstructed estimate
DEP_TAU_CAP = 0                            # REB-S: max steps beyond nominal departure it will trust (robust departure handling; 0 = never trust a reported extension beyond the nominal/booked time)


def _feature_window(reported, residual, pch, pdis, end, p_max, window=WINDOW):
    """Build a [window, 5] feature array ending at step `end` (exclusive),
    left-padding with the earliest available value if history is short."""
    start = end - window
    feat = np.zeros((window, 5), dtype=np.float32)
    for k, t in enumerate(range(start, end)):
        tt = max(t, 0)
        if tt < len(reported) and reported[tt] is not None:
            feat[k, 0] = reported[tt]
            feat[k, 1] = residual[tt] * 10.0
            feat[k, 2] = (residual[tt] ** 2) * 100.0
            feat[k, 3] = pch[tt] / p_max if p_max > 0 else 0
            feat[k, 4] = pdis[tt] / p_max if p_max > 0 else 0
        elif k > 0:
            feat[k] = feat[k - 1]
    return feat


def run_day(method, n_ev, n_bess, attack_cfg, seed, torch_model=None, ablation=None, dep_tau_cap=None):
    """
    ablation: optional dict of booleans to selectively disable REB-S
    components for the ablation study (Section 7.x): keys 'reconstruction',
    'robust_departure', 'risk_margin' (all True = full REB-S). Ignored for
    methods other than 'rebs'.
    dep_tau_cap: override DEP_TAU_CAP (steps) for the tau_cap sensitivity sweep.
    """
    ab = dict(reconstruction=True, robust_departure=True, risk_margin=True)
    if ablation:
        ab.update(ablation)
    tau_cap = DEP_TAU_CAP if dep_tau_cap is None else dep_tau_cap
    rng = np.random.default_rng(seed)
    evs = make_ev_fleet(n_ev, rng)
    bess = make_bess_fleet(n_bess, rng)
    price = np.tile(time_of_use_price(), 2)
    v2g = np.tile(v2g_price(), 2)
    base = base_load_profile(rng=rng)

    if method == "nominal" or attack_cfg.get("delta_max", 0) == 0:
        atk = make_attack_scenario(evs, bess, N_STEPS, 0, 0, 0, 0, 0, rng)
    else:
        atk = make_attack_scenario(
            evs, bess, N_STEPS,
            attack_cfg["delta_max"], attack_cfg["ddelta_max"],
            attack_cfg["coverage"], attack_cfg["duration"], attack_cfg.get("tau_max", 4), rng
        )

    true_soc_ev = {e["id"]: {} for e in evs}
    true_soc_bess = {b["id"]: {} for b in bess}
    phys_soc_ev = {e["id"]: {} for e in evs}     # physics-reconstructed estimate (Sec. 5.5 state reconstruction)
    phys_soc_bess = {b["id"]: {} for b in bess}
    for e in evs:
        true_soc_ev[e["id"]][e["arrival"]] = e["soc_arrival"]
        phys_soc_ev[e["id"]][e["arrival"]] = e["soc_arrival"]
    for b in bess:
        true_soc_bess[b["id"]][0] = b["soc_init"]
        phys_soc_bess[b["id"]][0] = b["soc_init"]

    committed = dict(Pch_ev={}, Pdis_ev={}, Pch_b={}, Pdis_b={})
    solve_times = []
    cusum = {e["id"]: CusumDetector(CUSUM_K, CUSUM_H) for e in evs}
    cusum.update({("b", b["id"]): CusumDetector(CUSUM_K, CUSUM_H) for b in bess})
    detect_records = []   # (asset_kind, id, checkpoint, risk_or_alarm, ground_truth)

    for ci, c in enumerate(CHECKPOINTS):
        nb = CHECKPOINTS[ci + 1] if ci + 1 < len(CHECKPOINTS) else N_STEPS

        # --- build reported streams & risk-adaptive parameters ---
        dep_req = {}
        eff_departure = {}
        bess_derate = {}
        risk_ev, risk_bess = {}, {}
        for e in evs:
            i = e["id"]
            tau = int(atk["tau"][i]) if i < len(atk["tau"]) else 0
            reported_departure = int(np.clip(e["departure"] + tau, e["arrival"] + 1, N_STEPS))
            if method == "rebs":
                # Robust departure handling (Sec. 5.5): never trust a reported
                # departure extension beyond DEP_TAU_CAP steps past the nominal
                # (separately-confirmed, e.g. booking-channel) departure time.
                if ab["robust_departure"]:
                    eff_departure[i] = min(reported_departure, e["departure"] + tau_cap)
                    eff_departure[i] = max(eff_departure[i], e["arrival"] + 1)
                else:
                    eff_departure[i] = reported_departure
            else:
                eff_departure[i] = reported_departure
            base_req = e["soc_dep_min_true"]
            if c <= e["arrival"] or c not in true_soc_ev[i]:
                dep_req[i] = base_req
                continue
            t_soc = true_soc_ev[i][c]
            reported_now = t_soc + atk["ev_bias"][i, c] + rng.normal(0, NOISE_STD)
            gt_attacked = bool(atk["ev_attacked_gt"][i, c])

            if method == "reactive":
                resid = reported_now - t_soc  # coordinator's own model has no bias -> resid ~= bias+noise
                alarm, _ = cusum[i].update(i, resid)
                detect_records.append(("ev", i, c, float(alarm), gt_attacked))
                dep_req[i] = base_req + (REACTIVE_MARGIN if alarm else 0.0)
            elif method == "rebs" and torch_model is not None and c - WINDOW >= e["arrival"] - WINDOW:
                reported_hist = [true_soc_ev[i].get(t, None) for t in range(0, c)]
                for t in range(0, c):
                    if reported_hist[t] is not None:
                        reported_hist[t] = reported_hist[t] + atk["ev_bias"][i, t]
                resid_hist = [0.0] * c
                pch_hist = [committed["Pch_ev"].get((i, t), 0.0) for t in range(c)]
                pdis_hist = [committed["Pdis_ev"].get((i, t), 0.0) for t in range(c)]
                for t in range(1, c):
                    if reported_hist[t] is not None and reported_hist[t - 1] is not None:
                        # Saturation-aware expected transition: use the actual
                        # realized (already-clipped) true-SoC diff rather than
                        # re-deriving it from the open-loop power formula,
                        # which would spuriously flag residual near SoC bounds
                        # (see the same fix in detector.py's training data).
                        exp_d = true_soc_ev[i].get(t, 0.0) - true_soc_ev[i].get(t - 1, true_soc_ev[i].get(t, 0.0))
                        resid_hist[t] = (reported_hist[t] - reported_hist[t - 1]) - exp_d
                feat = _feature_window(reported_hist, resid_hist, pch_hist, pdis_hist, c, e["p_max_kw"])
                risk = float(risk_scores(torch_model, feat[None, :, :])[0])
                risk_ev[i] = risk
                detect_records.append(("ev", i, c, risk, gt_attacked))
                dep_req[i] = base_req + (GAMMA_RISK * risk * BUFFER_SOC if ab["risk_margin"] else 0.0)
            else:
                dep_req[i] = base_req

        for bunit in bess:
            b = bunit["id"]
            if c not in true_soc_bess[b]:
                bess_derate[b] = 0.0
                continue
            t_soc = true_soc_bess[b][c]
            reported_now = t_soc + atk["bess_bias"][b, c] + rng.normal(0, NOISE_STD)
            gt_attacked = bool(atk["bess_attacked_gt"][b, c])
            key = ("b", b)
            if method == "reactive":
                resid = reported_now - t_soc
                alarm, _ = cusum[key].update(key, resid)
                detect_records.append(("bess", b, c, float(alarm), gt_attacked))
                bess_derate[b] = 0.3 if alarm else 0.0
            elif method == "rebs" and torch_model is not None:
                reported_hist = [true_soc_bess[b].get(t, None) for t in range(0, c)]
                for t in range(0, c):
                    if reported_hist[t] is not None:
                        reported_hist[t] = reported_hist[t] + atk["bess_bias"][b, t]
                resid_hist = [0.0] * c
                pch_hist = [committed["Pch_b"].get((b, t), 0.0) for t in range(c)]
                pdis_hist = [committed["Pdis_b"].get((b, t), 0.0) for t in range(c)]
                for t in range(1, c):
                    if reported_hist[t] is not None and reported_hist[t - 1] is not None:
                        exp_d = true_soc_bess[b].get(t, 0.0) - true_soc_bess[b].get(t - 1, true_soc_bess[b].get(t, 0.0))
                        resid_hist[t] = (reported_hist[t] - reported_hist[t - 1]) - exp_d
                feat = _feature_window(reported_hist, resid_hist, pch_hist, pdis_hist, c, bunit["p_max_kw"])
                risk = float(risk_scores(torch_model, feat[None, :, :])[0])
                risk_bess[b] = risk
                detect_records.append(("bess", b, c, risk, gt_attacked))
                bess_derate[b] = min(0.6, GAMMA_RISK * risk * 0.6) if ab["risk_margin"] else 0.0
            else:
                bess_derate[b] = 0.0

        # --- reported (or, for REB-S, risk-reconstructed) init SoC fed to the MILP ---
        init_soc_ev, init_soc_bess = {}, {}
        for e in evs:
            i = e["id"]
            lo = max(c, e["arrival"])
            if lo >= N_STEPS or lo >= eff_departure[i]:
                continue
            if lo == e["arrival"]:
                base_val = e["soc_arrival"]
            else:
                base_val = true_soc_ev[i].get(lo, e["soc_arrival"])
            reported_val = float(np.clip(base_val + atk["ev_bias"][i, lo], e["soc_min"], e["soc_max"]))
            if method == "rebs" and ab["reconstruction"] and lo in phys_soc_ev[i]:
                # Risk-weighted state reconstruction (Sec. 5.5): blend the
                # (possibly attacked) reported reading with a physics-based
                # estimate integrated purely from the coordinator's own
                # commanded power since the last trusted anchor, trusting the
                # physics estimate more as risk increases.
                alpha = 1.0 - risk_ev.get(i, 0.0)
                phys_val = float(np.clip(phys_soc_ev[i][lo], e["soc_min"], e["soc_max"]))
                init_soc_ev[i] = alpha * reported_val + (1 - alpha) * phys_val
            else:
                init_soc_ev[i] = reported_val
        for bunit in bess:
            b = bunit["id"]
            base_val = true_soc_bess[b].get(c, bunit["soc_init"])
            reported_val = float(np.clip(base_val + atk["bess_bias"][b, c], bunit["soc_min"], bunit["soc_max"]))
            if method == "rebs" and ab["reconstruction"] and c in phys_soc_bess[b]:
                alpha = 1.0 - risk_bess.get(b, 0.0)
                phys_val = float(np.clip(phys_soc_bess[b][c], bunit["soc_min"], bunit["soc_max"]))
                init_soc_bess[b] = alpha * reported_val + (1 - alpha) * phys_val
            else:
                init_soc_bess[b] = reported_val

        evs_window = [dict(e, departure=eff_departure[e["id"]]) for e in evs if e["id"] in init_soc_ev]
        if not evs_window and not init_soc_bess:
            continue

        out = scheduler.solve_window(evs_window, bess, price, v2g, base, c, N_STEPS,
                                      init_soc_ev, init_soc_bess, dep_req,
                                      FEEDER_MAX_KW, STATION_MAX_KW, DT_HOURS,
                                      bess_derate=bess_derate)
        solve_times.append(out["solve_time"])

        # --- commit decisions for [c, nb) and roll TRUE (and physics-reconstructed) state forward ---
        for e in evs_window:
            i = e["id"]
            lo, hi = out["windows"][i]
            t_end_commit = min(nb, hi)
            cur = true_soc_ev[i].get(c if c > e["arrival"] else e["arrival"], e["soc_arrival"])
            cur_phys = phys_soc_ev[i].get(c if c > e["arrival"] else e["arrival"], e["soc_arrival"])
            for t in range(lo, t_end_commit):
                ch = out["Pch_ev"].get((i, t), 0.0)
                dis = out["Pdis_ev"].get((i, t), 0.0)
                committed["Pch_ev"][(i, t)] = ch
                committed["Pdis_ev"][(i, t)] = dis
                cur = cur + (ETA_CH * ch - dis / ETA_DIS) * DT_HOURS / e["cap_kwh"]
                cur = float(np.clip(cur, e["soc_min"], e["soc_max"]))
                true_soc_ev[i][t + 1] = cur
                cur_phys = cur_phys + (ETA_CH * ch - dis / ETA_DIS) * DT_HOURS / e["cap_kwh"] + rng.normal(0, PROCESS_NOISE_STD)
                cur_phys = float(np.clip(cur_phys, e["soc_min"], e["soc_max"]))
                phys_soc_ev[i][t + 1] = cur_phys
        for bunit in bess:
            b = bunit["id"]
            cur = true_soc_bess[b].get(c, bunit["soc_init"])
            cur_phys = phys_soc_bess[b].get(c, bunit["soc_init"])
            for t in range(c, nb):
                ch = out["Pch_b"].get((b, t), 0.0)
                dis = out["Pdis_b"].get((b, t), 0.0)
                committed["Pch_b"][(b, t)] = ch
                committed["Pdis_b"][(b, t)] = dis
                cur = cur + (ETA_CH * ch - dis / ETA_DIS) * DT_HOURS / bunit["cap_kwh"]
                cur = float(np.clip(cur, bunit["soc_min"], bunit["soc_max"]))
                true_soc_bess[b][t + 1] = cur
                cur_phys = cur_phys + (ETA_CH * ch - dis / ETA_DIS) * DT_HOURS / bunit["cap_kwh"] + rng.normal(0, PROCESS_NOISE_STD)
                cur_phys = float(np.clip(cur_phys, bunit["soc_min"], bunit["soc_max"]))
                phys_soc_bess[b][t + 1] = cur_phys

    return dict(evs=evs, bess=bess, price=price, v2g=v2g, base=base,
                true_soc_ev=true_soc_ev, true_soc_bess=true_soc_bess,
                committed=committed, solve_times=solve_times,
                detect_records=detect_records, attack=atk)
