"""
Implements the stealthy SoC spoofing (eq. 9) and departure-time injection
(eq. 10) attack model from Section 4 of the paper.
"""
import numpy as np


def soc_spoofing_bias(n_steps, delta_max, ddelta_max, active_mask, rng):
    """
    Rate-limited random-walk bias delta_i(t), |delta|<=delta_max,
    |delta(t)-delta(t-1)|<=ddelta_max, active only where active_mask[t] is True.
    Returns an array of length n_steps.
    """
    delta = np.zeros(n_steps)
    cur = 0.0
    for t in range(n_steps):
        if active_mask[t]:
            step = rng.uniform(-ddelta_max, ddelta_max)
            cur = np.clip(cur + step, -delta_max, delta_max)
        else:
            cur = 0.0
        delta[t] = cur
    return delta


def make_attack_scenario(evs, bess, n_steps, delta_max, ddelta_max, coverage,
                          duration_steps, tau_max, rng):
    """
    Build a full attack scenario: which assets are attacked, when (a contiguous
    active window per attacked asset), and the resulting SoC bias trajectories
    plus a departure-time injection for attacked EVs.

    Returns dict with per-asset bias arrays and a ground-truth 'attacked'
    boolean array [n_assets, n_steps] used later to score detection.
    """
    n_ev, n_bess = len(evs), len(bess)
    ev_bias = np.zeros((n_ev, n_steps))
    bess_bias = np.zeros((n_bess, n_steps))
    ev_attacked_gt = np.zeros((n_ev, n_steps), dtype=bool)
    bess_attacked_gt = np.zeros((n_bess, n_steps), dtype=bool)
    tau = np.zeros(n_ev, dtype=int)

    n_ev_attacked = int(round(coverage * n_ev))
    n_bess_attacked = int(round(coverage * n_bess))
    attacked_ev_ids = rng.choice(n_ev, size=n_ev_attacked, replace=False) if n_ev_attacked > 0 else []
    attacked_bess_ids = rng.choice(n_bess, size=n_bess_attacked, replace=False) if n_bess_attacked > 0 else []

    if delta_max > 0:
        for i in attacked_ev_ids:
            start = int(rng.integers(0, max(1, n_steps - duration_steps)))
            mask = np.zeros(n_steps, dtype=bool)
            mask[start:start + duration_steps] = True
            ev_bias[i] = soc_spoofing_bias(n_steps, delta_max, ddelta_max, mask, rng)
            ev_attacked_gt[i] = mask
            if tau_max > 0:
                tau[i] = int(rng.integers(-tau_max, tau_max + 1))

        for b in attacked_bess_ids:
            start = int(rng.integers(0, max(1, n_steps - duration_steps)))
            mask = np.zeros(n_steps, dtype=bool)
            mask[start:start + duration_steps] = True
            bess_bias[b] = soc_spoofing_bias(n_steps, delta_max, ddelta_max, mask, rng)
            bess_attacked_gt[b] = mask

    return dict(ev_bias=ev_bias, bess_bias=bess_bias,
                ev_attacked_gt=ev_attacked_gt, bess_attacked_gt=bess_attacked_gt,
                tau=tau)
