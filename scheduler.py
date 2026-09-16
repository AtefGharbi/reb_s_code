"""
MILP scheduler implementing the REB-S joint EV-BESS formulation
(Section 3, equations 1-8), solved with CBC via PuLP over a receding
sub-horizon [t0, t_end). Charge/discharge exclusivity is enforced with the
explicit binary indicator u_i(t) (eqs. 2-3), not an LP relaxation. Risk-
adaptive tightening (eq. 11 / Sec. 5.4) is applied by the caller through
`dep_req_ev` and `bess_derate`.
"""
import time
import numpy as np
import pulp
from network import ETA_CH, ETA_DIS, PEAK_WEIGHT


def solve_window(evs, bess, price, v2g, base_load, t0, t_end,
                  init_soc_ev, init_soc_bess, dep_req_ev,
                  feeder_max, station_max, dt, bess_derate=None):
    """
    Solve the MILP over steps [t0, t_end) (half-open).
    init_soc_ev / init_soc_bess: dict asset_id -> SoC at t0 (the value the
        scheduler TRUSTS -- i.e. the reported, possibly attacked, reading).
    dep_req_ev: dict ev_id -> effective minimum departure SoC (already
        includes any risk-adaptive buffer added by the caller).
    bess_derate: dict bess_id -> fraction in [0,1] by which max discharge
        power is derated (risk-adaptive protection against over-cycling).
    Returns dict of per-step decisions and diagnostics (incl. solve_time).
    """
    t_start_wall = time.time()
    T = list(range(t0, t_end))
    prob = pulp.LpProblem("REBS_window", pulp.LpMinimize)

    Pch_ev, Pdis_ev, soc_ev = {}, {}, {}
    Pch_b, Pdis_b, soc_b = {}, {}, {}
    bess_derate = bess_derate or {}

    active_ev = [e for e in evs if e["arrival"] < t_end and e["departure"] > t0]
    u_ev = {}
    for e in active_ev:
        i = e["id"]
        lo, hi = max(t0, e["arrival"]), min(t_end, e["departure"])
        soc_ev[i, lo] = init_soc_ev[i]
        for t in range(lo, hi):
            u_ev[i, t] = pulp.LpVariable(f"u_ev_{i}_{t}", cat="Binary")
            Pch_ev[i, t] = pulp.LpVariable(f"Pch_ev_{i}_{t}", 0, e["p_max_kw"])
            Pdis_ev[i, t] = pulp.LpVariable(f"Pdis_ev_{i}_{t}", 0, e["p_max_kw"])
            prob += Pch_ev[i, t] <= u_ev[i, t] * e["p_max_kw"]
            prob += Pdis_ev[i, t] <= (1 - u_ev[i, t]) * e["p_max_kw"]
            soc_ev[i, t + 1] = pulp.LpVariable(f"soc_ev_{i}_{t+1}", e["soc_min"], e["soc_max"])
            prob += soc_ev[i, t + 1] == soc_ev[i, t] + \
                (ETA_CH * Pch_ev[i, t] - Pdis_ev[i, t] / ETA_DIS) * dt / e["cap_kwh"]
        prob += soc_ev[i, hi] >= dep_req_ev.get(i, e["soc_dep_min_true"])
        e["_window"] = (lo, hi)

    u_b = {}
    for bunit in bess:
        b = bunit["id"]
        soc_b[b, t0] = init_soc_bess[b]
        derate = bess_derate.get(b, 0.0)
        for t in T:
            u_b[b, t] = pulp.LpVariable(f"u_b_{b}_{t}", cat="Binary")
            Pch_b[b, t] = pulp.LpVariable(f"Pch_b_{b}_{t}", 0, bunit["p_max_kw"])
            Pdis_b[b, t] = pulp.LpVariable(f"Pdis_b_{b}_{t}", 0, bunit["p_max_kw"] * (1 - derate))
            prob += Pch_b[b, t] <= u_b[b, t] * bunit["p_max_kw"]
            prob += Pdis_b[b, t] <= (1 - u_b[b, t]) * bunit["p_max_kw"] * (1 - derate)
            soc_b[b, t + 1] = pulp.LpVariable(f"soc_b_{b}_{t+1}", bunit["soc_min"], bunit["soc_max"])
            prob += soc_b[b, t + 1] == soc_b[b, t] + \
                (ETA_CH * Pch_b[b, t] - Pdis_b[b, t] / ETA_DIS) * dt / bunit["cap_kwh"]

    Ppeak = pulp.LpVariable("Ppeak", 0, feeder_max)
    for t in T:
        grid_import = base_load[t] \
            + pulp.lpSum(Pch_ev[e["id"], t] - Pdis_ev[e["id"], t]
                          for e in active_ev if e["_window"][0] <= t < e["_window"][1]) \
            + pulp.lpSum(Pch_b[b["id"], t] - Pdis_b[b["id"], t] for b in bess)
        prob += grid_import <= feeder_max
        prob += Ppeak >= grid_import
        prob += pulp.lpSum(Pch_ev[e["id"], t] for e in active_ev
                            if e["_window"][0] <= t < e["_window"][1]) <= station_max

    # NOTE: EV charging is billed at the retail price; EV discharge (V2G) is
    # compensated at the separate, discounted v2g rate (not net-metered at
    # the retail price) -- billing both would double-count discharge revenue.
    energy_cost = pulp.lpSum(
        price[t] * Pch_ev[e["id"], t] * dt
        for e in active_ev for t in range(*e["_window"])
    ) + pulp.lpSum(
        price[t] * (Pch_b[b["id"], t] - Pdis_b[b["id"], t]) * dt
        for b in bess for t in T
    )
    v2g_revenue = pulp.lpSum(
        v2g[t] * Pdis_ev[e["id"], t] * dt
        for e in active_ev for t in range(*e["_window"])
    )
    degr_cost = pulp.lpSum(
        b["degr_coeff"] * (Pch_b[b["id"], t] + Pdis_b[b["id"], t]) * dt
        for b in bess for t in T
    )
    prob += energy_cost - v2g_revenue + degr_cost + PEAK_WEIGHT * Ppeak

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    solve_time = time.time() - t_start_wall

    out_Pch_ev = {(e["id"], t): Pch_ev[e["id"], t].value() or 0.0
                  for e in active_ev for t in range(*e["_window"])}
    out_Pdis_ev = {(e["id"], t): Pdis_ev[e["id"], t].value() or 0.0
                   for e in active_ev for t in range(*e["_window"])}
    out_Pch_b = {(b["id"], t): Pch_b[b["id"], t].value() or 0.0 for b in bess for t in T}
    out_Pdis_b = {(b["id"], t): Pdis_b[b["id"], t].value() or 0.0 for b in bess for t in T}

    return dict(Pch_ev=out_Pch_ev, Pdis_ev=out_Pdis_ev, Pch_b=out_Pch_b, Pdis_b=out_Pdis_b,
                status=pulp.LpStatus[prob.status], solve_time=solve_time,
                active_ev_ids=[e["id"] for e in active_ev],
                windows={e["id"]: e["_window"] for e in active_ev})
