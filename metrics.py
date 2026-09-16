import numpy as np
from network import ETA_CH, ETA_DIS, DT_HOURS


def unmanaged_peak(evs, bess, base, n_steps=96):
    """Heuristic 'no coordination' baseline: every EV charges at max power
    immediately on arrival until full or departed; BESS idle. Used only to
    compute the peak-demand-reduction percentage."""
    grid = base.copy()
    for e in evs:
        soc = e["soc_arrival"]
        for t in range(e["arrival"], min(e["departure"], n_steps)):
            if soc >= e["soc_max"]:
                break
            p = e["p_max_kw"]
            grid[t] += p
            soc += ETA_CH * p * DT_HOURS / e["cap_kwh"]
    return grid.max()


def compute_metrics(res, method):
    evs, bess = res["evs"], res["bess"]
    price, v2g_price, base = res["price"], res["v2g"], res["base"]
    committed = res["committed"]
    true_soc_ev, true_soc_bess = res["true_soc_ev"], res["true_soc_bess"]
    n_steps = 96

    total_cost, v2g_revenue, degr_cost = 0.0, 0.0, 0.0
    grid = base.copy()
    for e in evs:
        i = e["id"]
        for t in range(e["arrival"], n_steps):
            ch = committed["Pch_ev"].get((i, t), 0.0)
            dis = committed["Pdis_ev"].get((i, t), 0.0)
            if ch == 0.0 and dis == 0.0 and (i, t) not in committed["Pch_ev"]:
                continue
            total_cost += price[t] * ch * DT_HOURS
            rev = v2g_price[t] * dis * DT_HOURS
            v2g_revenue += rev
            total_cost -= rev
            grid[t] += ch - dis
    for bunit in bess:
        b = bunit["id"]
        for t in range(n_steps):
            ch = committed["Pch_b"].get((b, t), 0.0)
            dis = committed["Pdis_b"].get((b, t), 0.0)
            total_cost += price[t] * (ch - dis) * DT_HOURS
            degr_cost += bunit["degr_coeff"] * (ch + dis) * DT_HOURS
            grid[t] += ch - dis
    total_cost += degr_cost
    peak = grid.max()
    u_peak = unmanaged_peak(evs, bess, base)
    peak_reduction_pct = 100.0 * (u_peak - peak) / u_peak

    completed, violations = 0, 0
    for e in evs:
        i = e["id"]
        dep = e["departure"]
        soc_at_dep = None
        for t in range(dep, -1, -1):
            if t in true_soc_ev[i]:
                soc_at_dep = true_soc_ev[i][t]
                break
        if soc_at_dep is None:
            soc_at_dep = e["soc_arrival"]
        if soc_at_dep + 1e-6 >= e["soc_dep_min_true"]:
            completed += 1
        else:
            violations += 1
    completion_rate = 100.0 * completed / len(evs)

    total_dis_kwh = sum(committed["Pdis_ev"].get((e["id"], t), 0.0) * DT_HOURS
                         for e in evs for t in range(n_steps))
    total_ch_b_kwh = sum(committed["Pch_b"].get((b["id"], t), 0.0) * DT_HOURS
                          for b in bess for t in range(n_steps))
    total_dis_b_kwh = sum(committed["Pdis_b"].get((b["id"], t), 0.0) * DT_HOURS
                           for b in bess for t in range(n_steps))
    total_cap = sum(b["cap_kwh"] for b in bess) or 1.0
    efc = (total_ch_b_kwh + total_dis_b_kwh) / (2 * total_cap)

    return dict(
        method=method, total_cost=total_cost, peak_kw=peak,
        peak_reduction_pct=peak_reduction_pct, completion_rate_pct=completion_rate,
        constraint_violations=violations, bess_efc=efc, degr_cost=degr_cost,
        v2g_revenue=v2g_revenue, ev_discharge_kwh=total_dis_kwh,
        mean_solve_time=float(np.mean(res["solve_times"])) if res["solve_times"] else 0.0,
        max_solve_time=float(np.max(res["solve_times"])) if res["solve_times"] else 0.0,
    )


def detection_metrics(detect_records, threshold=0.5):
    if not detect_records:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, false_alarm_rate=np.nan, n=0)
    y_true = np.array([r[4] for r in detect_records], dtype=bool)
    score = np.array([r[3] for r in detect_records], dtype=float)
    y_pred = score >= threshold
    tp = int(np.sum(y_pred & y_true)); fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true)); tn = int(np.sum(~y_pred & ~y_true))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0 else float("nan")
    fa = fp / (fp + tn) if (fp + tn) else float("nan")
    acc = (tp + tn) / len(y_true)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, false_alarm_rate=fa, n=len(y_true))
