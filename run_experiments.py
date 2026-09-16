import time, json, resource
import numpy as np
import pandas as pd
import torch
from scipy import stats

from detector import TransformerDetector
from simulate import run_day, TRANSFORMER_THRESHOLD
from metrics import compute_metrics, detection_metrics


def _dm(records, method):
    """Apply the validation-calibrated Transformer threshold for REB-S
    records; CUSUM records are already a binarized alarm flag from its own
    validation-calibrated (k, h), so any threshold in (0, 1) is equivalent."""
    return detection_metrics(records, threshold=TRANSFORMER_THRESHOLD if method == "rebs" else 0.5)

RESULTS = "results"
SEEDS_MAIN = list(range(1, 21))              # 20 seeds (was 5)
MODERATE_ATTACK = dict(delta_max=0.06, ddelta_max=0.015, coverage=0.4, duration=40, tau_max=4)
INTENSITIES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
COVERAGES = [0.1, 0.2, 0.4, 0.6, 0.8]
SCALE_SIZES = [5, 10, 20, 40, 80]
SEEDS_SWEEP = list(range(1, 11))              # 10 seeds per sweep point

print("Loading trained Transformer detector...")
model = TransformerDetector()
model.load_state_dict(torch.load("trained_detector.pt"))
model.eval()


def pooled_detection_metrics(records, method):
    """Correct aggregation: pool the raw confusion-matrix counts across all
    rows before computing precision/recall/F1, rather than averaging
    per-row F1 scores (which is not equal to F1 of the averaged P/R and
    is not a meaningful statistic on its own). Uses the validation-
    calibrated threshold for REB-S; CUSUM records are already binary."""
    return _dm(records, method)


# ---------------- Table 2: operational performance (20 seeds, mean +/- std, stats) ----------------
print(f"\n=== Table 2: operational performance (N=10 EV, 2 BESS, moderate attack, {len(SEEDS_MAIN)} seeds) ===")
rows = []
for method in ["nominal", "unaware", "reactive", "rebs"]:
    for seed in SEEDS_MAIN:
        res = run_day(method, 10, 2, MODERATE_ATTACK, seed, torch_model=model)
        m = compute_metrics(res, method)
        m["seed"] = seed
        rows.append(m)
df_perf = pd.DataFrame(rows)
df_perf.to_csv(f"{RESULTS}/table2_raw.csv", index=False)

summary_rows = []
for method in ["nominal", "unaware", "reactive", "rebs"]:
    sub = df_perf[df_perf.method == method]
    summary_rows.append(dict(
        method=method,
        total_cost_mean=sub.total_cost.mean(), total_cost_std=sub.total_cost.std(),
        peak_reduction_mean=sub.peak_reduction_pct.mean(), peak_reduction_std=sub.peak_reduction_pct.std(),
        completion_mean=sub.completion_rate_pct.mean(), completion_std=sub.completion_rate_pct.std(),
        efc_mean=sub.bess_efc.mean(), efc_std=sub.bess_efc.std(),
        v2g_mean=sub.v2g_revenue.mean(), v2g_std=sub.v2g_revenue.std(),
        violations_mean=sub.constraint_violations.mean(), violations_std=sub.constraint_violations.std(),
    ))
summary2 = pd.DataFrame(summary_rows).set_index("method")
summary2.to_csv(f"{RESULTS}/table2_summary.csv")
print(summary2.round(3))

# Paired statistical tests: REB-S vs unaware, REB-S vs reactive, on completion rate and cost
print("\n--- Statistical significance (paired, same seeds) ---")
stat_rows = []
for metric in ["completion_rate_pct", "total_cost", "bess_efc"]:
    rebs_vals = df_perf[df_perf.method == "rebs"].sort_values("seed")[metric].values
    for other in ["unaware", "reactive"]:
        other_vals = df_perf[df_perf.method == other].sort_values("seed")[metric].values
        try:
            w_stat, w_p = stats.wilcoxon(rebs_vals, other_vals)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        t_stat, t_p = stats.ttest_rel(rebs_vals, other_vals)
        diff = rebs_vals - other_vals
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else np.nan
        stat_rows.append(dict(metric=metric, comparison=f"rebs_vs_{other}",
                               mean_diff=diff.mean(), wilcoxon_p=w_p, ttest_p=t_p, cohens_d=cohens_d))
        print(f"{metric:22s} REB-S vs {other:9s}: mean_diff={diff.mean():+.3f}  "
              f"Wilcoxon p={w_p:.4f}  t-test p={t_p:.4f}  Cohen's d={cohens_d:+.2f}")
pd.DataFrame(stat_rows).to_csv(f"{RESULTS}/table2_significance.csv", index=False)

# ---------------- Ablation study ----------------
print("\n=== Ablation study (REB-S components, 10 seeds, moderate attack) ===")
ABLATIONS = {
    "Full REB-S":            dict(reconstruction=True,  robust_departure=True,  risk_margin=True),
    "No state reconstruction": dict(reconstruction=False, robust_departure=True,  risk_margin=True),
    "No robust departure":   dict(reconstruction=True,  robust_departure=False, risk_margin=True),
    "No risk margin/derate": dict(reconstruction=True,  robust_departure=True,  risk_margin=False),
    "Detector only (all off)": dict(reconstruction=False, robust_departure=False, risk_margin=False),
}
ablation_rows = []
for name, ab in ABLATIONS.items():
    for seed in SEEDS_SWEEP:
        res = run_day("rebs", 10, 2, MODERATE_ATTACK, seed, torch_model=model, ablation=ab)
        m = compute_metrics(res, name)
        m["seed"] = seed
        ablation_rows.append(m)
df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(f"{RESULTS}/ablation_raw.csv", index=False)
summary_ab = df_ablation.groupby("method").agg(
    completion_mean=("completion_rate_pct", "mean"), completion_std=("completion_rate_pct", "std"),
    cost_mean=("total_cost", "mean"), efc_mean=("bess_efc", "mean"),
    violations_mean=("constraint_violations", "mean"),
).reindex(list(ABLATIONS.keys()))
summary_ab.to_csv(f"{RESULTS}/ablation_summary.csv")
print(summary_ab.round(3))

# ---------------- Detection sweep across attack INTENSITY (fixed coverage) ----------------
print("\n=== Detection sweep: attack intensity (reactive vs rebs) ===")
det_rows = []
pooled_records = {"reactive": [], "rebs": []}
for delta in INTENSITIES:
    cfg = dict(delta_max=delta, ddelta_max=max(delta / 4, 1e-6), coverage=0.4, duration=40, tau_max=4)
    for method in ["reactive", "rebs"]:
        all_records = []
        degr_costs = []
        for seed in SEEDS_SWEEP:
            res = run_day(method, 10, 2, cfg, seed, torch_model=model)
            all_records.extend(res["detect_records"])
            degr_costs.append(compute_metrics(res, method)["degr_cost"])
        if delta > 0:
            pooled_records[method].extend(all_records)
        dm = _dm(all_records, method)
        det_rows.append(dict(delta_max=delta, method=method, mean_degr_cost=float(np.mean(degr_costs)), **dm))
        print(f"delta={delta:.2f} {method:9s} n={dm['n']:3d} acc={dm['accuracy']:.3f} "
              f"prec={dm['precision']:.3f} rec={dm['recall']:.3f} f1={dm['f1']:.3f} FA={dm['false_alarm_rate']:.3f}")
df_det = pd.DataFrame(det_rows)
df_det.to_csv(f"{RESULTS}/table3_sensitivity_raw.csv", index=False)

# ---------------- Detection sweep across attack COVERAGE (fixed moderate intensity) ----------------
print("\n=== Detection sweep: attack coverage (reactive vs rebs) ===")
cov_rows = []
for cov in COVERAGES:
    cfg = dict(delta_max=0.06, ddelta_max=0.015, coverage=cov, duration=40, tau_max=4)
    for method in ["reactive", "rebs"]:
        all_records = []
        for seed in SEEDS_SWEEP:
            res = run_day(method, 10, 2, cfg, seed, torch_model=model)
            all_records.extend(res["detect_records"])
        pooled_records[method].extend(all_records)
        dm = _dm(all_records, method)
        cov_rows.append(dict(coverage=cov, method=method, **dm))
        print(f"coverage={cov:.1f} {method:9s} n={dm['n']:3d} acc={dm['accuracy']:.3f} "
              f"prec={dm['precision']:.3f} rec={dm['recall']:.3f} f1={dm['f1']:.3f}")
df_cov = pd.DataFrame(cov_rows)
df_cov.to_csv(f"{RESULTS}/table3b_coverage_raw.csv", index=False)

# Table 3 summary = CORRECTLY pooled confusion matrix across every intensity/coverage
# sweep point and seed collected above (fixes the mean-of-F1 aggregation bug).
print("\n=== Table 3 summary: POOLED confusion matrix (fixes mean-of-F1 bug) ===")
summary3 = pd.DataFrame({m: pooled_detection_metrics(pooled_records[m], m) for m in ["reactive", "rebs"]}).T
summary3.to_csv(f"{RESULTS}/table3_summary.csv")
print(summary3.round(3))
# sanity check: F1 must reconcile with precision/recall exactly now
for m in ["reactive", "rebs"]:
    p, r, f1 = summary3.loc[m, "precision"], summary3.loc[m, "recall"], summary3.loc[m, "f1"]
    recon = 2 * p * r / (p + r)
    print(f"  sanity check {m}: reported F1={f1:.4f} vs recomputed 2PR/(P+R)={recon:.4f}  (should match)")

# ---------------- Table 4: scalability ----------------
print("\n=== Scalability sweep (REB-S) ===")
scale_rows = []
for n in SCALE_SIZES:
    m_bess = max(1, n // 5)
    t0 = time.time()
    res = run_day("rebs", n, m_bess, MODERATE_ATTACK, seed=1, torch_model=model)
    wall = time.time() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    scale_rows.append(dict(
        n_ev=n, n_bess=m_bess,
        mean_solve_time=float(np.mean(res["solve_times"])),
        max_solve_time=float(np.max(res["solve_times"])),
        n_detection_calls=len(res["detect_records"]),
        wall_time_total=wall, peak_rss_mb=peak_rss_mb,
    ))
    print(scale_rows[-1])
df_scale = pd.DataFrame(scale_rows)
df_scale.to_csv(f"{RESULTS}/table4_scalability.csv", index=False)

print("\nAll experiments complete. Results written to ./results/")

# ==================================================================
# Additional experiments added in response to external review round 2
# ==================================================================

# ---------------- SoC-spoofing-ONLY ablation (isolates eq. 13) ----------------
print("\n=== SoC-spoofing-ONLY ablation (tau_max=0, no departure-time attack) ===")
SOC_ONLY_ATTACK = dict(delta_max=0.06, ddelta_max=0.015, coverage=0.4, duration=40, tau_max=0)
SOC_ONLY_VARIANTS = {
    "Unaware (baseline)": None,  # special-cased below: method='unaware'
    "Margin only (no reconstruction)": dict(reconstruction=False, robust_departure=True, risk_margin=True),
    "Reconstruction only (no margin)": dict(reconstruction=True, robust_departure=True, risk_margin=False),
    "Margin + reconstruction (full REB-S)": dict(reconstruction=True, robust_departure=True, risk_margin=True),
}
soc_only_rows = []
for name, ab in SOC_ONLY_VARIANTS.items():
    for seed in SEEDS_SWEEP:
        if name == "Unaware (baseline)":
            res = run_day("unaware", 10, 2, SOC_ONLY_ATTACK, seed, torch_model=model)
        else:
            res = run_day("rebs", 10, 2, SOC_ONLY_ATTACK, seed, torch_model=model, ablation=ab)
        m = compute_metrics(res, name)
        m["seed"] = seed
        soc_only_rows.append(m)
df_soc_only = pd.DataFrame(soc_only_rows)
df_soc_only.to_csv(f"{RESULTS}/soc_only_ablation_raw.csv", index=False)
summary_soc = df_soc_only.groupby("method").agg(
    completion_mean=("completion_rate_pct", "mean"), completion_std=("completion_rate_pct", "std"),
    cost_mean=("total_cost", "mean"), efc_mean=("bess_efc", "mean"),
    violations_mean=("constraint_violations", "mean"),
).reindex(list(SOC_ONLY_VARIANTS.keys()))
summary_soc.to_csv(f"{RESULTS}/soc_only_ablation_summary.csv")
print(summary_soc.round(3))

# ---------------- tau_cap sensitivity sweep ----------------
print("\n=== tau_cap sweep (robust-departure trust bound, moderate joint attack) ===")
TAU_CAPS = [0, 1, 2, 4]  # steps of 0.5h -> 0h, 0.5h, 1h, 2h
tau_rows = []
for tc in TAU_CAPS:
    for seed in SEEDS_SWEEP:
        res = run_day("rebs", 10, 2, MODERATE_ATTACK, seed, torch_model=model, dep_tau_cap=tc)
        m = compute_metrics(res, f"tau_cap={tc}")
        m["seed"] = seed
        m["tau_cap_steps"] = tc
        m["tau_cap_hours"] = tc * 0.5
        tau_rows.append(m)
df_tau = pd.DataFrame(tau_rows)
df_tau.to_csv(f"{RESULTS}/tau_cap_sweep_raw.csv", index=False)
summary_tau = df_tau.groupby("tau_cap_hours").agg(
    completion_mean=("completion_rate_pct", "mean"), completion_std=("completion_rate_pct", "std"),
    cost_mean=("total_cost", "mean"), violations_mean=("constraint_violations", "mean"),
)
summary_tau.to_csv(f"{RESULTS}/tau_cap_sweep_summary.csv")
print(summary_tau.round(3))

# ---------------- Coverage sweep with more seeds (30, per review) ----------------
print("\n=== Detection sweep: attack coverage, 30 seeds (was 10) ===")
SEEDS_COVERAGE = list(range(1, 31))
cov_rows2 = []
for cov in COVERAGES:
    cfg = dict(delta_max=0.06, ddelta_max=0.015, coverage=cov, duration=40, tau_max=4)
    for method in ["reactive", "rebs"]:
        all_records = []
        for seed in SEEDS_COVERAGE:
            res = run_day(method, 10, 2, cfg, seed, torch_model=model)
            all_records.extend(res["detect_records"])
        dm = _dm(all_records, method)
        cov_rows2.append(dict(coverage=cov, method=method, **dm))
        print(f"coverage={cov:.1f} {method:9s} n={dm['n']:4d} acc={dm['accuracy']:.3f} "
              f"prec={dm['precision']:.3f} rec={dm['recall']:.3f} f1={dm['f1']:.3f}")
df_cov2 = pd.DataFrame(cov_rows2)
df_cov2.to_csv(f"{RESULTS}/table3b_coverage_raw.csv", index=False)  # supersedes the 10-seed version

# ---------------- Extended scalability: 100, 160, 200 EVs + measured (not estimated) inference ----------------
print("\n=== Extended scalability sweep (REB-S), with measured detector inference time ===")
from detector import risk_scores as _risk_scores, WINDOW as _WINDOW, FEATS as _FEATS
SCALE_SIZES_EXT = [5, 10, 20, 40, 80, 100, 160, 200]
scale_rows2 = []
for n in SCALE_SIZES_EXT:
    m_bess = max(1, n // 5)
    t0 = time.time()
    res = run_day("rebs", n, m_bess, MODERATE_ATTACK, seed=1, torch_model=model)
    wall = time.time() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # Measured (not estimated) detector inference latency at this batch size:
    n_calls = max(len(res["detect_records"]), 1)
    x_dummy = np.random.randn(n_calls, _WINDOW, _FEATS).astype(np.float32)
    _risk_scores(model, x_dummy)  # warmup
    times = []
    for _ in range(20):
        ti0 = time.time()
        _risk_scores(model, x_dummy)
        times.append(time.time() - ti0)
    measured_inference_ms = float(np.mean(times)) * 1000

    scale_rows2.append(dict(
        n_ev=n, n_bess=m_bess,
        mean_solve_time=float(np.mean(res["solve_times"])),
        max_solve_time=float(np.max(res["solve_times"])),
        n_detection_calls=n_calls,
        measured_inference_ms=measured_inference_ms,
        wall_time_total=wall, peak_rss_mb=peak_rss_mb,
    ))
    print(scale_rows2[-1])
df_scale2 = pd.DataFrame(scale_rows2)
df_scale2.to_csv(f"{RESULTS}/table4_scalability.csv", index=False)  # supersedes the N<=80 version

print("\nAll round-2 experiments complete.")
import platform, sys as _sys
hw_info = dict(python=_sys.version.split()[0], torch=torch.__version__,
               platform=platform.platform(), processor=platform.processor() or "unknown")
try:
    import pulp as _pulp
    hw_info["pulp"] = _pulp.__version__
except Exception:
    pass
with open(f"{RESULTS}/hardware_info.json", "w") as f:
    json.dump(hw_info, f, indent=2)
print("hardware/software info:", hw_info)

