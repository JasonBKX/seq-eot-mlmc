"""Paper tables from saved runs (no new runs).

    python analysis/make_paper_tables.py [RESULTS_DIR]

RESULTS_DIR defaults to ../results. Prints, for the three test cases,
  (1) relative RMSE, relative bias (SE) and mean wall time at the three finest horizons,
  (2) wall time to reach a relative-RMSE target (log-log interpolation between horizons),
  (3) cost per run at the finest horizon: PDE solves, element work, PDE time (part inside MCMC), fit time.
The relative-bias SE combines the spread of the 30 runs with the reference SE.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

csv.field_size_limit(10**9)
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"

METHODS = ["MLSMC-pCN", "pCN + ESS sweeps", "LightSB", "TransportMH"]
PROBLEMS = {
    "1D elliptic": dict(ref="ref_is_noise005", unit=1e-4, targets=[10, 3, 1.5, 1], sources={
        "TransportMH": [("results_paper_T1_noise005", "TransportMH")],
        "pCN + ESS sweeps": [("results_paper_T1_noise005", "MLSMC_pCN")],
        "MLSMC-pCN": [("results_paper_T1_noise005_pcn_plain", "MLSMC_pCN")],
        "LightSB": [("results_paper_T1_noise005", "LightSB")]}),
    "Darcy flow": dict(ref="ref_darcy", unit=1e-3, targets=[30, 20, 10, 6.6], sources={
        "TransportMH": [("results_paper_darcy_L5", "TransportMH"), ("results_paper_darcy_L6", "TransportMH")],
        "pCN + ESS sweeps": [("results_paper_darcy_L5", "MLSMC_pCN"), ("results_paper_darcy_L6", "MLSMC_pCN")],
        "MLSMC-pCN": [("results_paper_darcy_L5_pcn_plain", "MLSMC_pCN"), ("results_paper_darcy_L6_pcn_plain", "MLSMC_pCN")],
        "LightSB": [("results_paper_darcy_L5", "LightSB"), ("results_paper_darcy_L6", "LightSB")]}),
    "Navier-Stokes": dict(ref="ref_ns", unit=1e-3, targets=[30, 20, 15, 10], sources={
        "TransportMH": [("results_ns2d_L4", "TransportMH")],
        "pCN + ESS sweeps": [("results_ns2d_L4", "MLSMC_pCN")],
        "MLSMC-pCN": [("results_ns2d_L4_pcn_plain", "MLSMC_pCN")],
        "LightSB": [("results_ns2d_L4", "LightSB")]}),
}


def num(x):
    return float(x) if x not in (None, "", "None") else float("nan")


def load(sources):
    out = {}
    for folder, method in sources:
        for r in csv.DictReader(open(ROOT / folder / "runs.csv")):
            if r["method"] != method or int(r["L"]) == 0:
                continue
            solves = sum(json.loads(r["fem_solves_by_level"]).values())
            out.setdefault(int(r["L"]), []).append(dict(
                est=float(r["estimate"]), total=float(r["total_seconds"]), solves=solves,
                work=num(r["fem_element_work"]), pde=num(r["pde_seconds"]),
                pde_mcmc=num(r["pde_mcmc_seconds"]), fit=num(r["fit_seconds"])))
    return out


def time_to_target(curve, target):
    """curve: [(L, t, rel)] sorted by L. First crossing, log-log interpolation."""
    for i, (L, t, rel) in enumerate(curve):
        if rel <= target:
            if i == 0:
                return t
            _, t0, r0 = curve[i - 1]
            w = (math.log(target) - math.log(r0)) / (math.log(rel) - math.log(r0))
            return math.exp(math.log(t0) + w * (math.log(t) - math.log(t0)))
    return None


for name, spec in PROBLEMS.items():
    ref = json.loads((ROOT / spec["ref"] / "reference.json").read_text())
    R, Rse, u = ref["value"], ref.get("se") or 0.0, spec["unit"]
    print(f"\n=== {name}: reference {R:.6g} (SE {Rse:.2g}); errors in units of {u:g}")
    curves, cost = {}, {}
    print(f"{'method':18s} {'L':>2s} {'relRMSE':>8s} {'bias':>8s} {'(SE)':>7s} {'time s':>8s}")
    for m in METHODS:
        data = load(spec["sources"][m])
        curve = []
        for L in sorted(data):
            e = np.array([x["est"] for x in data[L]])
            rel = math.sqrt(np.mean((e - R) ** 2)) / abs(R)
            bias = (e.mean() - R) / abs(R)
            se = math.hypot(e.std(ddof=1) / math.sqrt(len(e)), Rse) / abs(R)
            t = float(np.mean([x["total"] for x in data[L]]))
            curve.append((L, t, rel))
            if L > max(data) - 3:
                print(f"{m:18s} {L:2d} {rel / u:8.3g} {bias / u:+8.3g} {se / u:7.2g} {t:8.1f}")
        curves[m] = curve
        cost[m] = {k: float(np.nanmean([x[k] for x in data[max(data)]])) for k in ("solves", "work", "pde", "pde_mcmc", "fit")}
    print(f"\ntime to reach relative RMSE target (x{u:g}); '--' = not reached")
    print(f"{'method':18s} " + " ".join(f"{g:>7g}" for g in spec["targets"]))
    for m in METHODS:
        ts = [time_to_target(curves[m], g * u) for g in spec["targets"]]
        print(f"{m:18s} " + " ".join(f"{t:7.2g}" if t is not None else f"{'--':>7s}" for t in ts))
    print(f"\ncost per run at the finest horizon L={max(L for L, _, _ in curves['TransportMH'])}")
    print(f"{'method':18s} {'solves':>10s} {'work':>10s} {'PDE s':>8s} {'(MCMC)':>8s} {'fit s':>6s}")
    for m in METHODS:
        c = cost[m]
        print(f"{m:18s} {c['solves']:10.3g} {c['work']:10.3g} {c['pde']:8.1f} {c['pde_mcmc']:8.1f} {c['fit']:6.1f}")
