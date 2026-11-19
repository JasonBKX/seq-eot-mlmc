#!/usr/bin/env python3
"""Write configs whose per-level sample sizes follow Beskos et al. (2017), Sec. 3.3:

    N_l = ceil( c * eps^{-2} * K_L * h_l^{(beta+zeta)/2} ),
    K_L = sum_{l=0}^{L} h_l^{(beta-zeta)/2},   eps^2 ~ h_L^{2 alpha}   (MSE = bias^2 + var).

Elliptic example: h_l = 2^{-(l+3)}, zeta = 1 (linear-cost solve), alpha ~ 1,
beta = 2 (theory; the paper reports similar results with its empirical rates).
Constant factors (the 2^{-3} in h_l, c) are absorbed into one scale chosen so
that N_0 at the LARGEST horizon equals --n0-at-lmax. A floor --n-min keeps the
whitening (needs N > d = 50) well defined; floors are reported.

Usage (from the repository root):
    python make_beskos_allocation.py --base configs/config_paper_T1_noise005.json \
        --out my_config.json
"""
import argparse
import json
import math
from pathlib import Path


def allocation(L, beta, zeta, alpha, scale, n_min):
    K = sum(2.0 ** (-l * (beta - zeta) / 2) for l in range(L + 1))
    eps_inv2 = 2.0 ** (2 * alpha * L)
    raw = [scale * eps_inv2 * K * 2.0 ** (-l * (beta + zeta) / 2) for l in range(L + 1)]
    return [max(n_min, math.ceil(v)) for v in raw], raw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-level", type=int, default=5)
    p.add_argument("--min-level", type=int, default=1, help="smallest horizon L (0 = level-0 only)")
    p.add_argument("--solver", help="override FEM solver, e.g. banded (dense is limited to L<=6)")
    p.add_argument("--initial-max-proposals", type=int)
    p.add_argument("--n0-at-lmax", type=int, default=40000)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n-min", type=int, default=100)
    p.add_argument("--repeats", type=int)
    args = p.parse_args()

    cfg = json.loads(args.base.read_text())
    Lmax = args.max_level
    K = sum(2.0 ** (-l * (args.beta - args.zeta) / 2) for l in range(Lmax + 1))
    scale = args.n0_at_lmax / (2.0 ** (2 * args.alpha * Lmax) * K)
    configs = []
    for L in range(args.min_level, Lmax + 1):
        N, raw = allocation(L, args.beta, args.zeta, args.alpha, scale, args.n_min)
        floored = [l for l, (n, r) in enumerate(zip(N, raw)) if n > math.ceil(r)]
        configs.append({"name": f"L{L}", "N": N, "smc_N": max(N)})
        print(f"L{L}: N = {N}" + (f"   (floor {args.n_min} applied at levels {floored})" if floored else ""))
    cfg["configurations"] = configs
    cfg["figure5_configuration"] = f"L{Lmax}"
    cfg["allocation"] = {"rule": "Beskos et al. 2017 Sec 3.3: N_l ~ eps^-2 K_L h_l^((beta+zeta)/2), eps^2 ~ h_L^(2 alpha)",
                         "beta": args.beta, "zeta": args.zeta, "alpha": args.alpha,
                         "n0_at_lmax": args.n0_at_lmax, "n_min": args.n_min}
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.solver:
        cfg["solver"] = args.solver
    if args.initial_max_proposals:
        cfg["initial_max_proposals"] = args.initial_max_proposals
    cfg["figure4"] = dict(cfg.get("figure4", {}), levels=list(range(1, Lmax + 1)))
    args.out.write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
