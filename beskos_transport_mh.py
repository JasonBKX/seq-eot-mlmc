#!/usr/bin/env python3
"""Multilevel estimation with entropic optimal transport (light Schroedinger bridges) and MCMC.

Driver for the numerical experiments of
  J. Beh, P. Del Moral, A. Jasra, "Sequential entropic optimal transport for Bayesian inverse
  problems via multilevel Monte Carlo".

Problems (config key "problem"): "beskos" (1D elliptic, test case 1, Beskos et al. 2017),
"darcy2d" (2D Darcy flow, test case 2, darcy2d.py), "ns2d" (2D Navier--Stokes, test case 3, ns2d.py).

Methods (config key "methods"):
  LightSB     : LightSB transport pi_{l-1} -> pi_l fitted at each level (whitened coordinates,
                level-dependent epsilon); paired estimator mean[phi(Y) - phi(X)].
  TransportMH : the same transport followed by pCN Metropolis--Hastings sweeps targeting pi_l
                (config "transport_mh"); paired estimator. With "coupled": true the sweeps are run
                on both members of each pair with common proposal noise and acceptance uniforms.
  MLSMC_pCN   : multilevel SMC (Beskos et al. 2017, eq. (5)) with pCN-MH mutation; with
                "mlsmc_kernel": {"ess_sweeps": ...} the number of sweeps adapts to the ESS.
  MLSMC, MLSMC_LSB : further MLSMC variants (blocked random-walk, LightSB-MH mutation), not
                used in the paper.

Usage (from the repository root):
  python beskos_transport_mh.py --config configs/<config>.json --out <dir> [--reference-file ref.json]
  python beskos_transport_mh.py --config configs/<config>.json --reference-only --out <dir>
Each run writes <dir>/runs.csv (one row per repeat, method and horizon), <dir>/records/*.json and
<dir>/metadata.json. PDE solves are counted and timed separately (pde_seconds, pde_mcmc_seconds);
fit_seconds is the LightSB fitting time; total_seconds excludes the shared level-0 initialisation.
The level-0 pool is cached in the folder named by the config key "initial_cache".
All computations run on the CPU; PDE solves and the observable use float64.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import scipy
from scipy.linalg import solveh_banded
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beskos_elliptic import BeskosElliptic, blocked_rwm_reference
import darcy2d
import ns2d

SCHEMA = "beskos-transport-mh-v1"
FEM_SOLVERS = ("banded", "flux", "thomas", "dense")
DENSE_MAX_ELEMENTS = 512
THOMAS_MAX_ELEMENTS = 32768
LABELS = {"LightSB": "LightSB paired", "MLSMC": "MLSMC (reweighted, blocked RWM)", "SMC": "SMC (final mean)",
          "TransportMH": "LightSB transport + pCN-MH (paired)",
          "MLSMC_pCN": "MLSMC + pCN-MH mutation", "MLSMC_LSB": "MLSMC + LightSB-MH mutation"}
METHOD_CODE = {"LightSB": 11, "MLSMC": 22, "SMC": 33, "TransportMH": 44, "MLSMC_pCN": 55, "MLSMC_LSB": 66}


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


# All are seconds. Some inclusive diagnostics overlap; see the partition below.
PROFILE_COLUMNS = (
    "pde_seconds", "pde_reweight_seconds", "pde_mcmc_seconds",
    "fit_seconds", "propagation_seconds", "resampling_seconds",
    "mcmc_total_seconds", "mcmc_other_seconds", "sampler_other_seconds",
)
TIMING_PROFILE_VERSION = 1


class Timers:
    """Nested CPU wall-clock timers; no change to RNG state or sampled values."""
    def __init__(self):
        self.values = Counter()

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] += time.perf_counter() - start

    def breakdown(self, since=None):
        """Return a non-overlapping sampler partition plus inclusive diagnostics.

        sampler = pde + fit + propagation + resampling + mcmc_other + other.
        pde = pde_reweight + pde_mcmc.
        mcmc_total INCLUDES pde_mcmc, so is NOT an extra partition component.
        Call only AFTER the measured contexts have closed.
        """
        since = {} if since is None else since
        def elapsed(name):
            return float(self.values.get(name, 0.0) - since.get(name, 0.0))
        pde_reweight = elapsed("pde_reweight")
        pde_mcmc = elapsed("pde_mcmc")
        pde = pde_reweight + pde_mcmc
        fit = elapsed("fit")
        propagation = elapsed("propagation")
        resampling = elapsed("resampling")
        mcmc_total = elapsed("mcmc")
        mcmc_other = max(0.0, mcmc_total - pde_mcmc)
        accounted = pde + fit + propagation + resampling + mcmc_other
        other = max(0.0, elapsed("sampler") - accounted)
        return {
            "pde_seconds": pde,
            "pde_reweight_seconds": pde_reweight,
            "pde_mcmc_seconds": pde_mcmc,
            "fit_seconds": fit,
            "propagation_seconds": propagation,
            "resampling_seconds": resampling,
            "mcmc_total_seconds": mcmc_total,
            "mcmc_other_seconds": mcmc_other,
            "sampler_other_seconds": other,
        }

    def stage_breakdown(self, before):
        return {
            "sampler_seconds": float(self.values.get("sampler", 0.0) - before.get("sampler", 0.0)),
            "estimator_seconds": float(self.values.get("estimator", 0.0) - before.get("estimator", 0.0)),
            **self.breakdown(before),
        }


def print_timing_breakdown(result):
    """Printed outside all benchmark timers and after the result is recorded."""
    if any(result.get(key) is None for key in PROFILE_COLUMNS):
        print("  Detailed sampler timings unavailable in this record.", flush=True)
        return
    print(
        "  Sampler partition (already included in sampler_seconds):\n"
        f"    PDE log-density calls:      {result['pde_seconds']:10.6f} s\n"
        f"    Model initialization/fit:   {result['fit_seconds']:10.6f} s\n"
        f"    LightSB parent/child draw:  {result['propagation_seconds']:10.6f} s\n"
        f"    Weighted resampling:        {result['resampling_seconds']:10.6f} s\n"
        f"    MCMC excluding density:     {result['mcmc_other_seconds']:10.6f} s\n"
        f"    Other sampler work:         {result['sampler_other_seconds']:10.6f} s\n"
        f"    SAMPLER TOTAL:              {result['sampler_seconds']:10.6f} s",
        flush=True,
    )


class MeasuredBeskos(BeskosElliptic):
    """Same FEM hierarchy with selectable linear solvers and actual counters.

    banded: SciPy SPD tridiagonal solve; the default efficient assembled solver.
    flux: original algebraically equivalent vectorized flux-elimination solve.
    thomas: conventional tridiagonal assembly, Python loops over unknowns,
        vectorized over bounded parameter batches. No dense matrices.
    dense: conventional dense assembly plus torch.linalg.solve, one matrix per
        parameter sample, restricted to <=512 elements to prevent large builds.

    Exact element-average coefficients and the exact load are shared. The
    benchmark PDE counters are incremented once per parameter vector, regardless
    of backend. All solver work is within the existing density-call sub-timers.

    Data generation at h=2^-20 still uses the ORIGINAL streaming base solver,
    and the common Figure 4 diagnostic always uses assembled banded FEM. Both
    are outside the benchmark timers. Continuum phi/reference are unchanged.
    """
    def __init__(self, dimension=50, y=None, seed=2026, solver="banded",
                 parameter_batch=128, reference_order=64, noise_sd=0.25):
        if solver not in FEM_SOLVERS:
            raise ValueError(f"solver must be one of {FEM_SOLVERS}")
        self.solver = solver
        self._initializing = True
        self._tracking = False
        self._band_basis_cache = {}
        self._max_coefficient_entries = 1_048_576
        super().__init__(dimension=dimension, y=y, seed=seed, device="cpu",
                         parameter_batch=parameter_batch, reference_order=reference_order,
                         noise_sd=noise_sd)
        self._initializing = False
        self.reset_work()

    def reset_work(self):
        self._fem_counts = Counter()
        self._continuum_count = 0
        self._quad_points = 0
        self._tracking = True

    def work(self):
        counts = {str(k): int(v) for k, v in sorted(self._fem_counts.items())}
        return {"fem_solves_by_level": counts,
                "fem_solves_raw": sum(counts.values()),
                # One COMMON unit across all configurations; do NOT divide by 2^L.
                "fem_element_work": sum(int(v) * 2 ** (int(k) + 3) for k, v in counts.items()),
                "continuum_forward_evals": int(self._continuum_count),
                "continuum_quadrature_points": int(self._quad_points)}

    def _averages(self, batch, n):
        """Bound memory even on the very fine data/reference meshes."""
        if n <= self.element_chunk:
            if n not in self._band_basis_cache:
                self._band_basis_cache[n] = self._element_data(n)[0]
            return self.base_diffusion + batch @ self._band_basis_cache[n].T
        a = torch.empty((len(batch), n), dtype=torch.float64)
        for start in range(0, n, self.element_chunk):
            end = min(start + self.element_chunk, n)
            basis, _ = self._element_data(n, start, end)
            a[:, start:end] = self.base_diffusion + batch @ basis.T
        return a

    @torch.no_grad()
    def _banded_solution(self, u, level, full=False):
        ub, shape = self._physical_batch(u)
        n = self._mesh_size(level)
        if n > 2 ** 20:
            raise ValueError("This driver limits FEM meshes to 2^20 elements")
        h = 1.0 / n
        count = n + 1 if full else 3
        result = np.empty((len(ub), count), dtype=np.float64)
        rhs = 100.0 * h * h * np.arange(1, n, dtype=np.float64)
        bsize = min(self.parameter_batch, max(1, self._max_coefficient_entries // n))
        q_indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]
        for start in range(0, len(ub), bsize):
            batch = ub[start:start + bsize]
            averages = self._averages(batch, n).numpy()
            if not np.isfinite(averages).all() or np.any(averages <= 0):
                raise FloatingPointError("Nonpositive/nonfinite element diffusion")
            for j, a in enumerate(averages):
                conductance = a / h
                ab = np.zeros((2, n - 1), dtype=np.float64)
                # Lower-band storage: ab[0,j]=A[j,j]; ab[1,j]=A[j+1,j].
                ab[0] = conductance[:-1] + conductance[1:]
                ab[1, :-1] = -conductance[1:-1]
                interior = solveh_banded(ab, rhs, lower=True, overwrite_ab=True,
                                         overwrite_b=False, check_finite=False)
                if full:
                    result[start + j, 0] = result[start + j, -1] = 0.0
                    result[start + j, 1:-1] = interior
                else:
                    result[start + j] = interior[q_indices]
        return torch.from_numpy(result).reshape(*shape, count)

    @torch.no_grad()
    def _thomas_solution(self, u, level, full=False):
        """Conventional assembled tridiagonal FEM, with a Python-loop solver.

        O(B*n) working memory and O(B*n) arithmetic per batch, unlike dense
        O(n**2) storage/O(n**3) solution. Python-loop overhead is deliberately
        NOT compiled away, but no artificial delay is introduced. All physical
        input shapes [..., dimension] are restored to [..., 3] or [..., n+1].
        """
        ub, shape = self._physical_batch(u)
        n = self._mesh_size(level)
        if n > THOMAS_MAX_ELEMENTS:
            raise ValueError(
                f"thomas is limited to {THOMAS_MAX_ELEMENTS} elements to avoid "
                "very long Python loops. Use solver='banded' or 'flux'.")
        h = 1.0 / n
        n_int = n - 1
        count = n + 1 if full else 3
        result = torch.empty((len(ub), count), dtype=torch.float64)
        # Exact integral of 100*x against interior hat function i.
        load = 100.0 * h * h * torch.arange(1, n, dtype=torch.float64)
        bsize = min(self.parameter_batch, max(1, self._max_coefficient_entries // n))
        q_indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]
        for start in range(0, len(ub), bsize):
            a = self._averages(ub[start:start + bsize], n)
            if not torch.isfinite(a).all() or (a <= 0).any():
                raise FloatingPointError("Nonpositive/nonfinite element diffusion")
            conductance = a / h
            # Zero Dirichlet boundaries eliminated; off diagonal is symmetric.
            diagonal = conductance[:, :-1] + conductance[:, 1:]
            off = -conductance[:, 1:-1]
            rhs = load.unsqueeze(0).expand(len(a), n_int).clone()
            # Forward elimination, batched over physical parameter samples.
            for i in range(1, n_int):
                multiplier = off[:, i - 1] / diagonal[:, i - 1]
                diagonal[:, i] -= multiplier * off[:, i - 1]
                rhs[:, i] -= multiplier * rhs[:, i - 1]
            if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
                raise FloatingPointError("Nonpositive/nonfinite Thomas pivot")
            interior = torch.empty_like(rhs)
            interior[:, -1] = rhs[:, -1] / diagonal[:, -1]
            for i in range(n_int - 2, -1, -1):
                interior[:, i] = (rhs[:, i] - off[:, i] * interior[:, i + 1]) / diagonal[:, i]
            if not torch.isfinite(interior).all():
                raise FloatingPointError("Nonfinite Thomas solution")
            end = start + len(a)
            if full:
                result[start:end, 0] = result[start:end, -1] = 0.0
                result[start:end, 1:-1] = interior
            else:
                result[start:end] = interior[:, q_indices]
        return result.reshape(*shape, count)

    @torch.no_grad()
    def _dense_solution(self, u, level, full=False):
        """Dense FEM verification backend, not the efficient 1D cost model.

        Solves one (n-1)-square matrix at a time in CPU float64. Refuses large
        meshes BEFORE allocating a dense matrix. The data-generation mesh is
        never routed here by fem_values().
        """
        n = self._mesh_size(level)
        if n > DENSE_MAX_ELEMENTS:
            raise ValueError(
                f"dense is restricted to <= {DENSE_MAX_ELEMENTS} elements "
                f"(L<=6); requested {n}. Use solver='thomas' or 'banded'.")
        ub, shape = self._physical_batch(u)
        h = 1.0 / n
        count = n + 1 if full else 3
        result = torch.empty((len(ub), count), dtype=torch.float64)
        load = 100.0 * h * h * torch.arange(1, n, dtype=torch.float64)
        q_indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]
        bsize = min(self.parameter_batch, max(1, self._max_coefficient_entries // n))
        for start in range(0, len(ub), bsize):
            averages = self._averages(ub[start:start + bsize], n)
            if not torch.isfinite(averages).all() or (averages <= 0).any():
                raise FloatingPointError("Nonpositive/nonfinite element diffusion")
            for j, a in enumerate(averages):
                k = a / h
                off = -k[1:-1]
                matrix = torch.diag(k[:-1] + k[1:])
                matrix += torch.diag(off, diagonal=1)
                matrix += torch.diag(off, diagonal=-1)
                interior = torch.linalg.solve(matrix, load)
                if not torch.isfinite(interior).all():
                    raise FloatingPointError("Nonfinite dense solution")
                if full:
                    result[start + j, 0] = result[start + j, -1] = 0.0
                    result[start + j, 1:-1] = interior
                else:
                    result[start + j] = interior[q_indices]
        return result.reshape(*shape, count)

    @torch.no_grad()
    def sample_posterior(self, *args, **kwargs):
        """Untimed rejection initialisation through the vectorised flux solver.

        All backends return the same FEM values (tests assert agreement to 1e-10),
        so this changes only wall time OUTSIDE the benchmark timers, never the
        samples' distribution. Work counters are not touched (reset per method).
        """
        previous, batch = self._initializing, self.parameter_batch
        self._initializing = True
        self.parameter_batch = max(batch, 8192)   # fewer Python-level batches
        try:
            return BeskosElliptic.sample_posterior(self, *args, **kwargs)
        finally:
            self._initializing, self.parameter_batch = previous, batch

    def fem_values(self, u, level):
        if self._initializing:
            # Preserve the original synthetic data, independent of chosen timed solver.
            return BeskosElliptic.fem_values(self, u, level)
        if self._tracking:
            self._fem_counts[int(level)] += int(torch.as_tensor(u).numel() // self.dimension)
        if self.solver == "flux":
            return BeskosElliptic.fem_values(self, u, level)
        if self.solver == "thomas":
            return self._thomas_solution(u, level)
        if self.solver == "dense":
            return self._dense_solution(u, level)
        return self._banded_solution(u, level)

    @torch.no_grad()
    def fem_nodes(self, u, level):
        """Full nodal FEM solution, for the PDE-only Figure 4 diagnostic."""
        # Always assembled FEM for this validation diagnostic, outside timers.
        n = self._mesh_size(level)
        return torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64), self._banded_solution(u, level, full=True)

    problem_name = "beskos"

    def continuum_loglik_phi(self, u):
        """(log-likelihood under the continuum model, observable) for physical u [B, d]."""
        values = self.continuum_values(u)
        return -0.5 * ((values[:, [0, 2]] - self.y) / self.noise_sd).square().sum(-1), values[:, 1]

    def mode_label(self, z):
        return None

    def stored_y(self):
        return self.y.tolist()

    @torch.no_grad()
    def continuum_values(self, u, order=None):
        order = self.reference_order if order is None else int(order)
        if self._tracking:
            n = int(torch.as_tensor(u).numel() // self.dimension)
            self._continuum_count += n
            self._quad_points += n * 4 * order
        return BeskosElliptic.continuum_values(self, u, order)


SOURCE_DEFAULTS = {"field_scale": 0.3, "xi_true": 0.3, "noise_rel": 0.01, "noise_weak": 0.2,
                   "start_elements": 8, "weak_separation": 1.5, "weak_noise_free": True,
                   "weak_sensor_x": 0.25}


class SourceMomentProblem(MeasuredBeskos):
    """Bimodal source-location inverse problem (temporal moments, steady elliptic PDE).

    Unknowns u in [-1,1]^(1+50): u[0] -> source position xi = (u[0]+1)/2, u[1:] the Beskos
    coefficient field k(x) = 0.15 + field_scale * sum_j 0.4*4^-j u_j b_j(x). Every Beskos basis
    function is symmetric about x = 1/2, so k(x) = k(1-x) for every u.
    Forward model: zeroth temporal moment (total exposure) of an instantaneous unit release,
        -(k m0')' = delta(x - xi),  m0(0) = m0(1) = 0,
    observed as  [m0(1/2),  m0(1/4) + m0(3/4) (composite of two symmetric wells),  m0(x_w)],
    x_w = weak_sensor_x (default 1/4). Off-node sensors use the piecewise-linear FEM solution
    (linear interpolation of nodal values). A weak sensor placed at the source position
    (x_w = xi_true, not a dyadic node) sits in the source's element on coarse meshes, where the
    kink of m0 is cut off (O(h) error): coarse levels then misjudge the mode weights, which
    converge to the continuum odds over the first few levels ("mesh-sensitive" variant).
    The first two are invariant under xi -> 1 - xi (mirror modes); the third tilts the mode
    weights. With weak_separation = D its noise sd is |m0(1/4; xi) - m0(1/4; 1-xi)| / D (at the
    true field), and with weak_noise_free its reading is the clean value, so the mode odds are
    about exp(D^2/2) by design (D = 1.5 -> ~75/25) instead of depending on one noise draw.
    weak_separation = None falls back to a relative sd noise_weak. Continuum closed form with R(a,b) = int_a^b 1/k:
        m0(x) = R(0, min(x,xi)) R(max(x,xi), 1) / R(0,1).
    FEM level l: 8 * 2^l linear elements, exact element-average coefficient, exact
    point-source load phi_i(xi); one O(n) tridiagonal solve (flux elimination) per sample.
    Observable: phi(u) = xi (identical at every level). The 'solver' option is ignored:
    all backends of the Beskos driver solve the same tridiagonal system.
    """
    problem_name = "source_moments"

    def __init__(self, y=None, seed=2026, solver="banded", parameter_batch=128,
                 reference_order=64, field_scale=0.3, xi_true=0.3, noise_rel=0.01,
                 noise_weak=0.2, start_elements=8, weak_separation=1.5, weak_noise_free=True,
                 weak_sensor_x=0.25):
        if not (0 < field_scale * 0.4 / 3 < 0.15 and 0 < xi_true < 1 and xi_true != 0.5 and noise_rel > 0
                and noise_weak > 0 and start_elements >= 4 and start_elements % 4 == 0
                and (weak_separation is None or weak_separation > 0) and isinstance(weak_noise_free, bool)
                and 0 < weak_sensor_x < 1 and weak_sensor_x != 0.5):
            raise ValueError("Invalid source-problem settings")
        # Build the 50-mode Beskos field machinery with a dummy y, then re-dimension.
        super().__init__(dimension=50, y=[0.0, 0.0], seed=seed, solver=solver,
                         parameter_batch=parameter_batch, reference_order=reference_order)
        self.field_dim = 50
        self.dimension = 51
        self.sigma = self.sigma * float(field_scale)
        self._fem_cache, self._quad_cache, self._band_basis_cache = {}, {}, {}
        self.settings = {"field_scale": float(field_scale), "xi_true": float(xi_true),
                         "noise_rel": float(noise_rel), "noise_weak": float(noise_weak),
                         "start_elements": int(start_elements),
                         "weak_separation": None if weak_separation is None else float(weak_separation),
                         "weak_noise_free": bool(weak_noise_free)}
        self.weak_x = float(weak_sensor_x)
        if self.weak_x != 0.25:
            # Only recorded when non-default, so earlier targets keep their hashes.
            self.settings["weak_sensor_x"] = self.weak_x
        self.start_elements = int(start_elements)
        if y is None:
            g = torch.Generator().manual_seed(int(seed))
            field = 2.0 * torch.rand(50, generator=g, dtype=torch.float64) - 1.0
            self.u_true = torch.cat([torch.tensor([2.0 * xi_true - 1.0], dtype=torch.float64), field])
            clean = self.continuum_values(self.u_true[None], order=256)[0]
            mirror_u = self.u_true.clone(); mirror_u[0] = -mirror_u[0]
            mirror = self.continuum_values(mirror_u[None], order=256)[0]
            self.noiseless_y = clean
            weak_sd = (noise_weak * clean[2] if weak_separation is None
                       else (clean[2] - mirror[2]).abs() / float(weak_separation))
            self.noise = torch.stack([noise_rel * clean[0], noise_rel * clean[1], weak_sd])
            self.y = clean + self.noise * torch.randn(3, generator=g, dtype=torch.float64)
            if weak_noise_free:
                self.y[2] = clean[2]
        else:
            y = torch.as_tensor(y, dtype=torch.float64)
            if y.shape != (6,):
                raise ValueError("source problem y must be [y1, y2, y3, sd1, sd2, sd3]")
            self.y, self.noise = y[:3].clone(), y[3:].clone()
        self.noise_sd = self.noise.tolist()
        self.reset_work()

    # ---- description/data plumbing ----------------------------------------------------
    def description(self):
        weak = "m0(1/4)" if self.weak_x == 0.25 else f"m0({self.weak_x!r})"
        return {"problem": self.problem_name, "source_settings": self.settings,
                "observations": "m0(1/2), m0(1/4)+m0(3/4), " + weak, "observable": "xi"}

    def stored_y(self):
        """y as stored in metadata: observations followed by their noise sds."""
        return self.y.tolist() + self.noise.tolist()

    def mode_label(self, z):
        """True for the right mode (xi > 1/2 <=> z[0] > 0)."""
        return torch.as_tensor(z)[..., 0] > 0

    # ---- forward models -------------------------------------------------------------------
    def _mesh_size(self, level):
        if not isinstance(level, int) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        return self.start_elements * 2 ** level

    def _field_averages(self, n):
        if n not in self._band_basis_cache:
            e = torch.arange(n, dtype=torch.float64)
            h = 1.0 / n
            self._band_basis_cache[n] = self._basis((e + 0.5) * h) * torch.sinc(self.mode * h / 2.0)
        return self._band_basis_cache[n]

    @torch.no_grad()
    def _fem_m0(self, ub, level):
        """Nodal m0 at x_1..x_n (x_n = 1 is ~0) for physical parameters ub [B, 51]."""
        n = self._mesh_size(level)
        h = 1.0 / n
        basis = self._field_averages(n)
        e = torch.arange(n, dtype=torch.float64)
        out = []
        bsize = max(1, min(self.parameter_batch, 2_000_000 // n))
        for start in range(0, len(ub), bsize):
            b = ub[start:start + bsize]
            res = h / (0.15 + b[:, 1:] @ basis.T)
            if not torch.isfinite(res).all() or (res <= 0).any():
                raise FloatingPointError("Nonpositive/nonfinite diffusion coefficient")
            pos = (b[:, 0] + 1.0) / 2.0 * n
            k = torch.floor(pos)
            t = pos - k
            bk = torch.where(k >= 1, 1.0 - t, torch.zeros_like(t))
            bk1 = torch.where(k + 1 <= n - 1, t, torch.zeros_like(t))
            Q = (e[None] >= k[:, None]).double() * bk[:, None] + (e[None] >= k[:, None] + 1).double() * bk1[:, None]
            F0 = (res * Q).sum(-1) / res.sum(-1)
            out.append(torch.cumsum(res * (F0[:, None] - Q), dim=-1))
        if not out:
            return ub.new_empty((0, n))
        return torch.cat(out)

    @torch.no_grad()
    def fem_values(self, u, level):
        ub, shape = self._physical_batch(u)
        if self._tracking and not self._initializing:
            self._fem_counts[int(level)] += len(ub)
        n = self._mesh_size(level)
        m0 = self._fem_m0(ub, level)
        a, b, c = n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1
        obs = torch.stack([m0[:, b], m0[:, a] + m0[:, c], self._fem_point(m0, n, self.weak_x)], dim=-1)
        return obs.reshape(*shape, 3)

    @staticmethod
    def _fem_point(m0, n, x):
        """Piecewise-linear FEM solution at x from nodal values m0[:, i] = m0(x_{i+1})."""
        p = x * n
        i = int(math.floor(p))
        a = p - i
        if a == 0.0:
            return m0[:, i - 1]                      # node: identical to the original indexing
        left = m0[:, i - 1] if i >= 1 else torch.zeros_like(m0[:, 0])
        right = m0[:, i] if i <= n - 2 else torch.zeros_like(m0[:, 0])
        return (1.0 - a) * left + a * right

    @torch.no_grad()
    def fem_nodes(self, u, level):
        ub, _ = self._physical_batch(u)
        n = self._mesh_size(level)
        m0 = self._fem_m0(ub[:1], level)[0]
        return (torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64),
                torch.cat([m0.new_zeros(1), m0[:-1], m0.new_zeros(1)]))

    @torch.no_grad()
    def continuum_values(self, u, order=None):
        """Closed-form m0 observations; R integrals by Gauss-Legendre on [0,1/4],...,[3/4,1],
        on [0, xi] and, for a non-default weak sensor, on [0, x_w] (order nodes each; 1/k is analytic)."""
        ub, shape = self._physical_batch(u)
        order = self.reference_order if order is None else int(order)
        if self._tracking and not self._initializing:
            self._continuum_count += len(ub)
            self._quad_points += len(ub) * 5 * order
        t, w = np.polynomial.legendre.leggauss(order)
        t = torch.as_tensor(t, dtype=torch.float64)
        w = torch.as_tensor(w, dtype=torch.float64)
        quarter_x = (torch.arange(4, dtype=torch.float64)[:, None] / 4.0 + (t[None] + 1.0) / 8.0).reshape(-1)
        quarter_w = (w / 8.0).repeat(4)
        quarter_basis = self._basis(quarter_x)
        weak_w = weak_basis = None
        if self.weak_x != 0.25:                                                  # R(0, x_w) nodes
            weak_basis = self._basis(self.weak_x * (t + 1.0) / 2.0)
            weak_w = self.weak_x / 2.0 * w
        out = []
        for start in range(0, len(ub), self.parameter_batch):
            b = ub[start:start + self.parameter_batch]
            xi = (b[:, 0] + 1.0) / 2.0
            inv_k = quarter_w / (0.15 + b[:, 1:] @ quarter_basis.T)
            Rq = torch.cumsum(inv_k.reshape(-1, 4, order).sum(-1), dim=-1)    # R(0,1/4..1)
            xs = xi[:, None] * (t[None] + 1.0) / 2.0                             # [B, order]
            ph = math.pi * xs[..., None] * self.mode
            bx = torch.where(self.odd, ph.sin(), ph.cos()) * self.sigma          # [B, order, 50]
            kx = 0.15 + (bx * b[:, None, 1:]).sum(-1)
            Rxi = (xi[:, None] / 2.0 * w[None] / kx).sum(-1)                     # R(0, xi)
            R1 = Rq[:, 3]
            def m0_at(R0x, x):
                left = xi >= x                                                    # sensor left of source
                return torch.where(left, R0x * (R1 - Rxi), Rxi * (R1 - R0x)) / R1
            m_q, m_h, m_3q = m0_at(Rq[:, 0], 0.25), m0_at(Rq[:, 1], 0.5), m0_at(Rq[:, 2], 0.75)
            if self.weak_x == 0.25:
                m_w = m_q
            else:
                Rw = (weak_w / (0.15 + b[:, 1:] @ weak_basis.T)).sum(-1)
                m_w = m0_at(Rw, self.weak_x)
            out.append(torch.stack([m_h, m_q + m_3q, m_w], dim=-1))
        if not out:
            return ub.new_empty((*shape, 3))
        return torch.cat(out).reshape(*shape, 3)

    # ---- densities and observable -------------------------------------------------------
    @torch.no_grad()
    def log_density_u(self, u, level):
        ub, shape = self._batch(u)
        inside = (ub.abs() <= 1.0).all(-1)
        result = ub.new_full((len(ub),), -torch.inf)
        if inside.any():
            values = self.continuum_values(ub[inside]) if level is None else self.fem_values(ub[inside], level)
            result[inside] = -0.5 * ((values - self.y) / self.noise).square().sum(-1)
        return result.reshape(shape)

    def continuum_loglik_phi(self, u):
        values = self.continuum_values(u)
        return -0.5 * ((values - self.y) / self.noise).square().sum(-1), (u[:, 0] + 1.0) / 2.0

    @torch.no_grad()
    def phi_u(self, u):
        return (torch.as_tensor(u, dtype=torch.float64)[..., 0] + 1.0) / 2.0


class Darcy2DProblem:
    """2D Darcy flow posterior hierarchy (see darcy2d.py) behind the driver's problem interface.

    Coordinates: z = KL coefficients with a N(0, I) prior, so LightSB/pCN work on z directly
    (u = z, no transform). log_density_z = log prior + log likelihood at mesh level l.
    PDE work is counted in elements, 2 n_l^2 with n_l = n0 2^l, and wall time is measured.
    The initial pool is drawn by tempered SMC at level 0 (darcy2d.tempered_level0_pool); prior
    rejection would accept essentially nothing for this likelihood.
    Observable phi(z): mean permeability over [1/4, 3/4]^2 (no PDE, the same at every level).
    """
    problem_name = "darcy2d"

    def __init__(self, y=None, seed=2026, parameter_batch=128, **settings):
        opts = dict(darcy2d.DARCY_DEFAULTS)
        unknown = set(settings) - set(opts)
        if unknown:
            raise ValueError(f"unknown darcy settings {sorted(unknown)}")
        opts.update(settings)
        yy = None if y is None else np.asarray(y, dtype=float)
        self.core = darcy2d.DarcyCore(seed=seed, y=yy, **opts)
        self.settings = self.core.settings
        self.dimension = self.core.d
        self.y = torch.as_tensor(self.core.y, dtype=torch.float64)
        self.noise_sd = self.core.noise_sd
        self.parameter_batch = int(parameter_batch)
        self.reference_order = 0
        self._initializing = False
        self.reset_work()

    def reset_work(self):
        self._fem_counts = Counter()
        self._tracking = True

    def work(self):
        counts = {str(k): int(v) for k, v in sorted(self._fem_counts.items())}
        return {"fem_solves_by_level": counts,
                "fem_solves_raw": sum(counts.values()),
                "fem_element_work": sum(int(v) * self.core.elements(int(k)) for k, v in counts.items()),
                "continuum_forward_evals": 0, "continuum_quadrature_points": 0}

    def description(self):
        return {"problem": self.problem_name, "darcy_settings": self.settings,
                "data_level": self.core.data_level,
                "observations": f"pressure at a {self.settings['sensors']}x{self.settings['sensors']} sensor grid",
                "observable": "mean permeability over [1/4,3/4]^2",
                "posterior_reference": "level-L posterior (no continuum model)"}

    def stored_y(self):
        return self.core.y.tolist()

    def mode_label(self, z):
        return None

    def _np(self, z):
        zb = torch.as_tensor(z, dtype=torch.float64)
        shape = zb.shape[:-1]
        return zb.reshape(-1, self.dimension).numpy(), shape

    @torch.no_grad()
    def fem_values(self, u, level):
        Z, shape = self._np(u)
        if self._tracking and not self._initializing:
            self._fem_counts[int(level)] += len(Z)
        out = self.core.level(int(level)).sensor_values(Z) if len(Z) else np.empty((0, len(self.core.y)))
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(*shape, -1)

    @torch.no_grad()
    def log_density_z(self, z, level):
        if level is None:
            raise ValueError("darcy2d has no continuum model; use a finite level")
        Z, shape = self._np(z)
        if not np.isfinite(Z).all():
            raise ValueError("Parameters must be finite")
        values = self.fem_values(torch.from_numpy(Z), level).numpy()
        r = (values - self.core.y) / self.noise_sd
        out = -0.5 * np.sum(Z * Z, -1) - 0.5 * np.sum(r * r, -1)
        return torch.from_numpy(out).reshape(shape)

    log_density_u = log_density_z

    @torch.no_grad()
    def phi_u(self, u):
        Z, shape = self._np(u)
        return torch.from_numpy(self.core.phi(Z)).reshape(shape)

    phi_z = phi_u

    @torch.no_grad()
    def sample_posterior(self, n, level=0, seed=123, return_z=True, dtype=torch.float32,
                         max_proposals=None, **kwargs):
        if level != 0:
            raise ValueError("darcy2d initial sampler is for level 0 only")
        Z, info = darcy2d.tempered_level0_pool(self.core, int(n), int(seed))
        return torch.as_tensor(Z, dtype=dtype), info

    def pde_convergence(self, levels, seed):
        return darcy2d.pde_convergence(self.core, levels, seed)


class NS2DProblem(Darcy2DProblem):
    """2D Navier--Stokes (vorticity form, unit torus) posterior hierarchy (see ns2d.py).

    Unknown: KL coefficients z ~ N(0, I_d) of the initial vorticity; samplers work on z directly.
    Data: vorticity at a sensor grid at the observation times. Levels: second-order central finite
    differences (FFT-based) on N_l = n0 2^l grids with time step ~ h_l. Work unit: grid points x
    time steps. Observable phi(z): mean initial vorticity over [0, 1/2]^2 (linear, no PDE).
    The initial pool is drawn by tempered SMC at level 0, as for darcy2d.
    """
    problem_name = "ns2d"

    def __init__(self, y=None, seed=2026, parameter_batch=128, **settings):
        opts = dict(ns2d.NS_DEFAULTS)
        unknown = set(settings) - set(opts)
        if unknown:
            raise ValueError(f"unknown ns settings {sorted(unknown)}")
        opts.update(settings)
        yy = None if y is None else np.asarray(y, dtype=float)
        self.core = ns2d.NSCore(seed=seed, y=yy, **opts)
        self.settings = self.core.settings
        self.dimension = self.core.d
        self.y = torch.as_tensor(self.core.y, dtype=torch.float64)
        self.noise_sd = self.core.noise_sd
        self.parameter_batch = int(parameter_batch)
        self.reference_order = 0
        self._initializing = False
        self.reset_work()

    def description(self):
        s = self.settings
        return {"problem": self.problem_name, "ns_settings": s, "data_level": self.core.data_level,
                "observations": f"vorticity at a {s['sensors']}x{s['sensors']} grid at times {s['obs_times']}",
                "observable": "mean initial vorticity over [0,1/2]^2",
                "posterior_reference": "level-L posterior (no continuum model)"}

    @torch.no_grad()
    def fem_values(self, u, level):
        Z, shape = self._np(u)
        if self._tracking and not self._initializing:
            self._fem_counts[int(level)] += len(Z)
        out = self.core.sensor_values(Z, int(level)) if len(Z) else np.empty((0, len(self.core.y)))
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(*shape, -1)

    @torch.no_grad()
    def log_density_z(self, z, level):
        if level is None:
            raise ValueError("ns2d has no continuum model; use a finite level")
        return super().log_density_z(z, level)

    log_density_u = log_density_z

    @torch.no_grad()
    def sample_posterior(self, n, level=0, seed=123, return_z=True, dtype=torch.float32,
                         max_proposals=None, **kwargs):
        if level != 0:
            raise ValueError("ns2d initial sampler is for level 0 only")
        Z, info = darcy2d.tempered_level0_pool(self.core, int(n), int(seed))
        return torch.as_tensor(Z, dtype=dtype), info

    def pde_convergence(self, levels, seed):
        return ns2d.pde_convergence(self.core, levels, seed)


def make_problem(cfg, y=None):
    """Problem factory: 'beskos' (default, unchanged) or 'source_moments' (bimodal)."""
    kind = cfg.get("problem", "beskos")
    if kind == "beskos":
        return MeasuredBeskos(dimension=cfg["dimension"], y=y, seed=cfg["data_seed"], solver=cfg["solver"],
                              parameter_batch=cfg["parameter_batch"],
                              reference_order=cfg["observable_quadrature_order"], noise_sd=cfg["noise_sd"])
    if kind == "source_moments":
        return SourceMomentProblem(y=y, seed=cfg["data_seed"], solver=cfg["solver"],
                                   parameter_batch=cfg["parameter_batch"],
                                   reference_order=cfg["observable_quadrature_order"], **cfg["source"])
    if kind == "darcy2d":
        return Darcy2DProblem(y=y, seed=cfg["data_seed"], parameter_batch=cfg["parameter_batch"],
                              **cfg.get("darcy", {}))
    if kind == "ns2d":
        return NS2DProblem(y=y, seed=cfg["data_seed"], parameter_batch=cfg["parameter_batch"],
                           **cfg.get("ns", {}))
    raise ValueError("problem must be 'beskos', 'source_moments', 'darcy2d' or 'ns2d'")


class LightSB(nn.Module):
    """Diagonal adjusted Gaussian-mixture potential; all S[j,d] are trainable.

    Same endpoint formulas as the official implementation:
      v(y) = sum_j alpha_j N(y; r_j, epsilon S_j)
      c(x) = integral exp(x.y/epsilon) v(y) dy
      Y | x,j ~ N(r_j+S_j*x, epsilon S_j).
    """
    def __init__(self, dimension, components, epsilon, s_init=0.5):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.r = nn.Parameter(torch.zeros(components, dimension))
        self.log_s = nn.Parameter(torch.full((components, dimension), math.log(s_init)))
        self.log_alpha_raw = nn.Parameter(torch.full((components,), -epsilon * math.log(components)))

    def logits(self, x):
        return ((0.5 * x.square()) @ self.log_s.exp().T + x @ self.r.T) / self.epsilon + self.log_alpha_raw / self.epsilon

    def log_c(self, x):
        return torch.logsumexp(self.logits(x), -1)

    def log_v(self, y):
        variance = self.epsilon * self.log_s.exp()
        residual = y[:, None] - self.r[None]
        log_normal = -0.5 * ((residual.square() / variance).sum(-1)
                             + (2.0 * math.pi * variance).log().sum(-1))
        return torch.logsumexp(log_normal + self.log_alpha_raw / self.epsilon, -1)

    @torch.no_grad()
    def sample(self, x):
        j = torch.multinomial(self.logits(x).softmax(-1), 1).squeeze(-1)
        s = self.log_s.exp()[j]
        mean = self.r[j] + s * x
        return mean + (self.epsilon * s).sqrt() * torch.randn_like(mean)


INIT_MODES = ("source", "identity")
CORRECTION_MODES = ("none", "is")


def identity_scale(epsilon):
    """Optimal diagonal S for the EOT self-coupling of N(0,1) with N(0,1).

    Solves S^2 + epsilon*S - 1 = 0, so that Y = S*X + sqrt(epsilon*S)*xi has
    unit variance. In whitened coordinates this is the natural near-identity
    starting point: corr(X,Y) = S, e.g. 0.78 at epsilon=0.5, 0.975 at 0.05.
    """
    return 0.5 * (-epsilon + math.sqrt(epsilon * epsilon + 4.0))


@torch.no_grad()
def initialize_light_sb(model, anchors, settings):
    """Set r (and for 'identity' also S) from K weighted-drawn source anchors.

    'source':   r_j = anchor_j, S_j = S_init (original behaviour, same RNG use).
    'identity': S_j = identity_scale(eps) * 1, r_j = (1 - S) * anchor_j.
        Then E[Y | x, j] = (1-S) anchor_j + S x: approximately the identity map
        with the right marginal mean and variance for a whitened cloud, while the
        distinct anchors still break the symmetry between the K components.
        S_init is ignored in this mode.
    """
    mode = settings.get("init", "source")
    if mode == "source":
        model.r.copy_(anchors)
        return
    if mode != "identity":
        raise ValueError(f"light.init must be one of {INIT_MODES}")
    s0 = identity_scale(float(model.epsilon))
    model.log_s.fill_(math.log(s0))
    model.r.copy_((1.0 - s0) * anchors)


@torch.no_grad()
def log_marginal_density(model, parents, y, max_entries=2_000_000):
    """log q(y) = log (1/M) sum_k sum_j pi_j(x_k) N(y; r_j + S_j x_k, eps S_j).

    q is the EXACT law of a LightSB child whose parent is drawn uniformly from
    the M-point cloud `parents` (both in the coordinates the model was fitted in).
    Computed in float64 by expanding the Gaussian quadratic form, so memory is
    O(chunk * K * M), never O(chunk * K * M * d).
    """
    eps = float(model.epsilon)
    r = model.r.detach().double()
    S = model.log_s.detach().double().exp()
    log_alpha = model.log_alpha_raw.detach().double() / eps
    X = parents.detach().double()
    Y = y.detach().double()
    M, K = len(X), len(r)
    logits = ((0.5 * X.square()) @ S.T + X @ r.T) / eps + log_alpha       # [M,K]
    log_pi = (logits - torch.logsumexp(logits, -1, keepdim=True)).T         # [K,M]
    inv_var = 1.0 / (eps * S)                                               # [K,d]
    means = r[:, None, :] + S[:, None, :] * X[None]                         # [K,M,d]
    scaled_means = means * inv_var[:, None, :]                              # [K,M,d]
    const = (-0.5 * (means * scaled_means).sum(-1)                          # [K,M]
             - 0.5 * torch.log(2.0 * math.pi * eps * S).sum(-1)[:, None]
             + log_pi)
    chunk = max(1, max_entries // (K * M))
    out = torch.empty(len(Y), dtype=torch.float64)
    for start in range(0, len(Y), chunk):
        yb = Y[start:start + chunk]
        quad = -0.5 * (yb.square() @ inv_var.T)                             # [B,K]
        cross = torch.einsum("bd,kmd->bkm", yb, scaled_means)               # [B,K,M]
        total = quad[:, :, None] + cross + const[None]
        out[start:start + chunk] = torch.logsumexp(total.reshape(len(yb), -1), -1)
    return out - math.log(M)


def _fit_light_sb_legacy(source, log_ratio, settings):
    """No PDE calls during Adam iterations: reuse stage density-ratio values."""
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    source = source.to(dtype=dtype)
    model = LightSB(source.shape[1], settings["K"], settings["epsilon"], settings["S_init"]).to(dtype=dtype)
    weights = log_ratio.softmax(0)
    with torch.no_grad():
        j = torch.multinomial(weights, settings["K"], replacement=True)
        initialize_light_sb(model, source[j], settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    loss = None
    for _ in range(settings["iterations"]):
        if settings["full_batch"]:
            x, w = source, weights.to(dtype=dtype)
        else:
            ids = torch.randint(len(source), (min(len(source), settings["batch_size"]),))
            x = source[ids]
            # Deliberately KEEP the earlier minibatch-self-normalized objective.
            w = log_ratio[ids].softmax(0).to(dtype=dtype)
        loss = model.log_c(x).mean() - torch.dot(w, model.log_v(x))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LightSB objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    return model, float(loss.detach()), float(1.0 / weights.square().sum())



# Tempering here is an OPTIMIZER CONTINUATION ON A FIXED CLOUD. It is NOT
# an SMC resample/move tempering scheme. It cannot recover missing support or
# correct the mismatch between a learned source law q_k and nominal g_k/Z_k.
# Default remains the legacy mode for backwards-controlled comparisons.
IMPORTANCE_DEFAULTS = {
    "target_batching": "legacy_minibatch",
    "weight_tempering": False,
    "tempering_ess_fraction": 0.5,
    "tempering_warmup_fraction": 0.4,
    "tempering_stages": 4,
}
IMPORTANCE_MODES = ("legacy_minibatch", "global_resample", "global_uniform")


def importance_options(settings):
    """Validate explicit options, including direct calls outside merge_config."""
    opts = {k: settings.get(k, v) for k, v in IMPORTANCE_DEFAULTS.items()}
    if opts["target_batching"] not in IMPORTANCE_MODES:
        raise ValueError(f"light.target_batching must be one of {IMPORTANCE_MODES}")
    if type(opts["weight_tempering"]) is not bool:
        raise ValueError("light.weight_tempering must be true or false")
    for name, left_closed in (("tempering_ess_fraction", False),
                              ("tempering_warmup_fraction", True)):
        value = opts[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"light.{name} must be numeric")
        lower_ok = value >= 0 if left_closed else value > 0
        if not math.isfinite(value) or not lower_ok or value >= 1:
            raise ValueError(f"light.{name} must be {'in [0,1)' if left_closed else 'in (0,1)'}")
    stages = opts["tempering_stages"]
    if type(stages) is not int or stages < 1:
        raise ValueError("light.tempering_stages must be a positive integer")
    if opts["weight_tempering"] and opts["target_batching"] == "legacy_minibatch":
        raise ValueError("Tempered warm-up requires global_resample or global_uniform, "
                         "not legacy minibatch normalization")
    return opts


@torch.no_grad()
def normalized_ratio_weights(log_ratio, beta=1.0):
    """Stable full-cloud weights; normalizing constants cancel.

    For this CPU benchmark use float64 for weights even when the model is float32.
    Positive finite ratios are assumed, as in the existing elliptic pipeline.
    """
    lr = torch.as_tensor(log_ratio).detach().to(dtype=torch.float64)
    if lr.ndim != 1 or lr.numel() == 0 or not torch.isfinite(lr).all():
        raise ValueError("log_ratio must be a nonempty finite vector")
    if not math.isfinite(float(beta)) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    if beta == 0.0:
        return torch.full_like(lr, 1.0 / len(lr))
    return (float(beta) * (lr - lr.max())).softmax(0)


@torch.no_grad()
def ratio_weight_stats(weights):
    w = torch.as_tensor(weights).detach().double()
    if w.ndim != 1 or len(w) == 0 or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be a finite nonnegative vector")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    w = w / total
    ess = float(1.0 / w.square().sum())
    positive = w > 0
    return {"ess": ess, "ess_fraction": ess / len(w),
            "max_weight": float(w.max()),
            "entropy": float(-(w[positive] * w[positive].log()).sum())}


@torch.no_grad()
def initial_tempering_beta(log_ratio, ess_fraction=0.5):
    """Largest beta in [0,1] with full-cloud ESS >= ess_fraction * N.

    This is an initialization criterion only. ESS can, and generally will,
    drop below this threshold later when beta returns to one.
    """
    if not 0 < ess_fraction < 1:
        raise ValueError("ess_fraction must lie in (0,1)")
    raw = normalized_ratio_weights(log_ratio, 1.0)
    threshold = ess_fraction * len(raw)
    if float(1.0 / raw.square().sum()) >= threshold:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        w = normalized_ratio_weights(log_ratio, mid)
        if float(1.0 / w.square().sum()) >= threshold:
            low = mid
        else:
            high = mid
    return low


@torch.no_grad()
def make_weight_schedule(log_ratio, settings):
    """Use exactly 'iterations' updates; always finish with beta=1.

    The fraction allocated to warm-up is a fraction of the ORIGINAL budget,
    not extra optimizer updates. If raw ESS is adequate, skip warm-up.
    """
    opts = importance_options(settings)
    iterations = settings["iterations"]
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    beta0 = (initial_tempering_beta(log_ratio, opts["tempering_ess_fraction"])
             if opts["weight_tempering"] else 1.0)
    warmup = min(iterations - 1, int(math.floor(
        iterations * opts["tempering_warmup_fraction"])))
    schedule = []
    if beta0 < 1.0 and warmup > 0:
        count = min(opts["tempering_stages"], warmup)
        quotient, remainder = divmod(warmup, count)
        for j in range(count):
            schedule.append({"beta": beta0 + (1.0 - beta0) * j / count,
                             "updates": quotient + int(j < remainder)})
    else:
        warmup = 0
    schedule.append({"beta": 1.0, "updates": iterations - warmup})
    for item in schedule:
        item.update(ratio_weight_stats(normalized_ratio_weights(log_ratio, item["beta"])))
    return schedule


@torch.no_grad()
def empirical_light_loss(model, source, weights, chunk_size=512):
    """Post-fit UNtempered finite-cloud objective, not a population gap.

    This extra no-gradient diagnostic is inside fit_seconds. It does not
    invoke the PDE, change parameters, or draw random numbers.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = torch.zeros((), dtype=torch.float64, device=source.device)
    n = len(source)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        x = source[start:stop]
        total += model.log_c(x).double().sum() / n
        total -= torch.dot(weights[start:stop].double(), model.log_v(x).double())
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite post-fit empirical LightSB loss")
    return float(total)


def fit_light_sb(source, log_ratio, settings):
    """LightSB training with selectable empirical target minibatching.

    legacy_minibatch: original minibatch-self-normalized ratio objective.
    global_resample: uniform source batch, weighted-resampled target batch.
    global_uniform: uniform batch, target factor N*W_i without renormalization.
    full_batch: evaluate the complete weighted objective without resampling.

    New global modes use a globally normalized fixed-cloud target. Optional
    warm-up flattens its weights initially but ALWAYS returns to beta=1.
    This is not a guarantee of either posterior accuracy or variance reduction.
    """
    opts = importance_options(settings)
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    source = source.detach().to(dtype=dtype)
    if source.ndim != 2 or len(source) < 2 or not torch.isfinite(source).all():
        raise ValueError("source must be a finite [N,D] tensor with N>=2")
    lr = torch.as_tensor(log_ratio, device=source.device).detach()
    if lr.shape != (len(source),):
        raise ValueError("log_ratio must have one value per source particle")
    raw_weights = normalized_ratio_weights(lr, 1.0)
    raw = ratio_weight_stats(raw_weights)
    schedule = make_weight_schedule(lr, settings)
    legacy = opts["target_batching"] == "legacy_minibatch"

    lr_schedule = settings.get("lr_schedule", "constant")
    if legacy:
        if lr_schedule != "constant":
            raise ValueError("light.lr_schedule='cosine' needs a non-legacy target_batching")
        # Retain original operations and RNG draws for a controlled baseline.
        model, last_loss, ess = _fit_light_sb_legacy(source, lr, settings)
    else:
        model = LightSB(source.shape[1], settings["K"], settings["epsilon"],
                        settings["S_init"]).to(device=source.device, dtype=dtype)
        with torch.no_grad():
            initial_weights = normalized_ratio_weights(lr, schedule[0]["beta"])
            j = torch.multinomial(initial_weights, settings["K"], replacement=True)
            initialize_light_sb(model, source[j], settings)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
        B = min(len(source), settings["batch_size"])
        n = len(source)
        loss = None
        # lr_schedule='cosine': lr_k = lr * (1 + cos(pi k / total)) / 2 over ALL updates. Decaying to 0
        # removes the last-iterate SGD noise in the gate weights alpha_j/eps, which otherwise sets a
        # fit-to-fit random fraction of mass moved between modes (dominant LightSB error, source tests).
        total_updates = sum(item["updates"] for item in schedule)
        step = 0
        for item in schedule:
            # Full-cloud normalization ONCE per plateau, outside Adam updates.
            w64 = normalized_ratio_weights(lr, item["beta"])
            w_model = w64.to(dtype=dtype)
            for _ in range(item["updates"]):
                if settings["full_batch"]:
                    loss = model.log_c(source).mean() - torch.dot(w_model, model.log_v(source))
                else:
                    ids = torch.randint(n, (B,), device=source.device)
                    x = source[ids]
                    if opts["target_batching"] == "global_resample":
                        # Weighted resampling is ONLY for the target minibatch.
                        # Do NOT apply weights to log_v(y) again.
                        target_ids = torch.multinomial(w64, B, replacement=True)
                        y = source[target_ids]
                        target_term = model.log_v(y).mean()
                    else:
                        # E_uniform[N*W_I*f(X_I)] = sum_i W_i*f(X_i).
                        # In particular, never divide these batch weights by their sum.
                        target_term = (n * w_model[ids] * model.log_v(x)).mean()
                    loss = model.log_c(x).mean() - target_term
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite LightSB objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                if lr_schedule == "cosine":
                    for group in optimizer.param_groups:
                        group["lr"] = settings["lr"] * 0.5 * (1.0 + math.cos(math.pi * step / total_updates))
                step += 1
                optimizer.step()
            item["last_training_loss"] = float(loss.detach())
        last_loss, ess = float(loss.detach()), raw["ess"]

    # Raw/final ESS is always reported; warm-up ESS never replaces it.
    model.importance_diagnostics = {
        "target_batching": opts["target_batching"],
        "full_batch": bool(settings["full_batch"]),
        "source_particles": len(source),
        "raw_ess": raw["ess"], "raw_ess_fraction": raw["ess_fraction"],
        "raw_max_weight": raw["max_weight"], "raw_entropy": raw["entropy"],
        "tempering_requested": opts["weight_tempering"],
        "tempering_active": len(schedule) > 1,
        "initial_beta": schedule[0]["beta"], "final_beta": schedule[-1]["beta"],
        "initial_ess_fraction": schedule[0]["ess_fraction"],
        "warmup_updates": sum(t["updates"] for t in schedule if t["beta"] < 1.0),
        "untempered_updates": sum(t["updates"] for t in schedule if t["beta"] == 1.0),
        "total_updates": sum(t["updates"] for t in schedule),
        "schedule": schedule,
        "final_empirical_loss": empirical_light_loss(model, source, raw_weights,
                                                     settings["batch_size"]),
        "raw_ess_below_5_percent": raw["ess_fraction"] < 0.05,
    }
    return model, last_loss, ess

WHITEN_DEFAULTS = {"whiten": "none", "whiten_shrinkage": 1e-3,
                   "init": "source", "correction": "none"}
WHITEN_MODES = ("none", "diag", "full")


def whitening_options(settings):
    """Validates whitening, LightSB initialization and importance correction."""
    opts = {k: settings.get(k, v) for k, v in WHITEN_DEFAULTS.items()}
    if opts["whiten"] not in WHITEN_MODES:
        raise ValueError(f"light.whiten must be one of {WHITEN_MODES}")
    if opts["init"] not in INIT_MODES:
        raise ValueError(f"light.init must be one of {INIT_MODES}")
    if opts["correction"] not in CORRECTION_MODES:
        raise ValueError(f"light.correction must be one of {CORRECTION_MODES}")
    s = opts["whiten_shrinkage"]
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s < 1:
        raise ValueError("light.whiten_shrinkage must be in [0,1)")
    return opts


EPSILON_DEFAULTS = {"epsilon_schedule": "constant", "epsilon_decay": 0.5,
                    "epsilon_min": 0.0, "epsilon_by_level": None,
                    "lr_epsilon_scaling": False}
EPSILON_SCHEDULES = ("constant", "geometric", "list")


def epsilon_options(settings):
    opts = {k: settings.get(k, v) for k, v in EPSILON_DEFAULTS.items()}
    if opts["epsilon_schedule"] not in EPSILON_SCHEDULES:
        raise ValueError(f"light.epsilon_schedule must be one of {EPSILON_SCHEDULES}")
    decay, floor = opts["epsilon_decay"], opts["epsilon_min"]
    for name, value in (("epsilon_decay", decay), ("epsilon_min", floor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"light.{name} must be a finite number")
    if not 0 < decay <= 1:
        raise ValueError("light.epsilon_decay must be in (0,1]")
    if floor < 0:
        raise ValueError("light.epsilon_min must be >= 0")
    if opts["epsilon_schedule"] == "list":
        values = opts["epsilon_by_level"]
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError("light.epsilon_by_level must be a non-empty list of positive numbers, "
                             "entry l-1 used for transition l-1 -> l")
    if type(opts["lr_epsilon_scaling"]) is not bool:
        raise ValueError("light.lr_epsilon_scaling must be true or false")
    return opts


def epsilon_for_level(settings, level):
    """Entropic regularisation for transition level-1 -> level (level >= 1).

    constant:  eps_l = epsilon
    geometric: eps_l = max(epsilon * epsilon_decay**(l-1), epsilon_min)
               e.g. decay 0.25 = h_l^2 scaling, decay 0.5 = h_l scaling.
    list:      eps_l = epsilon_by_level[l-1]

    In whitened coordinates the optimal self-coupling has E|X-Y|^2 ~ eps*d for
    small eps, so a decreasing eps_l is NECESSARY (not sufficient) for the
    variance of the paired increments to decay with the level.
    """
    opts = epsilon_options(settings)
    if opts["epsilon_schedule"] == "constant":
        return float(settings["epsilon"])
    if opts["epsilon_schedule"] == "geometric":
        return max(float(settings["epsilon"]) * opts["epsilon_decay"] ** (level - 1),
                   float(opts["epsilon_min"]))
    values = opts["epsilon_by_level"]
    if level > len(values):
        raise ValueError(f"light.epsilon_by_level has {len(values)} entries; level {level} needs one")
    return float(values[level - 1])


def level_settings(settings, level):
    """Settings for one transition. Returns the SAME dict when nothing changes,
    so the constant schedule reproduces earlier runs exactly.

    lr_epsilon_scaling: lr_l = lr * eps_l / epsilon. Heuristic: with the
    identity initialisation the optimum lies within O(eps_l) of the start
    (1 - S ~ eps/2, r ~ eps/2 * anchor), so a fixed Adam step size would
    overshoot it once eps_l is small.
    """
    eps = epsilon_for_level(settings, level)
    opts = epsilon_options(settings)
    iters = iterations_for_level(settings, level)
    if (eps == float(settings["epsilon"]) and not opts["lr_epsilon_scaling"]
            and iters == settings["iterations"]):
        return settings
    out = dict(settings, epsilon=eps, iterations=iters)
    if opts["lr_epsilon_scaling"]:
        out["lr"] = float(settings["lr"]) * eps / float(settings["epsilon"])
    return out


def iterations_for_level(settings, level):
    """light.iterations_by_level = [n_1, n_2, ...]: SGD iterations for the fit of
    transition (level-1 -> level); the last entry is reused for deeper levels.
    Absent/None: light.iterations at every level (earlier runs unchanged).
    Motivation (T3, 9 Oct): the fit bias from too few iterations sits only in
    levels 1-4; levels >= 5 give the same corrections with 100 as with 1000."""
    schedule = settings.get("iterations_by_level")
    if schedule is None:
        return settings["iterations"]
    return int(schedule[min(level, len(schedule)) - 1])


@torch.no_grad()
def fit_whitening(x, mode, shrinkage=1e-3):
    """Affine map from the UNWEIGHTED source cloud, computed in float64.

    Shrinkage is RELATIVE to each coordinate's own variance,
        C_reg = C + shrinkage * diag(C),
    so narrow coordinates (e.g. z_1 with sd ~0.07) are not swamped by a
    trace-scaled ridge. Returns None for mode='none'.
    """
    if mode == "none":
        return None
    x64 = torch.as_tensor(x).detach().double()
    n, d = x64.shape
    if n < 2:
        raise ValueError("Whitening needs at least two source particles")
    mean = x64.mean(0)
    centered = x64 - mean
    var = centered.square().sum(0) / (n - 1)
    if not torch.isfinite(var).all() or (var <= 0).any():
        raise FloatingPointError("Degenerate source cloud: zero/non-finite coordinate variance")
    if mode == "diag":
        scale = (var * (1.0 + shrinkage)).sqrt()
        info = {"mode": "diag", "min_scale": float(scale.min()), "max_scale": float(scale.max()),
                "condition_number": float((scale.max() / scale.min()) ** 2)}
        return {"mode": "diag", "mean": mean, "scale": scale, "info": info}
    if n <= d:
        warnings.warn(f"Full whitening with n={n} <= d={d}: covariance relies on shrinkage")
    cov = centered.T @ centered / (n - 1)
    cov = cov + torch.diag(shrinkage * var)
    chol, failed = torch.linalg.cholesky_ex(cov)
    if failed:
        raise FloatingPointError("Cholesky of regularized source covariance failed; increase whiten_shrinkage")
    eig = torch.linalg.eigvalsh(cov)
    info = {"mode": "full", "min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()),
            "condition_number": float(eig.max() / eig.min()),
            "min_marginal_sd": float(var.sqrt().min()), "max_marginal_sd": float(var.sqrt().max())}
    return {"mode": "full", "mean": mean, "chol": chol, "info": info}


@torch.no_grad()
def whiten(x, W):
    """x -> A^{-1}(x - m), returned in x's dtype."""
    if W is None:
        return x
    c = x.double() - W["mean"]
    if W["mode"] == "diag":
        out = c / W["scale"]
    else:
        out = torch.linalg.solve_triangular(W["chol"], c.T, upper=False).T
    return out.to(dtype=x.dtype)


@torch.no_grad()
def unwhiten(xt, W):
    """x~ -> m + A x~, returned in xt's dtype."""
    if W is None:
        return xt
    t = xt.double()
    out = W["mean"] + (t * W["scale"] if W["mode"] == "diag" else t @ W["chol"].T)
    return out.to(dtype=xt.dtype)


def entry(level, n, contribution, within_var=None):
    return {"level": int(level), "source_level": int(level - 1),
            "h_source": None if level == 0 else 2.0 ** (-(level + 2)),
            "n_correction": int(n), "contribution": float(contribution),
            "within_particle_variance": None if within_var is None else float(within_var)}


def finish_result(problem, clock, start, estimate, contributions, diagnostics):
    return {"estimate": float(estimate), "contributions": contributions,
            "sampler_seconds": float(clock.values["sampler"]),
            "estimator_seconds": float(clock.values["estimator"]),
            "total_seconds": float(time.perf_counter() - start),
            "timing_profile_version": TIMING_PROFILE_VERSION,
            **clock.breakdown(),
            **problem.work(), "diagnostics": diagnostics}


@torch.no_grad()
def coupling_diagnostics(parents, children, delta, W, wopts):
    """How close are (parent, child) pairs? No PDE calls, no RNG draws."""
    parents64, children64 = parents.double(), children.double()
    pc = parents64 - parents64.mean(0)
    cc = children64 - children64.mean(0)
    corr = (pc * cc).sum(0) / (pc.norm(dim=0) * cc.norm(dim=0)).clamp_min(1e-300)
    return {
        "whiten": wopts["whiten"],
        "init": wopts["init"],
        "correction": wopts["correction"],
        "whitening": None if W is None else W["info"],
        "pair_corr_coord1": float(corr[0]),
        "pair_corr_coord2": float(corr[1]) if len(corr) > 1 else None,
        "pair_corr_median": float(corr.median()),
        "child_sd_over_parent_sd_coord1": float(cc[:, 0].std() / pc[:, 0].std()),
        "pair_msd_z": float((children64 - parents64).square().sum(-1).mean()),
        "delta_phi_variance": float(delta.var(unbiased=True)),
    }


def run_light_sb_is(problem, initial_z, N, settings, seed):
    """LightSB transport with an EXACT importance correction at every level.

    State before transition l-1 -> l: an UNWEIGHTED cloud x_1..x_M (M = N[l-1])
    approximating pi_{l-1}, with cached log gamma_{l-1}(x).
      1. ratio g_l/g_{l-1} on the cloud (as before) -> fit LightSB (whitened).
      2. Draw N[l] parents a_i uniformly from the cloud (without replacement if
         N[l] <= M), children Y_i ~ K_theta(x_{a_i}, .).
      3. Marginally Y_i ~ q_l(y) = (1/M) sum_k K_theta(x_k, y), an explicit
         Gaussian mixture. Weight W_i proportional to gamma_l(Y_i) / q_l(Y_i),
         self-normalised (the whitening Jacobian is constant and cancels).
      4. Paired correction
             Delta_l = sum_i W_i phi(Y_i) - (1/N[l]) sum_i phi(x_{a_i})
                     = (1/N[l]) sum_i [N[l] W_i phi(Y_i) - phi(x_{a_i})].
         Consistent for pi_l(phi) - pi_{l-1}(phi) WHATEVER the transport quality;
         variance is small when W_i ~ 1/N[l] and pairs are close.
      5. Multinomially resample the children by W to an unweighted cloud of
         size N[l]; carry gamma_l and phi along (no new evaluations).

    PDE work per transition: M solves at level l (ratio) + N[l] solves at level
    l (child weights). This is more than the uncorrected method and is counted.
    """
    torch.manual_seed(seed)
    problem.reset_work()
    clock = Timers()
    start = time.perf_counter()
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    wopts = whitening_options(settings)
    with clock.measure("sampler"):
        cloud = initial_z[:N[0]].to(dtype=dtype).clone()
        if len(N) > 1:
            with clock.measure("pde_reweight"):
                log_g_prev = problem.log_density_z(cloud.double(), 0)
    with clock.measure("estimator"):
        phi_cloud = problem.phi_z(cloud.double())
        terms = [entry(0, N[0], phi_cloud.mean(), phi_cloud.var(unbiased=True))]
    diagnostics = []
    for level in range(1, len(N)):
        before = dict(clock.values)
        with clock.measure("sampler"):
            with clock.measure("pde_reweight"):
                log_g_cloud = problem.log_density_z(cloud.double(), level)
            log_ratio = log_g_cloud - log_g_prev
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError("Non-finite LightSB density ratios")
            with clock.measure("fit"):
                W = fit_whitening(cloud, wopts["whiten"], wopts["whiten_shrinkage"])
                source_t = whiten(cloud, W)
                stage_settings = level_settings(settings, level)
                model, loss, ess = fit_light_sb(source_t, log_ratio, stage_settings)
            with clock.measure("propagation"):
                count = N[level]
                if count <= len(cloud):
                    ids = torch.randperm(len(cloud))[:count]
                else:
                    ids = torch.randint(len(cloud), (count,))
                children_t = model.sample(source_t[ids])
                children = unwhiten(children_t, W).to(dtype=dtype)
                if not torch.isfinite(children).all():
                    raise FloatingPointError("Non-finite LightSB samples")
                q_start = time.perf_counter()
                log_q = log_marginal_density(model, source_t, children_t)
                q_seconds = time.perf_counter() - q_start
            with clock.measure("pde_reweight"):
                log_g_child = problem.log_density_z(children.double(), level)
            log_w = log_g_child - log_q
            if not torch.isfinite(log_w).all():
                raise FloatingPointError("Non-finite LightSB importance weights")
            weights = normalized_ratio_weights(log_w)
            is_stats = ratio_weight_stats(weights)
            if is_stats["ess"] < 2.0:
                raise FloatingPointError(
                    f"Level {level}: marginal IS weights gamma_l/q_l collapsed "
                    f"(ESS={is_stats['ess']:.2f} of {count}). q_l is a kernel mixture with "
                    "bandwidth sqrt(eps*S) in d dimensions, so log q fluctuates like a "
                    "chi-square(d) term; this correction is not viable in high dimension.")
        with clock.measure("estimator"):
            phi_children = problem.phi_z(children.double())
            phi_parents = phi_cloud[ids].double()
            contribution = float(torch.dot(weights, phi_children) - phi_parents.mean())
            delta = count * weights * phi_children - phi_parents
            terms.append(entry(level, count, contribution, delta.var(unbiased=True)))
            importance = dict(model.importance_diagnostics)
            weighted_phi = float(torch.dot(normalized_ratio_weights(log_ratio), phi_cloud.double()))
            unweighted = phi_children - phi_parents
            importance.update({
                "source_phi_mean": float(phi_cloud.mean()),
                "importance_target_phi_mean": weighted_phi,
                "output_phi_mean": float(phi_children.mean()),
                "output_minus_importance_phi": float(phi_children.mean()) - weighted_phi,
                "is_weighted_output_phi_mean": float(torch.dot(weights, phi_children)),
                "is_ess": is_stats["ess"], "is_ess_fraction": is_stats["ess_fraction"],
                "is_max_weight": is_stats["max_weight"],
                "unweighted_paired_correction": float(unweighted.mean()),
                "is_paired_correction": contribution,
                "marginal_density_seconds": q_seconds,
            })
            importance.update({"epsilon_level": float(stage_settings["epsilon"]),
                               "lr_level": float(stage_settings["lr"]),
                               "iterations_level": int(stage_settings["iterations"])})
            importance.update(coupling_diagnostics(cloud[ids], children, unweighted, W, wopts))
            importance["is_delta_variance"] = float(delta.var(unbiased=True))
        with clock.measure("sampler"):
            with clock.measure("resampling"):
                r_ids = torch.multinomial(weights, count, replacement=True)
                cloud = children[r_ids].clone()
                log_g_prev = log_g_child[r_ids].clone()
        phi_cloud = phi_children[r_ids]  # cached values, no new evaluation
        diagnostics.append({"level": level, "ratio_ess": ess, "last_loss": loss,
                            "importance": importance,
                            "timing_breakdown": clock.stage_breakdown(before)})
        del model
    estimate = math.fsum(t["contribution"] for t in terms)
    answer = finish_result(problem, clock, start, estimate, terms, diagnostics)
    answer["final_mean_diagnostic"] = float(phi_cloud.mean())
    return answer


def run_light_sb(problem, initial_z, N, settings, seed):
    if whitening_options(settings)["correction"] == "is":
        return run_light_sb_is(problem, initial_z, N, settings, seed)
    torch.manual_seed(seed)
    problem.reset_work()
    clock = Timers()
    start = time.perf_counter()
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    wopts = whitening_options(settings)
    with clock.measure("sampler"):
        current = initial_z[:N[0]].to(dtype=dtype).clone()
    with clock.measure("estimator"):
        phi_current = problem.phi_z(current.double())
        terms = [entry(0, N[0], phi_current.mean(), phi_current.var(unbiased=True))]
    diagnostics = []
    for level in range(1, len(N)):
        before = dict(clock.values)
        with clock.measure("sampler"):
            # PDE evaluation in CPU float64, regardless of LightSB parameter dtype.
            z64 = current.double()
            with clock.measure("pde_reweight"):
                log_ratio = problem.log_density_z(z64, level) - problem.log_density_z(z64, level - 1)
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError("Non-finite LightSB density ratios")
            # Includes model/optimizer construction and every Adam update.
            # Cached log_ratio means there are still NO PDE calls in this timer.
            # Whitening statistics are part of fitting; they use no RNG draws,
            # so whiten='none' reproduces the earlier random stream exactly.
            with clock.measure("fit"):
                W = fit_whitening(current, wopts["whiten"], wopts["whiten_shrinkage"])
                source_t = whiten(current, W)
                stage_settings = level_settings(settings, level)
                model, loss, ess = fit_light_sb(source_t, log_ratio, stage_settings)
            with clock.measure("propagation"):
                count = N[level]
                if count <= len(current):
                    ids = torch.randperm(len(current))[:count]
                else:
                    ids = torch.randint(len(current), (count,))
                children = unwhiten(model.sample(source_t[ids]), W).to(dtype=dtype)
                if not torch.isfinite(children).all():
                    raise FloatingPointError("Non-finite LightSB samples")
        with clock.measure("estimator"):
            # Cache phi on the population; reuse it for the selected parents.
            phi_next = problem.phi_z(children.double())
            delta = phi_next - phi_current[ids]
            terms.append(entry(level, count, delta.mean(), delta.var(unbiased=True)))
            importance = dict(model.importance_diagnostics)
            weighted_phi = float(torch.dot(normalized_ratio_weights(log_ratio), phi_current.double()))
            importance.update({"source_phi_mean": float(phi_current.mean()),
                               "importance_target_phi_mean": weighted_phi,
                               "output_phi_mean": float(phi_next.mean()),
                               "output_minus_importance_phi": float(phi_next.mean()) - weighted_phi})
            importance.update({"epsilon_level": float(stage_settings["epsilon"]),
                               "lr_level": float(stage_settings["lr"]),
                               "iterations_level": int(stage_settings["iterations"])})
            importance.update(coupling_diagnostics(current[ids], children, delta, W, wopts))
            lab = problem.mode_label(children)
            if lab is not None:
                importance["right_mode_fraction"] = float(lab.double().mean())
        diagnostics.append({"level": level, "ratio_ess": ess, "last_loss": loss,
                            "importance": importance,
                            "timing_breakdown": clock.stage_breakdown(before)})
        current, phi_current = children, phi_next
        del model
    estimate = math.fsum(t["contribution"] for t in terms)
    answer = finish_result(problem, clock, start, estimate, terms, diagnostics)
    answer["final_mean_diagnostic"] = float(phi_current.mean())
    return answer


@torch.no_grad()
def rwm_mutate(problem, z, logp, level, settings, generator, clock=None):
    blocks = [list(range(i, min(i + settings["block_size"], z.shape[1])))
              for i in range(0, z.shape[1], settings["block_size"])]
    accepted = torch.zeros(len(blocks), dtype=torch.float64)
    sum_probability = torch.zeros(len(blocks), dtype=torch.float64)
    for _ in range(settings["sweeps"]):
        for b, block in enumerate(blocks):
            proposal = z.clone()
            proposal[:, block] += settings["scale"] * torch.randn(
                (len(z), len(block)), dtype=torch.float64, generator=generator)
            if clock is None:
                lp = problem.log_density_z(proposal, level)
            else:
                # Nested inside the inclusive MCMC timer in run_smc().
                with clock.measure("pde_mcmc"):
                    lp = problem.log_density_z(proposal, level)
            difference = lp - logp
            if not torch.isfinite(difference).all():
                raise FloatingPointError("Non-finite Metropolis log ratio")
            accept = torch.rand(len(z), dtype=torch.float64, generator=generator).log() < difference
            z[accept] = proposal[accept]
            logp[accept] = lp[accept]
            accepted[b] += accept.sum()
            sum_probability[b] += difference.clamp(max=0).exp().sum()
    denom = len(z) * settings["sweeps"]
    return z, logp, {"blocks": blocks, "acceptance_fraction": (accepted / denom).tolist(),
                    "mean_acceptance_probability": (sum_probability / denom).tolist()}


@torch.no_grad()
def run_smc(problem, initial_z, N, settings, seed, multilevel):
    """MLSMC eq.(5) or standard SMC final mean; SAME fixed observable.

    In MLSMC the final correction only uses the level L-1 population: no
    level-L resampling/mutation is needed or charged. This matters for timing.
    Standard SMC does mutate through L, and uses a constant N within each run.
    """
    generator = torch.Generator().manual_seed(seed)
    problem.reset_work()
    clock = Timers()
    start = time.perf_counter()
    L = len(N) - 1
    with clock.measure("sampler"):
        current = initial_z[:N[0]].double().clone()
        # No density evaluation is needed at all if L=0.
        if L > 0:
            with clock.measure("pde_reweight"):
                logp = problem.log_density_z(current, 0)
        else:
            logp = None
    terms, diagnostics = [], []
    if multilevel:
        with clock.measure("estimator"):
            phi_current = problem.phi_z(current)
            terms = [entry(0, N[0], phi_current.mean(), phi_current.var(unbiased=True))]
    for level in range(1, L + 1):
        before = dict(clock.values)
        with clock.measure("sampler"):
            with clock.measure("pde_reweight"):
                log_next = problem.log_density_z(current, level)
            log_ratio = log_next - logp
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError("Non-finite SMC log weights")
            weights = log_ratio.softmax(0)
            diag = {"level": level, "ratio_ess": float(1.0 / weights.square().sum())}
        if multilevel:
            with clock.measure("estimator"):
                # Same as dot(weights,phi)-mean(phi), centered for accuracy.
                correction = torch.dot(weights, phi_current - phi_current.mean())
                terms.append(entry(level, len(current), correction))
            if level == L:
                diag["final_mutation_skipped"] = True
                diag["timing_breakdown"] = clock.stage_breakdown(before)
                diagnostics.append(diag)
                break
        with clock.measure("sampler"):
            with clock.measure("resampling"):
                ids = torch.multinomial(weights, N[level], replacement=True, generator=generator)
                current = current[ids].clone()
                logp = log_next[ids].clone()
            with clock.measure("mcmc"):
                current, logp, acceptance = rwm_mutate(
                    problem, current, logp, level, settings, generator, clock=clock)
            diag.update(acceptance)
            lab = problem.mode_label(current)
            if lab is not None:
                diag["right_mode_fraction"] = float(lab.double().mean())
        if multilevel:
            with clock.measure("estimator"):
                phi_current = problem.phi_z(current)
        diag["timing_breakdown"] = clock.stage_breakdown(before)
        diagnostics.append(diag)
    with clock.measure("estimator"):
        if multilevel:
            estimate = math.fsum(t["contribution"] for t in terms)
        else:
            estimate = float(problem.phi_z(current).mean())
    return finish_result(problem, clock, start, estimate, terms, diagnostics)



# =============================================================================
# Transport + MCMC methods (transport_mh directory)
# =============================================================================
TRANSPORT_MH_DEFAULTS = {"sweeps": 1, "rho": "auto", "shrinkage": 1e-3,
                         # rho = "ess": step chosen from the training-weight ESS of each transition
                         "ess_rho_min": 0.8, "ess_low": 0.05, "ess_high": 0.5, "ess_sweeps": None,
                         # coupled = true: synchronously coupled pCN on (parent, child), see coupled_pcn_mh
                         "coupled": False}
MLSMC_KERNEL_DEFAULTS = {"sweeps": 1, "rho": "auto", "epsilon": 0.5, "shrinkage": 1e-3,
                         "iterations": 1000, "lr": 0.01, "init": "identity", "K": None}


def _validate_rho(rho, name):
    if rho == "auto" or (rho == "ess" and name == "transport_mh"):
        return
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 <= rho < 1:
        allowed = "'auto', 'ess'" if name == "transport_mh" else "'auto'"
        raise ValueError(f"{name}.rho must be {allowed} or a number in [0,1)")


def validate_transport_options(cfg):
    t, m = cfg["transport_mh"], cfg["mlsmc_kernel"]
    for name, block in (("transport_mh", t), ("mlsmc_kernel", m)):
        if type(block["sweeps"]) is not int or block["sweeps"] < 0:
            raise ValueError(f"{name}.sweeps must be a nonnegative integer")
        _validate_rho(block["rho"], name)
        s = block["shrinkage"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s < 1:
            raise ValueError(f"{name}.shrinkage must be in [0,1)")
    if not (0 <= t["ess_rho_min"] < 1 and 0 < t["ess_low"] < t["ess_high"] <= 1):
        raise ValueError("transport_mh needs 0<=ess_rho_min<1 and 0<ess_low<ess_high<=1")
    if not isinstance(t.get("coupled", False), bool):
        raise ValueError("transport_mh.coupled must be true or false")
    if t["ess_sweeps"] is not None and (type(t["ess_sweeps"]) is not int or t["ess_sweeps"] < 0):
        raise ValueError("transport_mh.ess_sweeps must be null or a nonnegative integer")
    if m["sweeps"] < 1:
        raise ValueError("mlsmc_kernel.sweeps must be >= 1 (MLSMC needs a mutation)")
    unknown = set(m) - set(MLSMC_KERNEL_DEFAULTS) - set(MLSMC_ESS_SWEEP_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown mlsmc_kernel keys: {sorted(unknown)}")
    me = {k: m.get(k, v) for k, v in MLSMC_ESS_SWEEP_DEFAULTS.items()}
    if me["ess_sweeps"] is not None and (type(me["ess_sweeps"]) is not int or me["ess_sweeps"] < 1):
        raise ValueError("mlsmc_kernel.ess_sweeps must be null or a positive integer")
    if not 0 < me["ess_low"] < me["ess_high"] <= 1:
        raise ValueError("mlsmc_kernel needs 0<ess_low<ess_high<=1")
    if not m["epsilon"] > 0 or type(m["iterations"]) is not int or m["iterations"] < 1 or not m["lr"] > 0:
        raise ValueError("mlsmc_kernel needs epsilon>0, integer iterations>=1, lr>0")
    if m["init"] not in INIT_MODES:
        raise ValueError(f"mlsmc_kernel.init must be one of {INIT_MODES}")
    if m["K"] is not None and (type(m["K"]) is not int or m["K"] < 1):
        raise ValueError("mlsmc_kernel.K must be null or a positive integer")


MLSMC_ESS_SWEEP_DEFAULTS = {"ess_sweeps": None, "ess_low": 0.05, "ess_high": 0.5}


def mlsmc_level_sweeps(ess_fraction, kcfg):
    """Control for TransportMH's ess_sweeps: the SAME interpolation rule applied to MLSMC's
    mutation. ess_fraction = ESS/N of the reweighting weights gamma_l/gamma_{l-1} on the cloud
    (the same quantity TransportMH calls its training ESS). With mlsmc_kernel.ess_sweeps unset
    (default) always mlsmc_kernel.sweeps, so earlier runs are unchanged."""
    opts = {k: kcfg.get(k, v) for k, v in MLSMC_ESS_SWEEP_DEFAULTS.items()}
    if opts["ess_sweeps"] is None:
        return kcfg["sweeps"]
    lo, hi = math.log(float(opts["ess_low"])), math.log(float(opts["ess_high"]))
    t = min(1.0, max(0.0, (math.log(max(float(ess_fraction), 1e-300)) - lo) / (hi - lo)))
    return int(round(opts["ess_sweeps"] + t * (kcfg["sweeps"] - opts["ess_sweeps"])))


def pcn_rho(rho, epsilon):
    """'auto' ties the pCN autocorrelation to the entropic scale: rho = S(eps),
    so the pCN innovation variance 1 - rho^2 = eps*S equals the LightSB
    identity-kernel noise variance (whitened units)."""
    return identity_scale(float(epsilon)) if rho == "auto" else float(rho)


def ess_adaptive_rho(ess_fraction, epsilon, mh):
    """TransportMH step for rho = 'ess'. The training-weight ESS fraction e of the transition
    measures how well the LightSB target pi_l is represented by the reweighted source cloud.
      e <= ess_low : poorly informed transport -> large moves, rho = ess_rho_min;
      e >= ess_high: well informed             -> rho = S(eps_l) (as 'auto');
      in between   : the innovation variance 1 - rho^2 is interpolated log-linearly in log e.
    Sweeps: with ess_sweeps = None always transport_mh.sweeps. Otherwise ess_sweeps at e <= ess_low,
    transport_mh.sweeps at e >= ess_high, and in between the same weight t on the log scale,
    n = round(ess_sweeps + t * (sweeps - ess_sweeps)), so transitions just above ess_low also get extra sweeps.
    Returns (rho, sweeps)."""
    v_lo = 1.0 - float(mh["ess_rho_min"]) ** 2
    v_hi = 1.0 - identity_scale(float(epsilon)) ** 2
    lo, hi = math.log(float(mh["ess_low"])), math.log(float(mh["ess_high"]))
    t = min(1.0, max(0.0, (math.log(max(float(ess_fraction), 1e-300)) - lo) / (hi - lo)))
    v = math.exp((1.0 - t) * math.log(v_lo) + t * math.log(v_hi)) if v_hi < v_lo else v_hi
    sweeps = mh["sweeps"]
    if mh.get("ess_sweeps") is not None:
        sweeps = int(round(mh["ess_sweeps"] + t * (mh["sweeps"] - mh["ess_sweeps"])))
    return math.sqrt(max(0.0, 1.0 - v)), sweeps


@torch.no_grad()
def log_kernel_pairs(model, x, y):
    """Row-wise log K_theta(x_i, y_i) = log sum_j pi_j(x_i) N(y_i; r_j + S_j x_i, eps S_j).

    Only the given (x_i, y_i) pairs are evaluated: O(n K d), no sum over a cloud.
    Coordinates are those the model was fitted in (whitened); float64.
    """
    eps = float(model.epsilon)
    r = model.r.detach().double()
    S = model.log_s.detach().double().exp()
    log_alpha = model.log_alpha_raw.detach().double() / eps
    X, Y = x.detach().double(), y.detach().double()
    log_pi = (((0.5 * X.square()) @ S.T + X @ r.T) / eps + log_alpha).log_softmax(-1)  # [n,K]
    var = eps * S                                                                     # [K,d]
    means = r[None] + S[None] * X[:, None]                                            # [n,K,d]
    log_n = -0.5 * (((Y[:, None] - means).square() / var[None]).sum(-1)
                    + torch.log(2.0 * math.pi * var).sum(-1)[None])                   # [n,K]
    return torch.logsumexp(log_pi + log_n, -1)


class _ModeTracker:
    """Counts accepted moves that switch mode (problem.mode_label); no-op for unimodal problems."""
    def __init__(self, problem, z):
        self.fn = getattr(problem, "mode_label", None)
        self.label = self.fn(z) if self.fn is not None else None
        self.crossings = 0.0

    def update(self, z_prop, accept):
        if self.label is None:
            return
        lab = self.fn(z_prop)
        self.crossings += float((accept & (lab != self.label)).sum())
        self.label = torch.where(accept, lab, self.label)

    def summary(self, n, sweeps):
        if self.label is None:
            return {}
        return {"mh_mode_switches_per_particle_sweep": self.crossings / max(1, n * sweeps),
                "right_mode_fraction_after_mh": float(self.label.double().mean())}


def _mh_summary(name, accepted, probability, n, sweeps, extra=None):
    out = {"mh_kernel": name, "mh_sweeps": sweeps,
           "mh_acceptance_fraction": float(accepted) / max(1, n * sweeps),
           "mh_mean_acceptance_probability": float(probability) / max(1, n * sweeps)}
    if extra:
        out.update(extra)
    return out


@torch.no_grad()
def pcn_mh(problem, z, log_g, level, rho, sweeps, W, clock, generator=None):
    """pCN Metropolis-Hastings targeting gamma_level, Gaussian reference N(m, C)
    taken from the whitening W (full) of the current cloud.

    In whitened t: proposal t' = rho t + sqrt(1-rho^2) xi is reversible w.r.t.
    N(0, I), so log alpha = [log g(t') + |t'|^2/2] - [log g(t) + |t|^2/2].
    The whitening Jacobian is constant and cancels. One PDE solve per sweep.
    """
    t = whiten(z.double(), W)
    log_g = log_g.clone()
    step = math.sqrt(max(0.0, 1.0 - rho * rho))
    accepted, probability = 0.0, 0.0
    tracker = _ModeTracker(problem, z.double())
    for _ in range(sweeps):
        proposal = rho * t + step * torch.randn(t.shape, dtype=torch.float64, generator=generator)
        z_prop = unwhiten(proposal, W)
        with clock.measure("pde_mcmc"):
            lg = problem.log_density_z(z_prop, level)
        log_a = (lg + 0.5 * proposal.square().sum(-1)) - (log_g + 0.5 * t.square().sum(-1))
        if torch.isnan(log_a).any():
            raise FloatingPointError("NaN in pCN acceptance ratio")
        accept = torch.rand(len(t), dtype=torch.float64, generator=generator).log() < log_a
        tracker.update(z_prop, accept)
        t[accept] = proposal[accept]
        log_g[accept] = lg[accept]
        accepted += float(accept.sum())
        probability += float(log_a.clamp(max=0).exp().sum())
    return unwhiten(t, W), log_g, _mh_summary("pcn", accepted, probability, len(t), sweeps,
                                              {"pcn_rho": rho, **tracker.summary(len(t), sweeps)})


@torch.no_grad()
def coupled_pcn_mh(problem, z_src, log_g_src, level_src, z_dst, log_g_dst, level_dst,
                   rho, sweeps, W, clock, generator=None):
    """Synchronously coupled pCN-MH on pairs (parent, child).

    The parent chain leaves pi_{level_src} invariant and the child chain pi_{level_dst}; each is
    exactly the pCN kernel of pcn_mh (same Gaussian reference from W, same rho), so each marginal
    chain is a valid MH chain for its own target. The two chains share the proposal noise xi and
    the accept uniform U. In whitened coordinates the proposals satisfy
        t'_dst - t'_src = rho (t_dst - t_src),
    so the pair distance shrinks by rho when both accept, is unchanged when both reject, and the
    pair can only separate when the two accept decisions differ. One PDE solve per sweep at each
    of the two levels.
    """
    ts, td = whiten(z_src.double(), W), whiten(z_dst.double(), W)
    ls, ld = log_g_src.clone(), log_g_dst.clone()
    step = math.sqrt(max(0.0, 1.0 - rho * rho))
    n = len(ts)
    acc_s = acc_d = prob_s = prob_d = disagree = both = 0.0
    dist0 = float((td - ts).square().sum(-1).mean())
    for _ in range(sweeps):
        xi = torch.randn(ts.shape, dtype=torch.float64, generator=generator)
        log_u = torch.rand(n, dtype=torch.float64, generator=generator).log()
        ps, pd = rho * ts + step * xi, rho * td + step * xi
        with clock.measure("pde_mcmc"):
            lgs = problem.log_density_z(unwhiten(ps, W), level_src)
            lgd = problem.log_density_z(unwhiten(pd, W), level_dst)
        las = (lgs + 0.5 * ps.square().sum(-1)) - (ls + 0.5 * ts.square().sum(-1))
        lad = (lgd + 0.5 * pd.square().sum(-1)) - (ld + 0.5 * td.square().sum(-1))
        if torch.isnan(las).any() or torch.isnan(lad).any():
            raise FloatingPointError("NaN in coupled pCN acceptance ratio")
        a_s, a_d = log_u < las, log_u < lad
        ts[a_s], ls[a_s] = ps[a_s], lgs[a_s]
        td[a_d], ld[a_d] = pd[a_d], lgd[a_d]
        acc_s += float(a_s.sum()); acc_d += float(a_d.sum())
        prob_s += float(las.clamp(max=0).exp().sum()); prob_d += float(lad.clamp(max=0).exp().sum())
        disagree += float((a_s != a_d).sum()); both += float((a_s & a_d).sum())
    m = max(1, n * sweeps)
    diag = _mh_summary("pcn_coupled", acc_d, prob_d, n, sweeps, {
        "pcn_rho": rho,
        "mh_source_acceptance_fraction": acc_s / m,
        "mh_source_mean_acceptance_probability": prob_s / m,
        "mh_accept_disagreement_fraction": disagree / m,
        "mh_both_accept_fraction": both / m,
        "pair_msd_whitened_before_mh": dist0,
        "pair_msd_whitened_after_mh": float((td - ts).square().sum(-1).mean()),
    })
    return unwhiten(ts, W), ls, unwhiten(td, W), ld, diag


@torch.no_grad()
def lightsb_mh(problem, z, log_g, level, model, W, sweeps, clock):
    """Metropolis-Hastings with the LightSB conditional K_theta as proposal:
        alpha = min{1, g(y) K(y,x) / (g(x) K(x,y))}   (whitened coordinates).
    Exact for gamma_level whatever the training quality; one PDE solve per sweep.
    Proposal randomness uses the global torch RNG (seeded per run).
    """
    dtype = model.r.dtype
    t = whiten(z.double(), W)
    log_g = log_g.clone()
    accepted, probability = 0.0, 0.0
    tracker = _ModeTracker(problem, z.double())
    for _ in range(sweeps):
        proposal = model.sample(t.to(dtype)).double()
        forward = log_kernel_pairs(model, t, proposal)
        backward = log_kernel_pairs(model, proposal, t)
        z_prop = unwhiten(proposal, W)
        with clock.measure("pde_mcmc"):
            lg = problem.log_density_z(z_prop, level)
        log_a = (lg + backward) - (log_g + forward)
        if torch.isnan(log_a).any():
            raise FloatingPointError("NaN in LightSB-MH acceptance ratio")
        accept = torch.rand(len(t), dtype=torch.float64).log() < log_a
        tracker.update(z_prop, accept)
        t[accept] = proposal[accept]
        log_g[accept] = lg[accept]
        accepted += float(accept.sum())
        probability += float(log_a.clamp(max=0).exp().sum())
    return unwhiten(t, W), log_g, _mh_summary("lightsb", accepted, probability, len(t), sweeps,
                                              tracker.summary(len(t), sweeps))


def run_transport_mh(problem, initial_z, N, settings, mh, seed):
    """YOUR SCHEME: LightSB transport pi_{l-1} -> pi_l, then pCN-MH towards pi_l.

    Per transition l-1 -> l (same fit/whitening/epsilon schedule as LightSB):
      1. ratio gamma_l/gamma_{l-1} on the source cloud; fit LightSB (whitened);
      2. children Y_i ~ K_theta(x_{a_i}, .)  (law q_l given the cloud);
      3. n = transport_mh.sweeps pCN-MH sweeps leaving pi_l invariant, reference
         Gaussian = full-covariance fit of the children, rho = transport_mh.rho
         ('auto' = S(eps_l), matching the coupling noise);
      4. paired estimator term mean[phi(Y'_i) - phi(x_{a_i})].
    Law after MH is q_l P^n, NOT pi_l: bias is reduced, not removed.
    sweeps=0 reproduces run_light_sb (whiten/init/correction='none') exactly.
    gamma_l of the final cloud is reused as gamma_{l} for the next ratio.
    Diagnostic phi of the pre-MH children is evaluated untracked/untimed.
    """
    if whitening_options(settings)["correction"] != "none":
        raise ValueError("TransportMH requires light.correction='none'")
    torch.manual_seed(seed)
    problem.reset_work()
    clock = Timers()
    start = time.perf_counter()
    dtype = torch.float64 if settings["dtype"] == "float64" else torch.float32
    wopts = whitening_options(settings)
    sweeps = mh["sweeps"]
    with clock.measure("sampler"):
        current = initial_z[:N[0]].to(dtype=dtype).clone()
    with clock.measure("estimator"):
        phi_current = problem.phi_z(current.double())
        terms = [entry(0, N[0], phi_current.mean(), phi_current.var(unbiased=True))]
    log_g_current = None   # cached log gamma_{level-1} on `current`, if known
    diagnostics = []
    for level in range(1, len(N)):
        before = dict(clock.values)
        with clock.measure("sampler"):
            z64 = current.double()
            with clock.measure("pde_reweight"):
                if log_g_current is None:
                    log_g_prev = problem.log_density_z(z64, level - 1)
                    log_ratio = problem.log_density_z(z64, level) - log_g_prev
                else:
                    log_g_prev = log_g_current
                    log_ratio = problem.log_density_z(z64, level) - log_g_current
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError("Non-finite LightSB density ratios")
            with clock.measure("fit"):
                W = fit_whitening(current, wopts["whiten"], wopts["whiten_shrinkage"])
                source_t = whiten(current, W)
                stage_settings = level_settings(settings, level)
                model, loss, ess = fit_light_sb(source_t, log_ratio, stage_settings)
            with clock.measure("propagation"):
                count = N[level]
                if count <= len(current):
                    ids = torch.randperm(len(current))[:count]
                else:
                    ids = torch.randint(len(current), (count,))
                children = unwhiten(model.sample(source_t[ids]), W).to(dtype=dtype)
                if not torch.isfinite(children).all():
                    raise FloatingPointError("Non-finite LightSB samples")
        pre_mh = children
        parents_moved = None
        log_g_children, mh_diag = None, {"mh_sweeps": 0}
        level_sweeps = sweeps
        training_ess = ratio_weight_stats(normalized_ratio_weights(log_ratio))["ess_fraction"]
        if mh["rho"] == "ess":
            rho, level_sweeps = ess_adaptive_rho(training_ess, stage_settings["epsilon"], mh)
        else:
            rho = pcn_rho(mh["rho"], stage_settings["epsilon"])
        if level_sweeps > 0:
            with clock.measure("sampler"):
                with clock.measure("mcmc"):
                    with clock.measure("pde_mcmc"):
                        log_g_children = problem.log_density_z(children.double(), level)
                    Wc = fit_whitening(children, "full", mh["shrinkage"])
                    if mh.get("coupled", False):
                        parents_moved, _, moved, log_g_children, mh_diag = coupled_pcn_mh(
                            problem, current[ids], log_g_prev[ids], level - 1,
                            children, log_g_children, level, rho, level_sweeps, Wc, clock)
                        parents_moved = parents_moved.to(dtype=dtype)
                    else:
                        moved, log_g_children, mh_diag = pcn_mh(problem, children, log_g_children,
                                                                level, rho, level_sweeps, Wc, clock)
                    children = moved.to(dtype=dtype)
        with clock.measure("estimator"):
            phi_next = problem.phi_z(children.double())
            if parents_moved is not None:
                # coupled variant: the correction pairs the moved parent with the moved child;
                # the unmoved parent cloud (phi_current) is the one used in the previous correction
                phi_parent = problem.phi_z(parents_moved.double())
                delta = phi_next - phi_parent
            else:
                delta = phi_next - phi_current[ids]
            terms.append(entry(level, count, delta.mean(), delta.var(unbiased=True)))
        # Diagnostics only: untracked (no work counted) and outside every timer.
        problem._tracking = False
        try:
            phi_pre = problem.phi_z(pre_mh.double()) if level_sweeps > 0 else phi_next
        finally:
            problem._tracking = True
        importance = dict(model.importance_diagnostics)
        weighted_phi = float(torch.dot(normalized_ratio_weights(log_ratio), phi_current.double()))
        importance.update({
            "source_phi_mean": float(phi_current.mean()),
            "importance_target_phi_mean": weighted_phi,
            "pre_mh_phi_mean": float(phi_pre.mean()),
            "output_phi_mean": float(phi_next.mean()),
            "pre_mh_minus_importance_phi": float(phi_pre.mean()) - weighted_phi,
            "output_minus_importance_phi": float(phi_next.mean()) - weighted_phi,
            "pre_mh_delta_phi_variance": float((phi_pre - phi_current[ids]).var(unbiased=True)),
            "epsilon_level": float(stage_settings["epsilon"]),
            "lr_level": float(stage_settings["lr"]),
            "iterations_level": int(stage_settings["iterations"]),
            "mh_rho_rule": str(mh["rho"]),
            "mh_coupled": bool(mh.get("coupled", False)),
            "mh_training_ess_fraction": float(training_ess),
            **mh_diag,
        })
        importance.update(coupling_diagnostics(current[ids] if parents_moved is None else parents_moved,
                                               children, delta, W, wopts))
        lab = problem.mode_label(children)
        if lab is not None:
            importance["right_mode_fraction"] = float(lab.double().mean())
            lab_pre = problem.mode_label(pre_mh)
            importance["right_mode_fraction_pre_mh"] = float(lab_pre.double().mean())
        diagnostics.append({"level": level, "ratio_ess": ess, "last_loss": loss,
                            "importance": importance,
                            "timing_breakdown": clock.stage_breakdown(before)})
        current, phi_current, log_g_current = children, phi_next, log_g_children
        del model
    estimate = math.fsum(t["contribution"] for t in terms)
    answer = finish_result(problem, clock, start, estimate, terms, diagnostics)
    answer["final_mean_diagnostic"] = float(phi_current.mean())
    return answer


def mutation_light_settings(light, kcfg):
    """LightSB settings for the pi_l self-transport used as an MH proposal."""
    out = dict(light, epsilon=float(kcfg["epsilon"]), iterations=kcfg["iterations"],
               lr=float(kcfg["lr"]), init=kcfg["init"], correction="none",
               epsilon_schedule="constant", lr_epsilon_scaling=False,
               weight_tempering=False)
    if kcfg["K"] is not None:
        out["K"] = kcfg["K"]
    if out["target_batching"] == "legacy_minibatch":
        out["target_batching"] = "global_resample"   # uniform weights: modes coincide
    return out


def run_mlsmc_kernel(problem, initial_z, N, light, kcfg, kind, seed):
    """MY SUGGESTION: MLSMC (Beskos eq. 5) with a pi_l-invariant mutation:
      kind='pcn': pCN-MH, reference = full Gaussian fit of the resampled cloud;
      kind='lsb': LightSB fitted as the self-transport pi_l -> pi_l on the
                  resampled cloud (uniform weights, whitened, identity init),
                  used as MH proposal with exact kernel ratio.
    Reweight -> correction term -> resample -> fit (lsb) -> MH sweeps.
    The final level is not mutated (as in run_smc). Identical estimator to MLSMC;
    only the mutation (and therefore cost/mixing) differs.
    """
    if kind not in ("pcn", "lsb"):
        raise ValueError("kind must be 'pcn' or 'lsb'")
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 7)
    problem.reset_work()
    clock = Timers()
    start = time.perf_counter()
    L = len(N) - 1
    msettings = mutation_light_settings(light, kcfg)
    mdtype = torch.float64 if msettings["dtype"] == "float64" else torch.float32
    with clock.measure("sampler"):
        current = initial_z[:N[0]].double().clone()
        if L > 0:
            with clock.measure("pde_reweight"):
                logp = problem.log_density_z(current, 0)
    with clock.measure("estimator"):
        phi_current = problem.phi_z(current)
        terms = [entry(0, N[0], phi_current.mean(), phi_current.var(unbiased=True))]
    diagnostics = []
    for level in range(1, L + 1):
        before = dict(clock.values)
        with clock.measure("sampler"):
            with clock.measure("pde_reweight"):
                log_next = problem.log_density_z(current, level)
            log_ratio = log_next - logp
            if not torch.isfinite(log_ratio).all():
                raise FloatingPointError("Non-finite SMC log weights")
            weights = log_ratio.softmax(0)
            diag = {"level": level, "ratio_ess": float(1.0 / weights.square().sum())}
        with clock.measure("estimator"):
            correction = torch.dot(weights, phi_current - phi_current.mean())
            terms.append(entry(level, len(current), correction))
        if level == L:
            diag["final_mutation_skipped"] = True
            diag["timing_breakdown"] = clock.stage_breakdown(before)
            diagnostics.append(diag)
            break
        with clock.measure("sampler"):
            with clock.measure("resampling"):
                ids = torch.multinomial(weights, N[level], replacement=True, generator=generator)
                cloud_before = current
                current = current[ids].clone()
                logp = log_next[ids].clone()
                diag["unique_after_resampling"] = int(torch.unique(ids).numel())
            with clock.measure("fit"):
                try:
                    W = fit_whitening(current, "full", kcfg["shrinkage"])
                except FloatingPointError:
                    # Resampling kept a single distinct particle (ESS ~ 1): take the pCN reference
                    # from the pre-resampling cloud instead (any fixed reference is valid for pCN-MH).
                    W = fit_whitening(cloud_before, "full", kcfg["shrinkage"])
                    diag["whitening_fallback"] = "pre-resampling cloud"
                if kind == "lsb":
                    source_t = whiten(current.to(dtype=mdtype), W)
                    zeros = torch.zeros(len(current), dtype=torch.float64)
                    with torch.enable_grad():
                        model, loss, _ = fit_light_sb(source_t, zeros, msettings)
                    diag["mutation_fit_last_loss"] = loss
            ess_fraction = diag["ratio_ess"] / len(weights)
            level_sweeps = mlsmc_level_sweeps(ess_fraction, kcfg)
            diag["ratio_ess_fraction"] = float(ess_fraction)
            diag["level_sweeps"] = int(level_sweeps)
            with clock.measure("mcmc"):
                if kind == "pcn":
                    rho = pcn_rho(kcfg["rho"], kcfg["epsilon"])
                    current, logp, mdiag = pcn_mh(problem, current, logp, level, rho,
                                                  level_sweeps, W, clock, generator)
                else:
                    current, logp, mdiag = lightsb_mh(problem, current, logp, level, model, W,
                                                      level_sweeps, clock)
                    del model
            diag.update(mdiag)
            diag["mutation_epsilon"] = float(kcfg["epsilon"])
            lab = problem.mode_label(current)
            if lab is not None:
                diag["right_mode_fraction"] = float(lab.double().mean())
        with clock.measure("estimator"):
            phi_current = problem.phi_z(current)
        diag["timing_breakdown"] = clock.stage_breakdown(before)
        diagnostics.append(diag)
    with clock.measure("estimator"):
        estimate = math.fsum(t["contribution"] for t in terms)
    return finish_result(problem, clock, start, estimate, terms, diagnostics)


def print_mh_diagnostics(record):
    for stage in record.get("diagnostics", []):
        d = stage.get("importance", stage)
        if d.get("mh_sweeps"):
            extra = (f", pre-MH transport err={d['pre_mh_minus_importance_phi']:+.4g}"
                     f" -> post={d['output_minus_importance_phi']:+.4g}"
                     if "pre_mh_minus_importance_phi" in d else "")
            print(f"  level {stage['level']}: {d['mh_kernel']}-MH x{d['mh_sweeps']} "
                  f"accept={d['mh_acceptance_fraction']:.3f}"
                  + (f" rho={d['pcn_rho']:.3f}" if "pcn_rho" in d else "") + extra, flush=True)


def target_description(problem):
    if problem.problem_name in ("darcy2d", "ns2d"):
        return {"dimension": problem.dimension, "y": problem.stored_y(), "noise_sd": problem.noise_sd,
                **problem.description()}
    desc = {"dimension": problem.dimension, "y": problem.stored_y(), "noise_sd": problem.noise_sd,
            "coefficient": "0.15 + sum_j 0.4*4^(-j)*u_j*b_j(x)",
            "posterior_reference": "continuum quadrature likelihood",
            "observable": "same continuum-quadrature p(0.5;u) at every level",
            "quadrature_order_per_quarter": problem.reference_order}
    if problem.problem_name != "beskos":       # Beskos description (and hash) unchanged
        desc.update(problem.description())
    return desc


IS_REFERENCE_DEFAULTS = {"is_samples": 50_000_000, "is_pilot": 5000, "is_informed_dims": 4,
                         "is_df": 5, "is_inflation": 1.2, "is_defensive": 0.05,
                         "is_batch": 65536, "is_batches_for_se": 20,
                         "is_split_coord": None, "is_split_threshold": 0.0, "is_min_component_weight": 0.1}


def _log_prior_z(z):
    """log density of z = atanh(u), u ~ Uniform(-1,1), per coordinate (summed)."""
    a = z.abs()
    return (math.log(2.0) - 2.0 * a - 2.0 * F.softplus(-2.0 * a)).sum(-1)


@torch.no_grad()
def importance_sampling_reference(problem, spec):
    """Self-normalised importance sampling reference for the continuum posterior.

    Proposal on z = atanh(u) (product form):
      * first k = is_informed_dims coordinates (the data-informed ones):
        defensive mixture (1-delta) * t_df(m, (infl^2) C) + delta * prior,
        with (m, C) from an iid rejection pilot of the continuum posterior;
      * remaining coordinates: exactly the prior (cancels in the weight).
    Weight: w(z) = L(u) * prior_A(z_A) / q_A(z_A) <= L(u) / delta <= 1 / delta,
    because the likelihood exp(-Phi) <= 1. Bounded weights => finite variance, so
    the delta-method standard error is valid; an independent-batch SE is also
    reported as a cross-check. Likelihood and phi share one continuum solve.
    """
    s = {k: spec.get(k, v) for k, v in IS_REFERENCE_DEFAULTS.items()}
    k, df, infl, delta = int(s["is_informed_dims"]), int(s["is_df"]), float(s["is_inflation"]), float(s["is_defensive"])
    M, batch, n_se = int(s["is_samples"]), int(s["is_batch"]), int(s["is_batches_for_se"])
    d = problem.dimension
    if not (1 <= k <= d and df >= 1 and infl > 0 and 0 < delta < 1 and M >= n_se * 2 and batch >= 1):
        raise ValueError("Invalid importance-sampling reference settings")
    # 1. iid pilot from the continuum posterior (vectorised, untimed).
    z_pilot, pilot_info = problem.sample_posterior(int(s["is_pilot"]), level=None, seed=int(spec["seed"]),
                                                   return_z=True, dtype=torch.float64,
                                                   max_proposals=int(spec["max_proposals"]))
    # Optional mixture: split the pilot on one informed coordinate (separated modes) and fit one
    # t component per group; weights = group fractions floored at is_min_component_weight.
    split = s["is_split_coord"]
    if split is None:
        groups = [torch.ones(len(z_pilot), dtype=torch.bool)]
    else:
        if not 0 <= int(split) < k:
            raise ValueError("is_split_coord must index one of the informed dimensions")
        thr = float(s["is_split_threshold"])
        groups = [z_pilot[:, int(split)] <= thr, z_pilot[:, int(split)] > thr]
    comp_m, comp_chol, comp_const, comp_w = [], [], [], []
    for gmask in groups:
        Zg = z_pilot[gmask][:, :k]
        if len(Zg) < max(2 * k, 20):
            continue                                   # too few pilot points: no component
        chol_g = torch.linalg.cholesky(torch.atleast_2d(torch.cov(Zg.T))) * infl
        comp_m.append(Zg.mean(0)); comp_chol.append(chol_g); comp_w.append(len(Zg))
        comp_const.append(math.lgamma((df + k) / 2) - math.lgamma(df / 2) - 0.5 * k * math.log(df * math.pi)
                          - float(torch.log(torch.diagonal(chol_g)).sum()))
    if not comp_m:
        raise RuntimeError("IS reference: no pilot group large enough to fit a proposal")
    comp_w = torch.tensor(comp_w, dtype=torch.float64); comp_w = comp_w / comp_w.sum()
    if len(comp_w) > 1:
        comp_w = comp_w.clamp_min(float(s["is_min_component_weight"])); comp_w = comp_w / comp_w.sum()
    comp_m, comp_chol = torch.stack(comp_m), torch.stack(comp_chol)
    comp_const = torch.tensor(comp_const, dtype=torch.float64)
    m, chol = comp_m[0], comp_chol[0]              # single-component path (unchanged draws)
    t_const = float(comp_const[0])
    pilot_phi = problem.phi_u(torch.tanh(z_pilot))
    centre = float(pilot_phi.mean())          # centring only for numerical accuracy
    generator = torch.Generator().manual_seed(int(spec["seed"]) + 17)
    eps = torch.finfo(torch.float64).eps
    old_batch = problem.parameter_batch
    problem.parameter_batch = max(old_batch, 8192)
    # Streaming sums, rescaled to a running maximum log-weight.
    ref_lw = -math.inf
    S0 = S1 = S00 = S01 = S11 = 0.0           # sum w, w*psi, w^2, w^2*psi, w^2*psi^2
    chunk_S0, chunk_S1, chunk_ref = [], [], []
    per_chunk = M // n_se
    done = 0
    max_w_share_num = 0.0
    try:
        while done < M:
            n = min(batch, M - done, per_chunk - (done % per_chunk))
            # proposal draws
            x = torch.randn((n, k), dtype=torch.float64, generator=generator)
            g = torch.randn((n, df), dtype=torch.float64, generator=generator).square().sum(-1) / df
            if len(comp_w) == 1:
                zA = m + (x / g.sqrt()[:, None]) @ chol.T
            else:
                comp = torch.multinomial(comp_w, n, replacement=True, generator=generator)
                zA = comp_m[comp] + torch.einsum("nij,nj->ni", comp_chol[comp], x / g.sqrt()[:, None])
            pick = torch.rand(n, dtype=torch.float64, generator=generator) < delta
            if pick.any():
                uA = (2 * torch.rand((int(pick.sum()), k), dtype=torch.float64, generator=generator) - 1)
                zA[pick] = torch.atanh(uA.clamp(-1 + eps, 1 - eps))
            uB = 2 * torch.rand((n, d - k), dtype=torch.float64, generator=generator) - 1
            zB = torch.atanh(uB.clamp(-1 + eps, 1 - eps))
            if len(comp_w) == 1:
                qstd = torch.linalg.solve_triangular(chol, (zA - m).T, upper=False).T
                log_t = t_const - 0.5 * (df + k) * torch.log1p(qstd.square().sum(-1) / df)
            else:
                comps = []
                for c_i in range(len(comp_w)):
                    q_c = torch.linalg.solve_triangular(comp_chol[c_i], (zA - comp_m[c_i]).T, upper=False).T
                    comps.append(math.log(float(comp_w[c_i])) + float(comp_const[c_i])
                                 - 0.5 * (df + k) * torch.log1p(q_c.square().sum(-1) / df))
                log_t = torch.logsumexp(torch.stack(comps), dim=0)
            log_pA = _log_prior_z(zA)
            log_qA = torch.logaddexp(math.log1p(-delta) + log_t, math.log(delta) + log_pA)
            u = torch.tanh(torch.cat([zA, zB], dim=1)).clamp(-1.0, 1.0)
            log_lik, phi_vals = problem.continuum_loglik_phi(u)
            psi = phi_vals - centre
            lw = log_lik + log_pA - log_qA
            if not torch.isfinite(lw).all() or not torch.isfinite(psi).all():
                raise FloatingPointError("Non-finite importance weight or observable")
            new_ref = max(ref_lw, float(lw.max()))
            if new_ref > ref_lw:
                scale = math.exp(ref_lw - new_ref) if math.isfinite(ref_lw) else 0.0
                S0 *= scale; S1 *= scale; S00 *= scale ** 2; S01 *= scale ** 2; S11 *= scale ** 2
                max_w_share_num *= scale
                ref_lw = new_ref
            w = torch.exp(lw - ref_lw)
            S0 += float(w.sum()); S1 += float((w * psi).sum())
            w2 = w.square()
            S00 += float(w2.sum()); S01 += float((w2 * psi).sum()); S11 += float((w2 * psi.square()).sum())
            max_w_share_num = max(max_w_share_num, float(w.max()))
            # per-chunk ratio estimators (independent batches) for a cross-check SE
            if not chunk_S0 or (done % per_chunk) == 0:
                chunk_S0.append(0.0); chunk_S1.append(0.0); chunk_ref.append(ref_lw)
            c = len(chunk_S0) - 1
            if ref_lw != chunk_ref[c]:
                sc = math.exp(chunk_ref[c] - ref_lw)
                chunk_S0[c] *= sc; chunk_S1[c] *= sc; chunk_ref[c] = ref_lw
            chunk_S0[c] += float(w.sum()); chunk_S1[c] += float((w * psi).sum())
            done += n
            if done % (10 * batch) < n or done == M:
                print(f"  IS reference: {done}/{M} samples, ESS so far {S0 * S0 / S00:.0f}", flush=True)
    finally:
        problem.parameter_batch = old_batch
    mu_psi = S1 / S0
    var_num = S11 - 2 * mu_psi * S01 + mu_psi * mu_psi * S00          # sum w^2 (psi - mu)^2
    se = math.sqrt(max(var_num, 0.0)) / S0
    chunk_est = [b / a for a, b in zip(chunk_S0, chunk_S1)]
    batch_se = float(np.std(chunk_est, ddof=1) / math.sqrt(len(chunk_est)))
    ess = S0 * S0 / S00
    return {
        "value": centre + mu_psi, "se": se, "batch_se": batch_se,
        "ess": ess, "ess_fraction": ess / M, "samples": M,
        "max_single_weight_share": max_w_share_num / S0,
        "batch_estimates": [centre + e for e in chunk_est],
        "proposal": {"informed_dims": k, "df": df, "inflation": infl, "defensive": delta,
                     "pilot_size": int(s["is_pilot"]), "pilot": pilot_info,
                     "pilot_mean_informed": m.tolist(), "split_coord": split,
                     "component_weights": comp_w.tolist(), "component_means": comp_m.tolist(),
                     "pilot_sd_over_prior_sd_first8": (z_pilot.std(0)[:8] / 0.9069).tolist()},
        "note": "Self-normalised IS, bounded weights (<= 1/defensive); SE by delta method, "
                "batch_se from independent sub-runs as a cross-check.",
    }


def compute_reference(problem, spec, out, external_file=None):
    """ONE reference for ALL horizons. Its error is not treated as known zero."""
    target = target_description(problem)
    signature = canonical_hash(target)
    path = out / "reference.json"
    candidate = Path(external_file) if external_file else path
    if candidate.exists():
        data = json.loads(candidate.read_text())
        if data.get("target_hash") != signature:
            raise ValueError("Reference targets a different dimension/data/observable. Do not reuse it.")
        if external_file:
            atomic_json(path, data)
        print(f"Reference loaded: {data['value']:.10g}" if data["value"] is not None
              else "Reference loaded: none", flush=True)
        return data
    mode = spec["mode"]
    if mode == "auto":
        mode = "quadrature" if problem.dimension == 2 else "mcmc"
    if problem.problem_name in ("darcy2d", "ns2d") and mode != "none":
        raise ValueError(f"{problem.problem_name}: use reference.mode 'none' plus make_reference_from_runs.py, "
                         "or --reference-file")
    start = time.perf_counter()
    info = {"target": target, "target_hash": signature, "mode": mode, "se": None,
            "quadrature_change": None}
    print(f"Computing common continuum-posterior reference ({mode})...", flush=True)
    if mode == "quadrature":
        if problem.dimension != 2:
            raise ValueError("Tensor reference quadrature is only implemented for dimension=2")
        values = [problem.reference_2d(None, order=int(o)) for o in spec["orders"]]
        info.update(value=values[-1], orders=spec["orders"], values=values,
                    quadrature_change=abs(values[-1] - values[-2]),
                    note="Order-change diagnostic, not a rigorous integration-error bound.")
    elif mode == "mcmc":
        ref = blocked_rwm_reference(problem, level=None, n_chains=spec["chains"],
                                   draws=spec["draws"], burn_in=spec["burn_in"], seed=spec["seed"])
        info.update(value=ref["estimate"], se=ref["se"], chain_means=ref["chain_means"].tolist(),
                    block_acceptance=ref["block_acceptance"].tolist(),
                    note="SE from independent chain means; finite-chain uncertainty remains.")
    elif mode == "rejection":
        u, rejection = problem.sample_posterior(spec["samples"], level=None, seed=spec["seed"],
                                               return_z=False, dtype=torch.float64,
                                               max_proposals=spec["max_proposals"])
        vals = problem.phi_u(u)
        info.update(value=float(vals.mean()), se=float(vals.std(unbiased=True) / math.sqrt(len(vals))),
                    rejection=rejection, samples=len(vals), note="IID numerical-posterior reference.")
    elif mode == "is":
        info.update(importance_sampling_reference(problem, spec))
    elif mode == "none":
        # For a run whose purpose is to BUILD a reference (see make_reference_from_runs.py):
        # no errors or plots are computed for it.
        info.update(value=None, note="No reference: build one with make_reference_from_runs.py")
    else:
        raise ValueError("reference.mode must be auto, quadrature, mcmc, rejection, is or none")
    info["seconds"] = time.perf_counter() - start
    atomic_json(path, info)
    if info["value"] is None:
        print("No reference value (reference.mode='none').", flush=True)
        return info
    print(f"Reference={info['value']:.10g}; SE={info['se']}; quadrature change={info['quadrature_change']}", flush=True)
    if mode == "is":
        print(f"  IS: ESS={info['ess']:.0f} ({info['ess_fraction']:.4f}), batch SE={info['batch_se']:.3g}, "
              f"largest single weight share={info['max_single_weight_share']:.2e}", flush=True)
    return info


def figure4(problem, spec, out):
    """PDE convergence at one fixed u; not an algorithm-performance comparison."""
    path = out / "figure4_data.json"
    if path.exists():
        return json.loads(path.read_text())
    if hasattr(problem, "pde_convergence"):
        data = problem.pde_convergence(spec["levels"], spec["seed"])
        atomic_json(path, data)
        write_csv(out / "figure4_fem_errors.csv", data["rows"])
        return data
    generator = torch.Generator().manual_seed(spec["seed"])
    if spec.get("u") is None:
        u = 1.8 * torch.rand(problem.dimension, dtype=torch.float64, generator=generator) - 0.9
    else:
        u = torch.as_tensor(spec["u"], dtype=torch.float64)
    rows = []
    for level in spec["levels"]:
        _, pc = problem.fem_nodes(u, level - 1)
        x, pf = problem.fem_nodes(u, level)
        n = len(x) - 1
        h = 1.0 / n
        coarse_prolongated = torch.empty_like(pf)
        coarse_prolongated[::2] = pc
        coarse_prolongated[1::2] = 0.5 * (pc[:-1] + pc[1:])
        difference = pf - coarse_prolongated
        seminorm2 = float((difference[1:] - difference[:-1]).square().sum() / h)
        # Exact L2 norm of a linear function on each fine interval.
        L2 = float(h / 3.0 * (difference[:-1].square() + difference[:-1] * difference[1:]
                              + difference[1:].square()).sum())
        rows.append({"level": level, "h": h, "H01_seminorm_squared": seminorm2,
                     "H1_full_norm_squared": seminorm2 + L2, "L2_norm_squared": L2})
    data = {"u": u.tolist(), "rows": rows, "primary_norm": "H01 gradient seminorm squared"}
    atomic_json(path, data)
    write_csv(out / "figure4_fem_errors.csv", rows)
    return data


def bootstrap_stat(v, statistic, draws, rng):
    v = np.asarray(v, dtype=float)
    if len(v) < 2 or draws < 1:
        return None
    samples = v[rng.integers(0, len(v), size=(draws, len(v)))]
    results = samples.mean(1) if statistic == "mean" else samples.var(1, ddof=1)
    return np.quantile(results, [0.025, 0.975]).tolist()


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_records(out):
    return [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]




def print_importance_diagnostics(result):
    """Do not let a high warm-up ESS hide a low ESS for the actual final weights."""
    for stage in result.get("diagnostics", []):
        d = stage.get("importance")
        if d is None:
            continue
        print(f"  level {stage['level']}: raw ESS={d['raw_ess']:.1f}/{d['source_particles']} "
              f"({d['raw_ess_fraction']:.3f}), maxW={d['raw_max_weight']:.4f}, "
              f"beta={d['initial_beta']:.4f}->1, "
              f"full-weight updates={d['untempered_updates']}/{d['total_updates']}, "
              f"output-minus-weighted phi={d['output_minus_importance_phi']:+.4g}", flush=True)
        if "pair_corr_coord1" in d:
            cond = (d["whitening"] or {}).get("condition_number")
            print(f"    whiten={d['whiten']}"
                  + (f" (cond={cond:.3g})" if cond is not None else "")
                  + f", pair corr z1={d['pair_corr_coord1']:.3f}, median={d['pair_corr_median']:.3f}, "
                  f"child/parent sd z1={d['child_sd_over_parent_sd_coord1']:.3f}, "
                  f"Var(dphi)={d['delta_phi_variance']:.4g}", flush=True)
        if "epsilon_level" in d:
            print(f"    epsilon_l={d['epsilon_level']:.4g}, lr_l={d['lr_level']:.3g}", flush=True)
        if "is_ess_fraction" in d:
            print(f"    IS correction: ESS={d['is_ess_fraction']:.3f}, maxW={d['is_max_weight']:.4f}, "
                  f"paired corr: unweighted={d['unweighted_paired_correction']:+.4g} "
                  f"-> IS={d['is_paired_correction']:+.4g}, q-eval {d['marginal_density_seconds']:.2f}s",
                  flush=True)
        if d["raw_ess_below_5_percent"]:
            print("    WARNING: raw ESS < 5%. Warm-up does not repair final weight degeneracy "
                  "or missing source support.", flush=True)


def export_importance_diagnostics(records, out):
    """One row per stored PDE transition; no new solver/estimator evaluations."""
    rows = []
    for record in records:
        if record["method"] not in ("LightSB", "TransportMH"):
            continue
        for stage in record.get("diagnostics", []):
            d = stage.get("importance")
            if d is None:  # Old records remain readable, without invented diagnostics.
                continue
            row = {"configuration": record["configuration"], "repeat": record["repeat"],
                   "level": stage["level"], **{k: v for k, v in d.items() if k != "schedule"}}
            row["beta_schedule"] = [s["beta"] for s in d["schedule"]]
            row["ess_fraction_schedule"] = [s["ess_fraction"] for s in d["schedule"]]
            rows.append(row)
    if rows:
        write_csv(out / "importance_diagnostics.csv", rows)
        groups = sorted({(r["configuration"], r["level"]) for r in rows})
        summary = []
        for name, level in groups:
            rr = [r for r in rows if r["configuration"] == name and r["level"] == level]
            summary.append({"configuration": name, "level": level, "runs": len(rr),
                            "tempered_runs": sum(r["tempering_active"] for r in rr),
                            "mean_raw_ess_fraction": float(np.mean([r["raw_ess_fraction"] for r in rr])),
                            "min_raw_ess_fraction": min(r["raw_ess_fraction"] for r in rr),
                            "mean_initial_beta": float(np.mean([r["initial_beta"] for r in rr])),
                            "mean_raw_max_weight": float(np.mean([r["raw_max_weight"] for r in rr])),
                            "mean_output_minus_importance_phi": float(np.mean(
                                [r["output_minus_importance_phi"] for r in rr]))})
        write_csv(out / "importance_summary.csv", summary)

def plot_results(out):
    meta = json.loads((out / "metadata.json").read_text())
    cfg = meta["config"]
    reference = json.loads((out / "reference.json").read_text())
    records = load_records(out)
    if not records:
        raise RuntimeError("No completed records found")
    target = reference["value"]
    methods = cfg["methods"]
    groups = {(c["name"], method): [r for r in records if r["configuration"] == c["name"] and r["method"] == method]
              for c in cfg["configurations"] for method in methods}
    summary = []
    rng = np.random.default_rng(67123)
    for c in cfg["configurations"]:
        for method in methods:
            rows = groups[(c["name"], method)]
            if not rows:
                continue
            estimates = np.array([r["estimate"] for r in rows])
            errors = estimates - target
            se2 = errors ** 2
            ci = bootstrap_stat(se2, "mean", cfg["bootstrap_draws"], rng)
            summary.append({"configuration": c["name"], "method": method, "L": len(c["N"]) - 1,
                            "repeats_completed": len(rows), "mean_estimate": float(estimates.mean()),
                            "bias_vs_reference": float(errors.mean()),
                            "variance_over_runs": float(estimates.var(ddof=1)) if len(rows) > 1 else None,
                            "L1": float(np.abs(errors).mean()), "L2_RMSE": float(np.sqrt(se2.mean())),
                            "MSE": float(se2.mean()), "MSE_bootstrap_low": None if ci is None else ci[0],
                            "MSE_bootstrap_high": None if ci is None else ci[1],
                            # Relative to the reference value (target != 0 for every problem here).
                            "relative_bias": float(errors.mean()) / abs(target),
                            "relative_MSE": float(se2.mean()) / target ** 2,
                            "relative_RMSE": float(np.sqrt(se2.mean())) / abs(target),
                            "sampler_seconds": float(np.mean([r["sampler_seconds"] for r in rows])),
                            "estimator_seconds": float(np.mean([r["estimator_seconds"] for r in rows])),
                            "total_seconds": float(np.mean([r["total_seconds"] for r in rows])),
                            "fem_element_work": float(np.mean([r["fem_element_work"] for r in rows])),
                            "fem_solves_raw": float(np.mean([r["fem_solves_raw"] for r in rows])),
                            "continuum_forward_evals": float(np.mean([r["continuum_forward_evals"] for r in rows]))})
            # Never infer old sub-timings or silently use zero for missing data.
            profiled = sum(all(r.get(key) is not None for key in PROFILE_COLUMNS) for r in rows)
            summary[-1]["timing_profiled_runs"] = profiled
            for key in PROFILE_COLUMNS:
                summary[-1][key] = (float(np.mean([r[key] for r in rows]))
                                   if profiled == len(rows) else None)
    write_csv(out / "figure3_summary.csv", summary)
    timing_keys = ("configuration", "method", "L", "repeats_completed", "timing_profiled_runs",
                   "sampler_seconds", *PROFILE_COLUMNS, "estimator_seconds", "total_seconds",
                   "fem_solves_raw", "fem_element_work")
    write_csv(out / "runtime_breakdown.csv", [{key: s[key] for key in timing_keys} for s in summary])
    write_csv(out / "runs.csv", [{k: r[k] for k in r if k != "schema"} for r in records])
    export_importance_diagnostics(records, out)
    # Correct units across L: element work is NOT divided by the run's finest mesh.
    for key, ylabel, suffix in (("total_seconds", "Mean total seconds (sampling + estimator)", "time"),
                                ("fem_element_work", r"Mean FEM work $\sum_\ell n_{\mathrm{solves},\ell}/h_\ell$", "pde_work")):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        plotted = False
        for method in methods:
            items = [s for s in summary if s["method"] == method and s["MSE"] > 0 and s[key] > 0]
            if not items:
                continue
            # Keep configuration order: do not hide nonmonotone curves by sorting MSE.
            ax.plot([s["MSE"] for s in items], [s[key] for s in items], marker="o", label=LABELS[method])
            for s in items:
                ax.annotate(s["configuration"], (s["MSE"], s[key]), xytext=(4, 5),
                            textcoords="offset points", fontsize=8)
            plotted = True
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Empirical MSE against one common reference")
        ax.set_ylabel(ylabel)
        ax.set_title("Figure 3 analogue: cost versus MSE")
        ax.grid(True, which="both", alpha=0.25)
        if plotted:
            ax.legend()
        save_plot(fig, out / f"figure3_{suffix}_vs_mse.png")
    # Reference uncertainty is not included in the conditional bootstrap intervals.
    min_rmse = min(s["L2_RMSE"] for s in summary)
    uncertainty = reference.get("se") or reference.get("quadrature_change") or 0.0
    if uncertainty > 0.1 * min_rmse:
        warnings.warn("Reference uncertainty/order-change exceeds 10% of the smallest observed RMSE. "
                      "Improve the reference before interpreting the low-error part of Figure 3.")
    data4 = json.loads((out / "figure4_data.json").read_text())
    rows4 = data4["rows"]
    xx = np.array([r["h"] for r in rows4])
    yy = np.array([r["H01_seminorm_squared"] for r in rows4])
    positive = (xx > 0) & (yy > 0)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.plot(xx[positive], yy[positive], marker="o", label="Shared FEM hierarchy (fixed u)")
    rates = {"figure4_beta": None}
    if positive.sum() >= 2:
        b, a = np.polyfit(np.log(xx[positive]), np.log(yy[positive]), 1)
        rates["figure4_beta"] = float(b)
        ax.plot(xx[positive], np.exp(a) * xx[positive] ** b, linestyle="--", label=f"Log-log fit: {b:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"Fine mesh width $h_\ell$")
    ax.set_ylabel(r"$\|p_\ell-p_{\ell-1}\|_{H_0^1}^2$")
    ax.set_title("Figure 4 analogue: PDE convergence, not sampler error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_plot(fig, out / "figure4_fem_h1.png")
    # Figure 5 uses ONE fixed allocation/configuration across independent runs.
    selected = cfg["figure5_configuration"]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    summary5 = []
    rates["figure5"] = {}
    plotted = False
    for method in [m for m in methods if m != "SMC"]:
        rows = groups.get((selected, method), [])
        if len(rows) < 2:
            continue
        terms = np.array([[t["contribution"] for t in r["contributions"]] for r in rows])
        sizes = np.array([t["n_correction"] for t in rows[0]["contributions"]])
        if not all(np.array_equal(sizes, [t["n_correction"] for t in r["contributions"]]) for r in rows):
            raise ValueError("Figure 5 cannot mix different allocations at the same configuration")
        # Covariance accounts for all dependence between level contributions.
        covariance = np.atleast_2d(np.cov(terms, rowvar=False, ddof=1))
        covrows = [{"level": i, **{f"level_{j}": float(covariance[i, j]) for j in range(len(sizes))}}
                   for i in range(len(sizes))]
        write_csv(out / f"figure5_covariance_{method}.csv", covrows)
        mrows = []
        for level in range(1, terms.shape[1]):
            n = int(sizes[level])
            var = float(terms[:, level].var(ddof=1))
            interval = bootstrap_stat(terms[:, level], "variance", cfg["bootstrap_draws"], rng)
            row = {"configuration": selected, "method": method, "repeats_completed": len(rows),
                   "destination_level": level, "source_level": level - 1,
                   "h_source": 2.0 ** (-(level + 2)), "n_correction": n,
                   "mean_correction": float(terms[:, level].mean()), "variance_of_mean_correction": var,
                   "N_times_variance": n * var,
                   "bootstrap_low": None if interval is None else n * interval[0],
                   "bootstrap_high": None if interval is None else n * interval[1]}
            summary5.append(row)
            if row["N_times_variance"] > 0:
                mrows.append(row)
        if not mrows:
            continue
        x = np.array([s["h_source"] for s in mrows])
        y = np.array([s["N_times_variance"] for s in mrows])
        ax.plot(x, y, marker="o", label=LABELS[method])
        plotted = True
        rates["figure5"][method] = {
            "empirical_slope": float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) >= 3 else None,
            "variance_of_total_estimate": float(terms.sum(1).var(ddof=1)),
            "sum_of_all_covariance_entries": float(covariance.sum()),
            "sum_of_diagonal_variances_ONLY": float(np.trace(covariance))}
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"Source mesh width $h_k$ for transition $k\to k+1$")
    ax.set_ylabel(r"$n_{k,\mathrm{corr}}\,\widehat{\mathrm{Var}}_{\mathrm{runs}}(\widehat\Delta_{k+1})$")
    ax.set_title(f"Figure 5 analogue: repeated-run correction variance ({selected})")
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Need at least two completed runs with L >= 1",
                transform=ax.transAxes, ha="center")
    save_plot(fig, out / "figure5_scaled_correction_variance.png")
    write_csv(out / "figure5_corrections.csv", summary5)
    atomic_json(out / "diagnostic_rates.json", rates)
    print("\nConfiguration  Method    runs          MSE        total s       FEM work")
    for s in summary:
        print(f"{s['configuration']:13s}  {s['method']:8s} {s['repeats_completed']:4d} "
              f"{s['MSE']:12.5g} {s['total_seconds']:12.5g} {s['fem_element_work']:14.5g}")
    print("\nSampler timing breakdown: means over repetitions (seconds)")
    print(f"{'Configuration':15s} {'Method':8s} {'runs':>5s} {'PDE':>10s} {'fit':>10s} "
          f"{'propagate':>10s} {'resample':>10s} {'MCMC*':>10s} {'other':>10s} {'sampler':>10s}")
    def fmt_seconds(value):
        return "        --" if value is None else f"{value:10.4f}"
    for s in summary:
        cols = ("pde_seconds", "fit_seconds", "propagation_seconds", "resampling_seconds",
                "mcmc_other_seconds", "sampler_other_seconds", "sampler_seconds")
        print(f"{s['configuration']:15s} {s['method']:8s} {s['repeats_completed']:5d} "
              + " ".join(fmt_seconds(s[key]) for key in cols))
    print("MCMC* excludes its PDE log-density calls, which are already in PDE.")
    print("fit includes model/optimizer initialization; propagate is LightSB parent selection + conditional sampling.")
    if any(s["timing_profiled_runs"] != s["repeats_completed"] for s in summary):
        print("-- means missing profiling data in one or more old records; rerun in a new directory to measure it.")
    print(f"Timing summary: {out / 'runtime_breakdown.csv'}")
    print(f"Figures and raw data saved in {out.resolve()}")


DEFAULTS = {
    "dimension": 50, "data_seed": 2026, "run_seed": 12345, "y": None, "noise_sd": 0.25,
    "problem": "beskos", "source": dict(SOURCE_DEFAULTS),
    "repeats": 100, "threads": 1, "solver": "banded", "parameter_batch": 128,
    "observable_quadrature_order": 64, "initial_max_proposals": 20_000_000,
    "methods": ["LightSB", "MLSMC"], "bootstrap_draws": 500,
    "light": {"iterations": 2000, "batch_size": 1024, "K": 32, "epsilon": 0.5,
              "lr": 0.001, "S_init": 0.5, "full_batch": False, "dtype": "float32",
              **IMPORTANCE_DEFAULTS, **WHITEN_DEFAULTS, **EPSILON_DEFAULTS},
    "rwm": {"block_size": 5, "scale": 0.15, "sweeps": 1},
    "transport_mh": dict(TRANSPORT_MH_DEFAULTS),
    "mlsmc_kernel": dict(MLSMC_KERNEL_DEFAULTS),
    "reference": {"mode": "auto", "orders": [160, 240], "chains": 16, "draws": 2000,
                  "burn_in": 200, "seed": 9123, "samples": 20000, "max_proposals": 40_000_000},
    "figure4": {"levels": list(range(1, 9)), "seed": 745, "u": None},
    "figure5_configuration": None,
}


def merge_config(source):
    cfg = copy.deepcopy(DEFAULTS)
    for key, val in source.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    if not cfg.get("configurations"):
        raise ValueError("Config must provide an explicit list of configurations with name and N")
    if cfg["dimension"] < 1 or cfg["repeats"] < 2 or cfg["threads"] < 1:
        raise ValueError("Need dimension>=1, repeats>=2, threads>=1")
    if cfg["solver"] not in FEM_SOLVERS:
        raise ValueError(f"solver must be one of {FEM_SOLVERS}")
    seen = set()
    for c in cfg["configurations"]:
        if not c.get("name") or c["name"] in seen or not c["name"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Configuration names must be unique simple alphanumeric names")
        seen.add(c["name"])
        if not c.get("N") or any(type(n) is not int or n < 2 for n in c["N"]):
            raise ValueError("Every configuration needs explicit integer N_l >= 2")
        if len(c["N"]) > 13:
            raise ValueError("For safety this driver limits benchmark horizons to L<=12")
        if cfg["solver"] == "dense" and 2 ** (len(c["N"]) + 2) > DENSE_MAX_ELEMENTS:
            raise ValueError(
                f"Configuration {c['name']}: dense solver requires L<=6 "
                f"(<= {DENSE_MAX_ELEMENTS} elements). Use 'thomas' or 'banded'.")
        if "SMC" in cfg["methods"] and (type(c.get("smc_N")) is not int or c["smc_N"] < 2):
            raise ValueError("When SMC is enabled, each configuration needs smc_N >= 2")
    if len(set(cfg["methods"])) != len(cfg["methods"]) or any(m not in LABELS for m in cfg["methods"]):
        raise ValueError(f"methods can contain each of {list(LABELS)} once")
    if not cfg["figure5_configuration"]:
        cfg["figure5_configuration"] = max(cfg["configurations"], key=lambda c: len(c["N"]))["name"]
    if cfg["figure5_configuration"] not in seen:
        raise ValueError("figure5_configuration must name an existing configuration")
    p = cfg["light"]
    for key in ("iterations", "batch_size", "K"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"light.{key} must be a positive integer")
    if min(p["epsilon"], p["lr"], p["S_init"]) <= 0 or p["dtype"] not in ("float32", "float64"):
        raise ValueError("Invalid LightSB settings")
    importance_options(p)
    whitening_options(p)
    epsilon_options(p)
    if p.get("lr_schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("light.lr_schedule must be 'constant' or 'cosine'")
    schedule = p.get("iterations_by_level")
    if schedule is not None:
        if (not isinstance(schedule, list) or not schedule
                or any(type(n) is not int or n < 1 for n in schedule)):
            raise ValueError("light.iterations_by_level must be a non-empty list of positive integers or null")
    max_transitions = max(len(c["N"]) - 1 for c in cfg["configurations"])
    for level in range(1, max_transitions + 1):
        if not epsilon_for_level(p, level) > 0:
            raise ValueError(f"light: epsilon for level {level} must be positive (set epsilon_min > 0)")
    validate_transport_options(cfg)
    if cfg.get("initial_cache") is not None and not isinstance(cfg["initial_cache"], str):
        raise ValueError("initial_cache must be null or a folder name")
    if cfg["problem"] not in ("beskos", "source_moments", "darcy2d", "ns2d"):
        raise ValueError("problem must be 'beskos', 'source_moments', 'darcy2d' or 'ns2d'")
    if cfg["problem"] == "ns2d":
        unknown_n = set(cfg.get("ns", {})) - set(ns2d.NS_DEFAULTS)
        if unknown_n:
            raise ValueError(f"unknown ns settings {sorted(unknown_n)}")
        cfg["dimension"] = int(cfg.get("ns", {}).get("d", ns2d.NS_DEFAULTS["d"]))
        cfg["noise_sd"] = float(cfg.get("ns", {}).get("noise_sd", ns2d.NS_DEFAULTS["noise_sd"]))
    if cfg["problem"] == "darcy2d":
        unknown_d = set(cfg.get("darcy", {})) - set(darcy2d.DARCY_DEFAULTS)
        if unknown_d:
            raise ValueError(f"unknown darcy settings {sorted(unknown_d)}")
        cfg["dimension"] = int(cfg.get("darcy", {}).get("d", darcy2d.DARCY_DEFAULTS["d"]))
        cfg["noise_sd"] = float(cfg.get("darcy", {}).get("noise_sd", darcy2d.DARCY_DEFAULTS["noise_sd"]))
    unknown = set(cfg["source"]) - set(SOURCE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown source settings {sorted(unknown)}")
    if cfg["problem"] == "source_moments":
        cfg["dimension"] = 51                       # 1 source position + 50 field coefficients
    if isinstance(cfg["noise_sd"], bool) or not isinstance(cfg["noise_sd"], (int, float)) or not cfg["noise_sd"] > 0:
        raise ValueError("noise_sd must be a positive number")
    if cfg["rwm"]["block_size"] < 1 or cfg["rwm"]["sweeps"] < 1 or cfg["rwm"]["scale"] <= 0:
        raise ValueError("Invalid blocked RWM settings")
    if any(type(l) is not int or not 1 <= l <= 13 for l in cfg["figure4"]["levels"]):
        raise ValueError("Figure4 levels must be integers in [1,13]")
    return cfg


def experiment_signature(cfg):
    spec = copy.deepcopy(cfg)
    # Increasing repeats and changing bootstrap display details is safe on resume.
    spec.pop("repeats", None)
    spec.pop("bootstrap_draws", None)
    names = (Path(__file__).name, "beskos_elliptic.py", "darcy2d.py") + (("ns2d.py",) if cfg.get("problem") == "ns2d" else ())
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in names}
    return canonical_hash({"schema": SCHEMA, "config": spec, "code": code_hashes})


def initial_pool(problem, n, seed, max_proposals, cache_dir=None):
    """Level-0 pool (untimed). With cache_dir, pools are stored on disk keyed by the target
    (problem, data, observable), n and seed, so studies of the same target with the same
    run_seed reuse identical pools instead of redrawing them (e.g. ~350 s per L9 repeat for
    T1 at noise 0.05). The pool is a deterministic function of the key, so cached and fresh
    pools are identical; only wall time outside the benchmark timers changes."""
    if not cache_dir:
        z, info = problem.sample_posterior(n, level=0, seed=seed, dtype=torch.float64,
                                           max_proposals=max_proposals)
        return z, info
    key = canonical_hash({"target": target_description(problem), "n": int(n), "seed": int(seed),
                          "max_proposals": int(max_proposals)})[:32]
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"pool_{key}.pt"
    if path.exists():
        data = torch.load(path)
        return data["z"].to(torch.float64), dict(json.loads(data["info"]), from_cache=True)
    z, info = problem.sample_posterior(n, level=0, seed=seed, dtype=torch.float64, max_proposals=max_proposals)
    tmp = path.with_suffix(".tmp")
    torch.save({"z": z.detach().clone(), "info": json.dumps(info)}, tmp)
    os.replace(tmp, path)
    return z, dict(info, from_cache=False)


def run_experiments(cfg, out, resume=False, reference_file=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    signature = experiment_signature(cfg)
    meta_path = out / "metadata.json"
    previous = None
    if meta_path.exists():
        if not resume:
            raise FileExistsError("Output already contains a study. Use --resume or choose a new --out.")
        previous = json.loads(meta_path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Config or code changed. Use a NEW output directory; do not mix experiments.")
    # Reuse the saved exact observations on resume.
    y = previous["target"]["y"] if previous else cfg.get("y")
    problem = make_problem(cfg, y)
    meta = {"schema": SCHEMA, "signature": signature, "config": cfg,
            "target": target_description(problem),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                            "device": "cpu", "threads": cfg["threads"]},
            "timing": "Warm-cache conditional-on-initial-population benchmark; independent eta0 rejection initialization, reference and Figure4 excluded. Sampling + fixed-phi estimator evaluation included.",
            "timing_profile": {
                "version": TIMING_PROFILE_VERSION,
                "pde_seconds": "Complete log-density calls: coordinate transforms, FEM assembly/solve, likelihood; not pure linear-solver time. Excludes observable evaluations.",
                "pde_reweight_seconds": "Initial SMC log density plus densities used for reweighting or LightSB cached ratios.",
                "pde_mcmc_seconds": "Proposal log-density calls inside MCMC, including rejected proposals.",
                "fit_seconds": "LightSB model and optimizer initialization plus all optimizer updates; no PDE calls.",
                "propagation_seconds": "LightSB uniform parent selection and conditional child sampling.",
                "resampling_seconds": "SMC/MLSMC weighted ancestor selection and population/log-density copies.",
                "mcmc_total_seconds": "Inclusive mutation time: OVERLAPS pde_mcmc_seconds. Do not add to the partition.",
                "mcmc_other_seconds": "mcmc_total_seconds minus pde_mcmc_seconds.",
                "sampler_other_seconds": "Remaining sampler wall time, including unclassified checks/copies and timer overhead.",
                "partition": "sampler_seconds = pde_seconds + fit_seconds + propagation_seconds + resampling_seconds + mcmc_other_seconds + sampler_other_seconds (up to timing roundoff).",
                "per_level": "diagnostics[].timing_breakdown covers each transition. Initial density/observable work precedes those transitions and is in run totals only.",
                "overhead": "Profiling has overhead; no timing calls are added inside the Adam update loop. Console/file output is outside recorded timers. CPU-only.",
            },
            "pde_work": "Actual FEM forward counts weighted by number of elements, in common units across L. Continuum observable/reference work logged separately.",
            "light_importance": {
                **importance_options(cfg["light"]),
                "interpretation": "Optional fixed-cloud optimizer warm-up ending at beta=1; NOT SMC tempering. "
                                  "Global minibatches target the full-cloud empirical measure, not a guaranteed posterior. "
                                  "One transport per PDE level; final raw ESS always retained."
            },
            "light_whitening": {
                **whitening_options(cfg["light"]),
                **epsilon_options(cfg["light"]),
                "epsilon_by_transition": [epsilon_for_level(cfg["light"], l) for l in range(1, max(len(c["N"]) for c in cfg["configurations"]))],
                "interpretation": "Per-level affine whitening from the unweighted source cloud; "
                                  "LightSB (epsilon, S, r) operates in whitened units; children mapped back to z. "
                                  "init='identity': S=identity_scale(eps), r=(1-S)*anchor. "
                                  "correction='is': children weighted by gamma_l/q_l with q_l the exact "
                                  "LightSB mixture marginal, paired self-normalised correction, then multinomial "
                                  "resampling; extra level-l solves for every child are counted in FEM work."
            },
            "estimator": "Same continuum phi at all levels. LightSB unweighted pairs; MLSMC equation (5); SMC final mean."}
    atomic_json(meta_path, meta)
    print("FEM benchmark solver:", cfg["solver"], flush=True)
    if cfg["solver"] == "dense":
        warnings.warn("Dense FEM is a verification/sensitivity backend. FEM element-work "
                      "counts do not model its cubic dense-solve cost; report wall time separately.")
    print("Fixed observations:", problem.y.tolist(), flush=True)
    compute_reference(problem, cfg["reference"], out, reference_file)
    print("Computing shared PDE-convergence diagnostic...", flush=True)
    figure4(problem, cfg["figure4"], out)
    max_L = max(len(c["N"]) - 1 for c in cfg["configurations"])
    # Warm mesh/basis/quadrature caches for ALL methods before any method timer.
    zero = torch.zeros(1, problem.dimension, dtype=torch.float64)
    for level in range(max_L + 1):
        problem.fem_values(zero, level)
    problem.phi_u(zero)
    # Warm Adam and small matrix operations separately (not a production fit).
    warm = torch.randn(8, problem.dimension)
    settings = dict(cfg["light"], iterations=2, K=min(cfg["light"]["K"], 4), batch_size=8)
    fit_light_sb(warm, torch.zeros(8, dtype=torch.float64), settings)
    for ci, c in enumerate(cfg["configurations"]):
        N = c["N"]
        for r in range(cfg["repeats"]):
            paths = {m: out / "records" / f"{c['name']}_{r:05d}_{m}.json" for m in cfg["methods"]}
            missing = [m for m in cfg["methods"] if not paths[m].exists()]
            if not missing:
                continue
            initial_seed = cfg["run_seed"] + 100_000 * ci + 100 * r
            n_initial = max([N[0]] + ([c["smc_N"]] if "SMC" in cfg["methods"] else []))
            t0 = time.perf_counter()
            initial, init_info = initial_pool(problem, n_initial, initial_seed, cfg["initial_max_proposals"],
                                              cfg.get("initial_cache"))
            init_seconds = time.perf_counter() - t0
            # Vary execution order so one method does not always run first.
            order = list(np.random.default_rng(initial_seed + 90).permutation(missing))
            for method in order:
                print(f"{c['name']} repeat {r+1}/{cfg['repeats']}: {method}", flush=True)
                seed = initial_seed + METHOD_CODE[method]
                try:
                    if method == "LightSB":
                        answer = run_light_sb(problem, initial, N, cfg["light"], seed)
                    elif method == "MLSMC":
                        answer = run_smc(problem, initial, N, cfg["rwm"], seed, True)
                    elif method == "TransportMH":
                        answer = run_transport_mh(problem, initial, N, cfg["light"], cfg["transport_mh"], seed)
                    elif method == "MLSMC_pCN":
                        answer = run_mlsmc_kernel(problem, initial, N, cfg["light"], cfg["mlsmc_kernel"], "pcn", seed)
                    elif method == "MLSMC_LSB":
                        answer = run_mlsmc_kernel(problem, initial, N, cfg["light"], cfg["mlsmc_kernel"], "lsb", seed)
                    else:
                        answer = run_smc(problem, initial, [c["smc_N"]] * len(N), cfg["rwm"], seed, False)
                    record = {"schema": SCHEMA, "signature": signature, "configuration": c["name"],
                              "method": method, "repeat": r, "seed": seed, "L": len(N)-1,
                              "N": N if method != "SMC" else [c["smc_N"]] * len(N),
                              "common_initial_pool_size": n_initial,
                              "common_initialization_seconds_excluded": init_seconds,
                              "initialization": init_info, **answer}
                    atomic_json(paths[method], record)
                    print(f"  estimate={record['estimate']:.8f}, total={record['total_seconds']:.3f}s, "
                          f"FEM work={record['fem_element_work']}", flush=True)
                    print_timing_breakdown(record)
                    if method in ("LightSB", "TransportMH"):
                        print_importance_diagnostics(record)
                    print_mh_diagnostics(record)
                except Exception as error:
                    atomic_json(out / "last_failure.json", {"configuration": c["name"], "repeat": r,
                                "method": method, "seed": seed, "error": repr(error)})
                    raise  # Do not silently discard failed runs and report survivor-only MSE.
    if json.loads((out / "reference.json").read_text())["value"] is None:
        print("No reference value (reference.mode='none'): skipping error summaries and plots.", flush=True)
        return
    plot_results(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON with preallocated N for every configuration")
    parser.add_argument("--out", type=Path, default=Path("beskos_figures_results"))
    parser.add_argument("--include-smc", action="store_true")
    parser.add_argument("--repeats", type=int, help="Override repetitions; >=2")
    parser.add_argument("--light-iterations", type=int)
    parser.add_argument("--reference-file", type=Path, help="Reuse a reference.json for exactly the same target")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Compute a reference in --out without running the experiment")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from saved runs; no simulation")
    args = parser.parse_args()
    if args.plot_only:
        if args.reference_file:
            new_ref = json.loads(args.reference_file.read_text())
            old_ref = json.loads((args.out / "reference.json").read_text())
            if new_ref.get("target_hash") != old_ref.get("target_hash"):
                raise ValueError("New reference belongs to a different target")
            atomic_json(args.out / "reference.json", new_ref)
        plot_results(args.out)
        return
    if args.config is None:
        parser.error("Provide --config (e.g. config_smoke.json)")
    raw = json.loads(args.config.read_text())
    if args.include_smc:
        raw["methods"] = list(dict.fromkeys(raw.get("methods", DEFAULTS["methods"]) + ["SMC"]))
    if args.repeats is not None:
        raw["repeats"] = args.repeats
    if args.light_iterations is not None:
        raw.setdefault("light", {})["iterations"] = args.light_iterations
    cfg = merge_config(raw)
    torch.set_num_threads(cfg["threads"])
    # Limits BLAS threads too; torch.set_num_threads alone does not limit SciPy.
    with threadpool_limits(limits=cfg["threads"]):
        if args.reference_only:
            args.out.mkdir(parents=True, exist_ok=True)
            problem = make_problem(cfg, cfg.get("y"))
            compute_reference(problem, cfg["reference"], args.out, args.reference_file)
        else:
            run_experiments(cfg, args.out, args.resume, args.reference_file)


if __name__ == "__main__":
    main()
