"""2D Darcy flow inverse problem (NumPy/SciPy core, no torch).

    -div( kappa(x) grad p ) = 0  on D = (0,1)^2,
    p = 1 on x1 = 0,  p = 0 on x1 = 1,  zero flux on x2 = 0 and x2 = 1.

log kappa(x) = sum_k z_k sqrt(lambda_k) psi_k(x),  z ~ N(0, I_d)  (truncated KL, cosine basis,
Matern-like spectrum lambda_ij = sigma^2 (pi^2 (i^2 + j^2) + tau^2)^(-alpha), the d largest kept).

Level l: uniform n x n grid, n = n0 * 2^l, each square cut into two triangles (P1 FEM),
kappa constant per triangle = exp(log kappa at the triangle centroid).
Data: pressure at an m x m sensor grid (i/(m+1), j/(m+1)), read by P1 interpolation
(sensors are generally not mesh nodes, so coarse meshes see a smeared field).
Observable (the same at every level, no PDE): phi(z) = mean of kappa over the central square
[1/4, 3/4]^2 (effective permeability of the centre), midpoint rule on a fixed 64 x 64 grid.

Solvers: batched dense Cholesky for small systems, sparse LU (SuperLU, MMD ordering) per sample
otherwise. Stiffness assembly is linear in kappa: data = B @ kappa with a precomputed sparse B.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DARCY_DEFAULTS = {"d": 25, "n0": 4, "sensors": 5, "noise_sd": 0.025, "sigma": 1.0, "tau": 3.0,
                  "alpha": 2.0, "data_level": 6, "dense_max": 100, "phi_grid": 64}


def kl_modes(d, sigma, tau, alpha):
    """(i, j, sqrt(lambda), normalisation) for the d largest cosine modes."""
    # (0, 0) is left out: a constant shift of log kappa leaves the pressure unchanged
    # (p is invariant under kappa -> c kappa), so that mode would be prior-only.
    cand = [(i, j) for i in range(40) for j in range(40) if (i, j) != (0, 0)]
    lam = np.array([sigma ** 2 * (math.pi ** 2 * (i * i + j * j) + tau ** 2) ** (-alpha) for i, j in cand])
    order = np.argsort(-lam, kind="stable")[:d]
    ij = np.array([cand[k] for k in order], dtype=float)
    # L2-normalised cosines: c_0 = 1, c_k = sqrt(2)
    norm = np.where(ij[:, 0] > 0, math.sqrt(2), 1.0) * np.where(ij[:, 1] > 0, math.sqrt(2), 1.0)
    # rescale so that the pointwise prior variance of log kappa averages sigma^2 over D
    s = np.sqrt(lam[order])
    s = s * sigma / math.sqrt(np.sum(s ** 2))
    return ij, s, norm


class DarcyLevel:
    """Mesh, assembly maps and sensor interpolation for one level."""

    def __init__(self, n, modes, sensors_xy, dense_max):
        self.n = n
        ij, s, norm = modes
        h = 1.0 / n
        # nodes (a, b) -> a + (n+1) * b, x1 = a h, x2 = b h
        a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        a, b = a.ravel(), b.ravel()
        v00 = a + (n + 1) * b
        v10 = v00 + 1
        v01 = v00 + (n + 1)
        v11 = v01 + 1
        # two triangles per square: (v00, v10, v11) and (v00, v11, v01)
        tri = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
        self.n_elements = len(tri)
        nodes_x = (np.arange((n + 1) ** 2) % (n + 1)) * h
        nodes_y = (np.arange((n + 1) ** 2) // (n + 1)) * h
        P = np.stack([nodes_x[tri], nodes_y[tri]], -1)                  # [E, 3, 2]
        cent = P.mean(1)
        # local P1 stiffness for each element (geometry only; area h^2/2)
        e1 = P[:, 2] - P[:, 1]; e2 = P[:, 0] - P[:, 2]; e3 = P[:, 1] - P[:, 0]
        E = np.stack([e1, e2, e3], 1)                                    # edge opposite vertex k
        area = 0.5 * h * h
        Kloc = np.einsum("eki,eli->ekl", E, E) / (4.0 * area)            # [E, 3, 3]
        # field basis at centroids: [E, d]
        self.field_basis = (norm * s)[None] * np.cos(math.pi * ij[None, :, 0] * cent[:, 0:1]) \
            * np.cos(math.pi * ij[None, :, 1] * cent[:, 1:2])
        # boundary: left x=0 (p=1), right x=1 (p=0); free = the rest
        nn = (n + 1) ** 2
        left = nodes_x < 0.5 * h
        right = nodes_x > 1 - 0.5 * h
        free = ~(left | right)
        fidx = -np.ones(nn, dtype=np.int64)
        fidx[free] = np.arange(free.sum())
        self.n_free = int(free.sum())
        rows = np.repeat(tri, 3, axis=1).ravel()
        cols = np.tile(tri, (1, 3)).ravel()
        vals = Kloc.reshape(len(tri), 9).ravel()
        elem = np.repeat(np.arange(len(tri)), 9)
        # free-free block: CSR pattern + map from kappa
        m = free[rows] & free[cols]
        fr, fc, fv, fe = fidx[rows[m]], fidx[cols[m]], vals[m], elem[m]
        key = fr * self.n_free + fc
        ukey, inv = np.unique(key, return_inverse=True)
        self.B = sp.csr_matrix((fv, (inv, fe)), shape=(len(ukey), len(tri)))
        self.pat_rows = (ukey // self.n_free).astype(np.int64)
        self.pat_cols = (ukey % self.n_free).astype(np.int64)
        tmp = sp.csr_matrix((np.arange(1, len(ukey) + 1, dtype=float), (self.pat_rows, self.pat_cols)),
                            shape=(self.n_free, self.n_free))
        tmp.sort_indices()
        self.csr_indptr, self.csr_indices = tmp.indptr.copy(), tmp.indices.copy()
        self.csr_perm = (tmp.data.astype(np.int64) - 1)                  # csr slot -> pattern entry
        # rhs = - sum over left-boundary columns (p = 1 there)
        m2 = free[rows] & left[cols]
        self.Brhs = sp.csr_matrix((-vals[m2], (fidx[rows[m2]], elem[m2])), shape=(self.n_free, len(tri)))
        # sensors: P1 interpolation from FULL nodal vector; p_full = offset + S_free p_free
        S = []
        off = np.zeros(len(sensors_xy))
        for k, (x, y) in enumerate(sensors_xy):
            i = min(int(x / h), n - 1); j = min(int(y / h), n - 1)
            s_, t_ = x / h - i, y / h - j
            n00 = i + (n + 1) * j; n10 = n00 + 1; n01 = n00 + (n + 1); n11 = n01 + 1
            if t_ <= s_:      # lower triangle (v00, v10, v11)
                w = {n00: 1 - s_, n10: s_ - t_, n11: t_}
            else:             # upper triangle (v00, v11, v01)
                w = {n00: 1 - t_, n11: s_, n01: t_ - s_}
            row = {}
            for node, wt in w.items():
                if free[node]:
                    row[fidx[node]] = row.get(fidx[node], 0.0) + wt
                elif left[node]:
                    off[k] += wt
            S.append(row)
        Sm = sp.lil_matrix((len(sensors_xy), self.n_free))
        for k, row in enumerate(S):
            for c, wt in row.items():
                Sm[k, c] = wt
        self.S = Sm.tocsr()
        self.S_offset = off
        self.dense = self.n_free <= dense_max
        self.free, self.left, self.fidx = free, left, fidx
        self.nodes_x, self.nodes_y = nodes_x, nodes_y
        if self.dense:
            self.dense_flat = self.pat_rows * self.n_free + self.pat_cols

    def kappa(self, Z):
        return np.exp(Z @ self.field_basis.T)                             # [B, E]

    def solve_free(self, Z):
        """Free-node pressures [B, n_free] for parameters Z [B, d]."""
        K = self.kappa(Z)
        data = (self.B @ K.T).T                                           # [B, nnz]
        rhs = (self.Brhs @ K.T).T                                         # [B, n_free]
        if self.dense:
            A = np.zeros((len(Z), self.n_free * self.n_free))
            A[:, self.dense_flat] = data
            A = A.reshape(len(Z), self.n_free, self.n_free)
            c = np.linalg.cholesky(A)
            y = np.linalg.solve(c, rhs[..., None])
            return np.linalg.solve(np.swapaxes(c, -1, -2), y)[..., 0]
        out = np.empty((len(Z), self.n_free))
        for b in range(len(Z)):
            A = sp.csc_matrix((data[b, self.csr_perm], self.csr_indices, self.csr_indptr),
                              shape=(self.n_free, self.n_free))   # symmetric: CSR of A == CSC of A^T = A
            out[b] = spla.splu(A, permc_spec="MMD_AT_PLUS_A").solve(rhs[b])
        return out

    def sensor_values(self, Z):
        pf = self.solve_free(Z)
        return (self.S @ pf.T).T + self.S_offset

    def full_nodes(self, z):
        pf = self.solve_free(z[None])[0]
        p = np.zeros(len(self.free))
        p[self.left] = 1.0
        p[self.free] = pf
        return p


class DarcyCore:
    def __init__(self, d=25, n0=4, sensors=5, noise_sd=0.025, sigma=1.0, tau=3.0, alpha=2.0,
                 data_level=6, dense_max=100, phi_grid=64, seed=2026, y=None, max_level=7):
        self.d, self.n0, self.noise_sd = int(d), int(n0), float(noise_sd)
        self.settings = dict(d=int(d), n0=int(n0), sensors=int(sensors), noise_sd=float(noise_sd),
                             sigma=float(sigma), tau=float(tau), alpha=float(alpha),
                             data_level=data_level, dense_max=int(dense_max), phi_grid=int(phi_grid))
        self.modes = kl_modes(self.d, sigma, tau, alpha)
        g = np.arange(1, sensors + 1) / (sensors + 1)
        self.sensors_xy = [(x, y_) for y_ in g for x in g]
        self.dense_max = int(dense_max)
        self._levels = {}
        ij, s, norm = self.modes
        t = (np.arange(phi_grid) + 0.5) / phi_grid
        qx, qy = np.meshgrid(0.25 + 0.5 * t, 0.25 + 0.5 * t, indexing="ij")
        qx, qy = qx.ravel(), qy.ravel()
        self.phi_basis = (norm * s)[None] * np.cos(math.pi * ij[None, :, 0] * qx[:, None]) \
            * np.cos(math.pi * ij[None, :, 1] * qy[:, None])
        if y is None:
            rng = np.random.default_rng(seed)
            self.z_true = rng.standard_normal(self.d)
            dl = data_level if data_level is not None else max_level
            self.y_clean = self.level(dl).sensor_values(self.z_true[None])[0]
            self.y = self.y_clean + self.noise_sd * rng.standard_normal(len(self.y_clean))
            self.data_level = dl
        else:
            self.y = np.asarray(y, dtype=float)
            self.z_true = self.y_clean = None
            self.data_level = data_level

    def mesh_n(self, level):
        return self.n0 * 2 ** int(level)

    def elements(self, level):
        return 2 * self.mesh_n(level) ** 2

    def level(self, level):
        if level not in self._levels:
            self._levels[level] = DarcyLevel(self.mesh_n(level), self.modes, self.sensors_xy, self.dense_max)
        return self._levels[level]

    def loglik(self, Z, level):
        r = (self.level(level).sensor_values(Z) - self.y) / self.noise_sd
        return -0.5 * np.sum(r * r, -1)

    def log_density(self, Z, level):
        return -0.5 * np.sum(Z * Z, -1) + self.loglik(Z, level)

    def phi(self, Z):
        return np.exp(Z @ self.phi_basis.T).mean(-1)


def tempered_level0_pool(core, n, seed, sweeps=5, ess_target=0.5, rho0=0.5, max_stages=200):
    """Approximate iid draws from pi_0 (level-0 posterior) by adaptive likelihood tempering:
    prior -> pi_0 with ESS-chosen increments, multinomial resampling, then `sweeps` whitened
    pCN-MH moves per stage (Gaussian reference fitted to the cloud; exact MH ratio, so each move
    leaves the tempered target invariant). Level-0 solves are tiny (15 unknowns), so this is
    cheap; it replaces prior-rejection sampling, whose acceptance is ~0 for this likelihood.
    Returns (Z [n, d], diagnostics)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, core.d))
    ll = core.loglik(Z, 0)
    beta, stages, rho, accs = 0.0, 0, float(rho0), []
    solves = n

    def ess_frac(lw):
        w = np.exp(lw - lw.max()); w /= w.sum()
        return 1.0 / np.sum(w * w) / len(w)

    while beta < 1.0:
        if stages >= max_stages:
            raise RuntimeError("tempered_level0_pool: too many stages")
        hi = 1.0 - beta
        if ess_frac(hi * ll) >= ess_target:
            db = hi
        else:
            lo = 0.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess_frac(mid * ll) >= ess_target else (lo, mid)
            db = max(lo, 1e-12)
        w = np.exp(db * ll - (db * ll).max()); w /= w.sum()
        idx = rng.choice(n, n, p=w)
        Z, ll = Z[idx], ll[idx]
        beta = 1.0 if db >= 1.0 - beta - 1e-15 else beta + db
        m = Z.mean(0)
        C = np.cov(Z.T) + 1e-8 * np.eye(core.d)
        Lc = np.linalg.cholesky(C)
        T = np.linalg.solve(Lc, (Z - m).T).T
        logp = -0.5 * np.sum(Z * Z, 1) + beta * ll
        acc = 0.0
        for _ in range(sweeps):
            Tp = rho * T + math.sqrt(1.0 - rho * rho) * rng.standard_normal(T.shape)
            Zp = m + Tp @ Lc.T
            llp = core.loglik(Zp, 0)
            solves += n
            lpp = -0.5 * np.sum(Zp * Zp, 1) + beta * llp
            la = (lpp + 0.5 * np.sum(Tp * Tp, 1)) - (logp + 0.5 * np.sum(T * T, 1))
            a = np.log(rng.random(n)) < la
            T[a], logp[a], ll[a] = Tp[a], lpp[a], llp[a]
            acc += a.mean()
        acc /= sweeps
        accs.append(float(acc))
        rho = float(min(0.98, max(0.0, rho + 0.3 * (0.3 - acc))))   # aim at ~30% acceptance
        Z = m + T @ Lc.T
        stages += 1
    return Z, {"method": "tempered SMC + whitened pCN at level 0", "stages": stages,
               "sweeps_per_stage": sweeps, "final_acceptance": accs[-1], "level0_solves": int(solves),
               "unique_after_last_resampling": int(len(np.unique(idx)))}


def pde_convergence(core, levels, seed):
    """Sensor-value differences between consecutive levels at one prior draw (Figure 4 analogue)."""
    z = np.random.default_rng(seed).standard_normal(core.d)
    rows = []
    for level in levels:
        a = core.level(level - 1).sensor_values(z[None])[0]
        b = core.level(level).sensor_values(z[None])[0]
        diff = b - a
        rows.append({"level": int(level), "h": 1.0 / core.mesh_n(level),
                     "sensor_diff_max": float(np.abs(diff).max()),
                     "sensor_diff_rms": float(np.sqrt(np.mean(diff * diff))),
                     "H01_seminorm_squared": float(np.mean(diff * diff))})
    return {"z": z.tolist(), "rows": rows, "primary_norm": "mean squared sensor difference"}


if __name__ == "__main__":
    import time
    core = DarcyCore()
    exact = np.array([1.0 - x for x, _ in core.sensors_xy])
    print("constant-kappa error (p = 1 - x):",
          max(np.abs(core.level(l).sensor_values(np.zeros((1, core.d)))[0] - exact).max() for l in range(4)))
    rows = pde_convergence(core, [1, 2, 3, 4, 5], 745)["rows"]
    for r in rows:
        print(f"level {r['level']}: max sensor change {r['sensor_diff_max']:.2e}")
    rng = np.random.default_rng(0)
    for l in range(6):
        L = core.level(l)
        B = {0: 20000, 1: 5000, 2: 1000, 3: 200, 4: 50, 5: 10}[l]
        Z = rng.standard_normal((B, core.d))
        L.sensor_values(Z[:2])
        t = time.perf_counter(); L.sensor_values(Z); dt = (time.perf_counter() - t) / B
        print(f"level {l}: {L.n_elements} elements, {'dense batched' if L.dense else 'sparse LU'}, {dt * 1e3:.3f} ms per solve")
    t = time.perf_counter()
    Z, info = tempered_level0_pool(core, 20000, seed=1)
    lr = core.loglik(Z, 1) - core.loglik(Z, 0)
    w = np.exp(lr - lr.max()); w /= w.sum()
    print(f"level-0 pool of 20000: {time.perf_counter() - t:.1f} s, {info['stages']} tempering stages; "
          f"ESS fraction of the level 0 -> 1 weights: {1.0 / np.sum(w * w) / len(w):.4f}")
