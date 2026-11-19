"""2D incompressible Navier--Stokes inverse problem on the unit torus (NumPy core, no torch).

Vorticity form on T^2 = [0,1)^2:
    dw/dt + u . grad w = nu Lap w + f,   u = (d psi/dy, -d psi/dx),   -Lap psi = w,
with a fixed forcing f(x) = f_amp (sin 2pi(x1+x2) + cos 2pi(x1+x2)).

Unknown: the initial vorticity w_0(x) = sum_k z_k sqrt(lambda_k) e_k(x), z ~ N(0, I_d), with
e_k the L2-normalised real Fourier modes sqrt(2) cos(2pi k.x), sqrt(2) sin(2pi k.x) over the
d/2 wavevectors of smallest |k| (half-plane, k != 0) and lambda_k ~ (4 pi^2 |k|^2 + tau^2)^(-alpha),
rescaled so that the average pointwise prior variance of w_0 is sigma^2.

Level l: N_l x N_l grid, N_l = n0 2^l. scheme="fd2" (default): second-order central finite
differences for all derivatives and the Laplacian (implemented with FFTs through the modified
wavenumbers sin(k h)/h and (2 sin(k h/2)/h)^2), so the spatial error is O(h^2);
scheme="spectral": pseudo-spectral with 2/3 dealiasing (exponential convergence, so the level
corrections vanish after the grid resolves the flow). Integrating-factor RK4 in time with
n_steps_l = steps0 2^l steps on [0, T] (time step proportional to h_l).
Cost per solve ~ N_l^2 log N_l * n_steps_l ~ h_l^(-3).
Data: vorticity at an m x m grid of points (i/m, j/m) (grid points of every level) at the
observation times, plus N(0, noise_sd^2) noise; generated on level data_level.
Observable (no PDE, the same at every level, linear in z hence Lipschitz):
phi(z) = mean of w_0 over the square [0, 1/2]^2 (midpoint rule on a fixed grid).
"""
from __future__ import annotations

import math
import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq

NS_DEFAULTS = {"d": 36, "n0": 8, "steps0": 8, "T": 1.0, "obs_times": (0.5, 1.0), "sensors": 4,
               "nu": 1e-2, "sigma": 2.0, "tau": 3.0, "alpha": 2.0, "f_amp": 0.5,
               "noise_sd": 0.25, "data_level": 5, "phi_grid": 64, "batch": 64, "scheme": "fd2"}


def fourier_modes(d, sigma, tau, alpha):
    """(wavevectors [d/2, 2], sqrt(lambda) per real basis function [d], kind [d] 0=cos 1=sin)."""
    if d % 2:
        raise ValueError("d must be even (cos and sin per wavevector)")
    cand = [(a, b) for a in range(0, 12) for b in range(-11, 12)
            if (a > 0 or b > 0) and (a, b) != (0, 0)]
    cand.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, -k[0], -k[1]))     # stable, deterministic ties
    K = np.array(cand[: d // 2], dtype=float)
    lam = (4 * math.pi ** 2 * (K ** 2).sum(1) + tau ** 2) ** (-alpha)
    s = np.repeat(np.sqrt(lam), 2)
    s = s * sigma / math.sqrt(np.sum(s ** 2))          # average pointwise variance sigma^2
    kind = np.tile([0, 1], d // 2)
    return np.repeat(K, 2, axis=0), s, kind


def basis_on(points, modes):
    """Real Fourier basis * sqrt(lambda) at points [P, 2] -> [P, d]."""
    K, s, kind = modes
    arg = 2 * math.pi * points @ K.T                    # [P, d]
    return math.sqrt(2.0) * s[None] * np.where(kind[None] == 0, np.cos(arg), np.sin(arg))


class NSLevel:
    """Grid, wavenumbers and integrator for one level."""

    def __init__(self, N, n_steps, T, obs_times, nu, f_amp, modes, sensors, scheme="fd2"):
        self.N, self.n_steps, self.T = int(N), int(n_steps), float(T)
        self.dt = self.T / self.n_steps
        obs_steps = [round(t / self.dt) for t in obs_times]
        if any(abs(o * self.dt - t) > 1e-9 for o, t in zip(obs_steps, obs_times)):
            raise ValueError(f"observation times must be multiples of dt on level N={N}")
        self.obs_steps = obs_steps
        x = np.arange(N) / N
        X1, X2 = np.meshgrid(x, x, indexing="ij")
        self.ic_basis = basis_on(np.stack([X1.ravel(), X2.ravel()], 1), modes)   # [N*N, d]
        k1 = 2 * math.pi * fftfreq(N, 1.0 / N)
        k2 = 2 * math.pi * rfftfreq(N, 1.0 / N)
        K1, K2 = np.meshgrid(k1, k2, indexing="ij")
        h = 1.0 / N
        if scheme == "fd2":          # central differences: d/dx -> i sin(kh)/h, Lap -> -(2 sin(kh/2)/h)^2
            self.K1, self.K2 = np.sin(K1 * h) / h, np.sin(K2 * h) / h
            ksq = (2 * np.sin(K1 * h / 2) / h) ** 2 + (2 * np.sin(K2 * h / 2) / h) ** 2
        elif scheme == "spectral":
            self.K1, self.K2 = K1, K2
            ksq = K1 ** 2 + K2 ** 2
        else:
            raise ValueError("scheme must be 'fd2' or 'spectral'")
        self.inv_ksq = np.where(ksq > 0, 1.0 / np.where(ksq > 0, ksq, 1.0), 0.0)
        kmax = N / 3.0
        n1 = np.abs(fftfreq(N, 1.0 / N)); n2 = rfftfreq(N, 1.0 / N)
        N1, N2 = np.meshgrid(n1, n2, indexing="ij")
        self.dealias = ((N1 < kmax) & (N2 < kmax)).astype(float) if scheme == "spectral" else 1.0
        self.E = np.exp(-nu * ksq * self.dt)
        self.E2 = np.exp(-nu * ksq * self.dt / 2)
        f = f_amp * (np.sin(2 * math.pi * (X1 + X2)) + np.cos(2 * math.pi * (X1 + X2)))
        self.fh = rfft2(f)
        if N % sensors:
            raise ValueError("sensors must divide the grid size of every level")
        step = N // sensors
        self.sensor_idx = (np.arange(sensors) * step)

    def nonlinear(self, wh):
        """FFT of -(u . grad w), dealiased; wh [B, N, N//2+1]."""
        psih = wh * self.inv_ksq
        u = irfft2(1j * self.K2 * psih, s=(self.N, self.N))
        v = irfft2(-1j * self.K1 * psih, s=(self.N, self.N))
        wx = irfft2(1j * self.K1 * wh, s=(self.N, self.N))
        wy = irfft2(1j * self.K2 * wh, s=(self.N, self.N))
        return -rfft2(u * wx + v * wy) * self.dealias

    def step(self, wh):
        dt, E, E2, fh = self.dt, self.E, self.E2, self.fh
        k1 = dt * (self.nonlinear(wh) + fh)
        k2 = dt * (self.nonlinear(E2 * (wh + 0.5 * k1)) + fh)
        k3 = dt * (self.nonlinear(E2 * wh + 0.5 * k2) + fh)
        k4 = dt * (self.nonlinear(E * wh + E2 * k3) + fh)
        return E * wh + (E * k1 + 2 * E2 * (k2 + k3) + k4) / 6.0

    def initial(self, Z):
        return (Z @ self.ic_basis.T).reshape(len(Z), self.N, self.N)

    def sensor_values(self, Z, batch=64):
        """Vorticity at the sensors at the observation times, [B, n_obs * m * m]."""
        out = []
        for b0 in range(0, len(Z), batch):
            wh = rfft2(self.initial(Z[b0:b0 + batch]))
            obs = []
            for n in range(1, self.n_steps + 1):
                wh = self.step(wh)
                if n in self.obs_steps:
                    w = irfft2(wh, s=(self.N, self.N))
                    obs.append(w[:, self.sensor_idx][:, :, self.sensor_idx].reshape(len(w), -1))
            out.append(np.concatenate(obs, 1))
        return np.concatenate(out, 0) if out else np.empty((0, 0))

    def final_field(self, z):
        wh = rfft2(self.initial(z[None]))
        for _ in range(self.n_steps):
            wh = self.step(wh)
        return irfft2(wh, s=(self.N, self.N))[0]


class NSCore:
    """Same interface as darcy2d.DarcyCore: level(l).sensor_values, loglik, log_density, phi."""

    def __init__(self, d=36, n0=8, steps0=8, T=1.0, obs_times=(0.5, 1.0), sensors=4, nu=1e-2,
                 sigma=2.0, tau=3.0, alpha=2.0, f_amp=0.5, noise_sd=0.25, data_level=5, phi_grid=64,
                 batch=64, scheme="fd2", seed=2026, y=None):
        self.d, self.n0, self.steps0, self.noise_sd = int(d), int(n0), int(steps0), float(noise_sd)
        self.T, self.obs_times, self.sensors = float(T), tuple(float(t) for t in obs_times), int(sensors)
        self.nu, self.f_amp, self.batch, self.scheme = float(nu), float(f_amp), int(batch), str(scheme)
        self.settings = dict(d=self.d, n0=self.n0, steps0=self.steps0, T=self.T, obs_times=list(self.obs_times),
                             sensors=self.sensors, nu=self.nu, sigma=float(sigma), tau=float(tau),
                             alpha=float(alpha), f_amp=self.f_amp, noise_sd=self.noise_sd,
                             data_level=data_level, phi_grid=int(phi_grid), batch=self.batch,
                             scheme=self.scheme)
        self.modes = fourier_modes(self.d, sigma, tau, alpha)
        self._levels = {}
        t = (np.arange(phi_grid) + 0.5) / phi_grid
        q1, q2 = np.meshgrid(0.5 * t, 0.5 * t, indexing="ij")
        self.phi_vec = basis_on(np.stack([q1.ravel(), q2.ravel()], 1), self.modes).mean(0)   # [d]
        if y is None:
            rng = np.random.default_rng(seed)
            self.z_true = rng.standard_normal(self.d)
            self.y_clean = self.level(data_level).sensor_values(self.z_true[None])[0]
            self.y = self.y_clean + self.noise_sd * rng.standard_normal(len(self.y_clean))
        else:
            self.y = np.asarray(y, dtype=float)
            self.z_true = self.y_clean = None
        self.data_level = data_level

    def mesh_n(self, level):
        return self.n0 * 2 ** int(level)

    def n_steps(self, level):
        return self.steps0 * 2 ** int(level)

    def elements(self, level):
        """Work unit: grid points x time steps (FFT log factor ignored)."""
        return self.mesh_n(level) ** 2 * self.n_steps(level)

    def level(self, level):
        if level not in self._levels:
            self._levels[level] = NSLevel(self.mesh_n(level), self.n_steps(level), self.T, self.obs_times,
                                          self.nu, self.f_amp, self.modes, self.sensors, self.scheme)
        return self._levels[level]

    def sensor_values(self, Z, level):
        n = self.mesh_n(level)
        return self.level(level).sensor_values(Z, max(4, min(self.batch, 2 ** 16 // (n * n))))

    def loglik(self, Z, level):
        r = (self.sensor_values(Z, level) - self.y) / self.noise_sd
        return -0.5 * np.sum(r * r, -1)

    def log_density(self, Z, level):
        return -0.5 * np.sum(Z * Z, -1) + self.loglik(Z, level)

    def phi(self, Z):
        return Z @ self.phi_vec


def pde_convergence(core, levels, seed):
    """Sensor-value differences between consecutive levels at one prior draw (Figure 4 analogue)."""
    z = np.random.default_rng(seed).standard_normal(core.d)
    rows = []
    for level in levels:
        a = core.sensor_values(z[None], level - 1)[0]
        b = core.sensor_values(z[None], level)[0]
        diff = b - a
        rows.append({"level": int(level), "h": 1.0 / core.mesh_n(level),
                     "sensor_diff_max": float(np.abs(diff).max()),
                     "sensor_diff_rms": float(np.sqrt(np.mean(diff * diff))),
                     "H01_seminorm_squared": float(np.mean(diff * diff))})
    return {"z": z.tolist(), "rows": rows, "primary_norm": "mean squared sensor difference"}


if __name__ == "__main__":
    import time
    import darcy2d
    core = NSCore()
    print("settings:", core.settings)
    print("y range:", float(np.abs(core.y).max()), " obs:", len(core.y))
    rows = pde_convergence(core, [1, 2, 3, 4, 5], 745)["rows"]
    for r in rows:
        print(f"level {r['level']}: max sensor change {r['sensor_diff_max']:.2e}  rms {r['sensor_diff_rms']:.2e}")
    rng = np.random.default_rng(0)
    for l in range(6):
        B = {0: 2000, 1: 500, 2: 128, 3: 64, 4: 16, 5: 4}[l]
        Z = rng.standard_normal((B, core.d))
        core.sensor_values(Z[:2], l)
        t = time.perf_counter(); core.sensor_values(Z, l); dt = (time.perf_counter() - t) / B
        print(f"level {l}: N={core.mesh_n(l)}, steps={core.n_steps(l)}, {dt * 1e3:.2f} ms per solve")
    t = time.perf_counter()
    Z, info = darcy2d.tempered_level0_pool(core, 5000, seed=1)
    lr = core.loglik(Z, 1) - core.loglik(Z, 0)
    w = np.exp(lr - lr.max()); w /= w.sum()
    print(f"level-0 pool of 5000: {time.perf_counter() - t:.1f} s, {info['stages']} tempering stages; "
          f"ESS fraction of the level 0 -> 1 weights: {1.0 / np.sum(w * w) / len(w):.4f}")
    for l in (2, 3):
        lr = core.loglik(Z[:1000], l) - core.loglik(Z[:1000], l - 1)
        w = np.exp(lr - lr.max()); w /= w.sum()
        print(f"  (on pi_0 cloud) ESS of gamma_{l}/gamma_{l-1}: {1.0 / np.sum(w * w) / len(w):.3f}")
