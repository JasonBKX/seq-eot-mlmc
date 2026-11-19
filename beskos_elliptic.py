"""Elliptic inverse problem from Beskos et al., Multilevel SMC Samplers.

Sources:
  Accepted manuscript, Sections 4 and 5 (50 parameters):
  https://discovery.ucl.ac.uk/id/eprint/1465702/1/Beskos_BeskosJasraLawTemponeZhou_17.pdf
  Original preprint, Section 5 (2 parameters):
  https://arxiv.org/pdf/1503.07259v1

Original implementation, not the authors' experiment code or random dataset.
Dependencies: numpy, torch. All PDE calculations use float64, without autograd.

Physical variables u are in [-1,1]^dimension. For Gaussian LightSB kernels use
z = atanh(u), log_density_z, and phi_z. The Jacobian is included in the density.
"""
from __future__ import annotations

import math
from functools import partial
from numbers import Integral
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class BeskosElliptic:
    """Linear-FEM posterior hierarchy and midpoint-solution observable.

    dimension=50: accepted-manuscript parameterization.
    dimension=2: original-preprint parameterization.
    Other positive dimensions are truncations of the same coefficient series.

    level l uses 2**(l+3) elements; l=0 is already a posterior, not the prior.
    y is held fixed for every level. When y is omitted, a NEW seeded synthetic
    dataset is generated using h=2**(-20), then N(0, 0.25**2 I) noise is added.
    The original paper does not give its realized u_true/y or random seed in
    its experiment specification.
    """

    def __init__(self, dimension: int = 50, y=None, seed: int = 2026,
                 device="cpu", parameter_batch: int = 256,
                 element_chunk: int = 8192, reference_order: int = 64,
                 noise_sd: float = 0.25):
        if not isinstance(dimension, Integral) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if min(parameter_batch, element_chunk, reference_order) < 1:
            raise ValueError("Batch sizes and quadrature order must be positive")
        self.dimension = int(dimension)
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.parameter_batch = int(parameter_batch)
        self.element_chunk = int(element_chunk)
        self.reference_order = int(reference_order)
        # transport_mh copy: noise_sd is configurable (paper value 0.25).
        if not float(noise_sd) > 0:
            raise ValueError("noise_sd must be positive")
        self.noise_sd = float(noise_sd)
        self.base_diffusion = 0.15
        self.mode = torch.arange(1, self.dimension + 1,
                                 dtype=self.dtype, device=self.device)
        self.sigma = 0.4 * torch.pow(4.0, -self.mode)
        self.odd = (self.mode.to(torch.int64) % 2 == 1)
        self._fem_cache = {}
        self._quad_cache = {}
        self.u_true = None
        self.noiseless_y = None

        if y is None:
            # A local RNG makes the dataset independent of later training draws.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.u_true = (2.0 * torch.rand(
                self.dimension, generator=generator, dtype=self.dtype) - 1.0
            ).to(self.device)
            # l+3=20: the data-generation mesh specified in the paper.
            self.noiseless_y = self.fem_values(self.u_true, level=17)[[0, 2]]
            noise = self.noise_sd * torch.randn(
                2, generator=generator, dtype=self.dtype).to(self.device)
            self.y = self.noiseless_y + noise
        else:
            self.y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        if self.y.shape != (2,) or not torch.isfinite(self.y).all():
            raise ValueError("y must contain two finite observations")
        self.y = self.y.detach().clone()

    def _batch(self, values):
        values = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError(f"Expected shape [..., {self.dimension}]")
        if not torch.isfinite(values).all():
            raise ValueError("Parameters must be finite")
        return values.reshape(-1, self.dimension), values.shape[:-1]

    def _physical_batch(self, u):
        batch, shape = self._batch(u)
        if (batch.abs() > 1).any():
            raise ValueError("PDE parameters u must lie in [-1,1]^dimension")
        return batch, shape

    def _basis(self, x):
        """Weighted sine/cosine modes at spatial points x: [points, dimension]."""
        phase = math.pi * x[:, None] * self.mode[None, :]
        return torch.where(self.odd[None, :], phase.sin(), phase.cos()) * self.sigma

    @staticmethod
    def _mesh_size(level):
        if not isinstance(level, Integral) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        return 2 ** (int(level) + 3)

    def _element_data(self, n, start=0, stop=None):
        """Exact element-average coefficient basis and cumulative FEM loads."""
        stop = n if stop is None else stop
        e = torch.arange(start, stop, dtype=self.dtype, device=self.device)
        h = 1.0 / n
        midpoint = (e + 0.5) * h
        # torch.sinc(t)=sin(pi*t)/(pi*t): exact average of each mode.
        basis = self._basis(midpoint) * torch.sinc(self.mode * h / 2.0)
        # For interior node i, b_i = integral 100*x*hat_i(x) dx = 100*h^2*i.
        # Q_e = sum_{i=1}^e b_i = 50*h^2*e*(e+1).
        Q = 50.0 * h * h * e * (e + 1.0)
        return basis, Q

    @torch.no_grad()
    def fem_values(self, u, level: int):
        """Return p_l(1/4;u), p_l(1/2;u), p_l(3/4;u).

        This solves the paper's continuous, piecewise-linear FEM system with
        exact element integrals. Flux elimination is algebraically equivalent
        to the tridiagonal solve, not a different PDE discretization.
        Work is linear in the element count for a fixed parameter dimension.
        """
        ub, shape = self._physical_batch(u)
        n = self._mesh_size(level)
        h = 1.0 / n
        quarter = n // 4
        answers = []
        if n <= self.element_chunk and n not in self._fem_cache:
            self._fem_cache[n] = self._element_data(n)
        for batch in ub.split(self.parameter_batch):
            if n <= self.element_chunk:
                basis, Q = self._fem_cache[n]
                a_average = self.base_diffusion + batch @ basis.T
                resistance = h / a_average
                R = resistance.reshape(-1, 4, quarter).sum(-1)
                RQ = (resistance * Q).reshape(-1, 4, quarter).sum(-1)
            else:
                # Bound memory even for the 2**20-element data-generation solve.
                R = batch.new_zeros((len(batch), 4))
                RQ = batch.new_zeros((len(batch), 4))
                for j in range(4):
                    for start in range(j * quarter, (j + 1) * quarter,
                                       self.element_chunk):
                        stop = min(start + self.element_chunk, (j + 1) * quarter)
                        basis, Q = self._element_data(n, start, stop)
                        resistance = h / (self.base_diffusion + batch @ basis.T)
                        R[:, j] += resistance.sum(-1)
                        RQ[:, j] += (resistance * Q).sum(-1)
            # Discrete flux: F_e = F_0-Q_e; zero boundary values determine F_0.
            F0 = RQ.sum(-1) / R.sum(-1)
            pressure = F0[:, None] * R.cumsum(-1) - RQ.cumsum(-1)
            answers.append(pressure[:, :3])
        if not answers:
            return ub.new_empty((*shape, 3))
        return torch.cat(answers).reshape(*shape, 3)

    def _quadrature_data(self, order):
        if order not in self._quad_cache:
            t, w = np.polynomial.legendre.leggauss(order)
            t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
            left = torch.arange(4, dtype=self.dtype, device=self.device) / 4.0
            x = (left[:, None] + (t[None, :] + 1.0) / 8.0).reshape(-1)
            weights = w.repeat(4) / 8.0
            self._quad_cache[order] = (x, weights, self._basis(x))
        return self._quad_cache[order]

    @torch.no_grad()
    def continuum_values(self, u, order: Optional[int] = None):
        """High-accuracy numerical continuum values at 1/4, 1/2, 3/4.

        Integrating -(a*p')'=100*x gives a*p'=c-50*x^2.
        c=50*integral(x^2/a)/integral(1/a), with both integrals over [0,1].
        Gauss-Legendre quadrature on four subintervals evaluates this formula.
        This is a numerical reference, not a claim of exact quadrature.
        """
        ub, shape = self._physical_batch(u)
        order = self.reference_order if order is None else int(order)
        if order < 2:
            raise ValueError("quadrature order must be at least 2")
        x, w, basis = self._quadrature_data(order)
        answers = []
        for batch in ub.split(self.parameter_batch):
            inv_a = w / (self.base_diffusion + batch @ basis.T)
            I0 = inv_a.reshape(-1, 4, order).sum(-1)
            I2 = (inv_a * x.square()).reshape(-1, 4, order).sum(-1)
            c = 50.0 * I2.sum(-1) / I0.sum(-1)
            pressure = c[:, None] * I0.cumsum(-1) - 50.0 * I2.cumsum(-1)
            answers.append(pressure[:, :3])
        if not answers:
            return ub.new_empty((*shape, 3))
        return torch.cat(answers).reshape(*shape, 3)

    @torch.no_grad()
    def log_density_u(self, u, level: Optional[int]):
        """log g_l(u), excluding constants; -inf outside the uniform prior.

        level=None uses the continuum quadrature forward map for a reference
        posterior. Finite integer levels use the FEM forward map.
        """
        ub, shape = self._batch(u)
        inside = (ub.abs() <= 1.0).all(-1)
        result = ub.new_full((len(ub),), -torch.inf)
        if inside.any():
            values = (self.continuum_values(ub[inside]) if level is None
                      else self.fem_values(ub[inside], level))
            residual = (values[:, [0, 2]] - self.y) / self.noise_sd
            result[inside] = -0.5 * residual.square().sum(-1)
        return result.reshape(shape)

    @torch.no_grad()
    def log_density_z(self, z, level: Optional[int]):
        """Density in unconstrained LightSB coordinates u=tanh(z)."""
        zb, shape = self._batch(z)
        a = zb.abs()
        log_jacobian = (2.0 * (math.log(2.0) - a - F.softplus(-2.0 * a))).sum(-1)
        value = self.log_density_u(zb.tanh(), level) + log_jacobian
        return value.reshape(shape)

    def make_log_densities(self, L: int):
        self._mesh_size(L)  # validate
        return [partial(self.log_density_z, level=l) for l in range(L + 1)]

    @torch.no_grad()
    def phi_u(self, u):
        """Paper's fixed observable phi(u)=p(0.5;u), via continuum quadrature."""
        return self.continuum_values(u)[..., 1]

    @torch.no_grad()
    def phi_z(self, z):
        """Same physical observable evaluated on LightSB coordinates."""
        zb, shape = self._batch(z)
        return self.phi_u(zb.tanh()).reshape(shape)

    @torch.no_grad()
    def phi_level_z(self, z, level: int):
        """Optional level-dependent observable p_l(0.5;tanh(z))."""
        zb, shape = self._batch(z)
        return self.fem_values(zb.tanh(), level)[..., 1].reshape(shape)

    @torch.no_grad()
    def sample_posterior(self, n: int, level: Optional[int] = 0, seed: int = 123,
                         return_z: bool = True, dtype=torch.float32,
                         proposal_batch: int = 8192,
                         max_proposals: int = 20_000_000):
        """Prior-rejection sampling of a specified posterior (usually level 0).

        Acceptance probability exp(-Phi_l) is bounded by one. Samples are iid
        from the numerical posterior, up to arithmetic/RNG accuracy. No burn-in.
        Returns (samples, diagnostics). May be expensive for a concentrated
        likelihood; max_proposals raises rather than silently returning a bias.
        """
        if not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        if proposal_batch < 1 or max_proposals < 1:
            raise ValueError("proposal_batch and max_proposals must be positive")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        kept, n_kept, proposed = [], 0, 0
        while n_kept < n:
            count = min(proposal_batch, max_proposals - proposed)
            if count <= 0:
                raise RuntimeError(
                    f"Rejection sampling accepted {n_kept}/{n} after "
                    f"{proposed} proposals. Increase max_proposals or use an "
                    "MCMC initialization with convergence diagnostics.")
            u = 2.0 * torch.rand((count, self.dimension), generator=generator,
                                  dtype=self.dtype, device=self.device) - 1.0
            log_accept = self.log_density_u(u, level)
            log_uniform = torch.rand(count, generator=generator, dtype=self.dtype,
                                     device=self.device).log()
            accepted = u[log_uniform < log_accept]
            kept.append(accepted)
            n_kept += len(accepted)
            proposed += count
        samples = torch.cat(kept, dim=0)[:n]
        if return_z:
            eps = torch.finfo(self.dtype).eps
            samples = torch.atanh(samples.clamp(-1.0 + eps, 1.0 - eps))
        diagnostics = {"proposals": proposed, "accepted_before_truncation": n_kept,
                       "acceptance_rate": n_kept / proposed}
        return samples.to(dtype=dtype), diagnostics

    @torch.no_grad()
    def reference_2d(self, level: Optional[int], order: int = 160):
        """Deterministic posterior expectation of phi, only for dimension=2.

        Tensor Gauss-Legendre quadrature in physical parameter space [-1,1]^2.
        Re-run with a larger order to assess integration error.
        """
        if self.dimension != 2:
            raise ValueError("reference_2d is only valid for dimension=2")
        t, w = np.polynomial.legendre.leggauss(order)
        t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
        U1, U2 = torch.meshgrid(t, t, indexing="ij")
        u = torch.stack((U1.reshape(-1), U2.reshape(-1)), dim=-1)
        integration_weight = (w[:, None] * w[None, :]).reshape(-1)
        log_weight = self.log_density_u(u, level) + integration_weight.log()
        posterior_weight = torch.softmax(log_weight, dim=0)
        return float((posterior_weight * self.phi_u(u)).sum())


@torch.no_grad()
def blocked_rwm_reference(problem: BeskosElliptic, level: Optional[int],
                          n_chains: int = 16, draws: int = 2000,
                          burn_in: int = 200, tail_block_size: int = 5,
                          seed: int = 9123):
    """Optional independent-chain blocked random-walk Metropolis reference.

    This is an auxiliary validation sampler, NOT a reproduction of the paper's
    full MLSMC algorithm or its exact proposal tuning. Works in physical u.
    Each chain starts independently from the specified numerical posterior via
    rejection sampling. Blocked RWM therefore preserves this target from start.
    SE is std(independent chain means)/sqrt(n_chains), not iid SE across draws.
    Estimates still have Monte Carlo and forward/quadrature approximation error.
    """
    if n_chains < 2 or draws < 2 or burn_in < 0 or tail_block_size < 1:
        raise ValueError("Need >=2 chains/draws, nonnegative burn-in, positive blocks")
    d = problem.dimension
    u, init = problem.sample_posterior(n_chains, level=level, seed=seed,
                                       return_z=False, dtype=torch.float64)
    blocks = [[j] for j in range(min(2, d))]
    blocks += [list(range(j, min(j + tail_block_size, d)))
               for j in range(2, d, tail_block_size)]
    # User-adjustable implementation choices, not published tuning constants.
    # transport_mh copy: the two data-informed coordinates' step sizes scale with
    # the observation noise (identical to the original at noise_sd=0.25).
    shrink = problem.noise_sd / 0.25
    scales = [0.02 * shrink if block == [0] else 0.08 * shrink if block == [1] else 0.25
              for block in blocks]
    generator = torch.Generator(device=problem.device).manual_seed(seed + 1)
    logp = problem.log_density_u(u, level)
    saved = []
    accepted = torch.zeros(len(blocks), dtype=torch.float64, device=problem.device)
    for step in range(burn_in + draws):
        for b, (block, scale) in enumerate(zip(blocks, scales)):
            proposal = u.clone()
            proposal[:, block] += scale * torch.randn(
                (n_chains, len(block)), generator=generator,
                dtype=torch.float64, device=problem.device)
            logp_proposal = problem.log_density_u(proposal, level)
            log_uniform = torch.rand(n_chains, generator=generator,
                                     dtype=torch.float64, device=problem.device).log()
            take = log_uniform < (logp_proposal - logp)
            u = torch.where(take[:, None], proposal, u)
            logp = torch.where(take, logp_proposal, logp)
            accepted[b] += take.sum()
        if step >= burn_in:
            saved.append(problem.phi_u(u))
    values = torch.stack(saved)             # [draws, independent chains]
    chain_means = values.mean(0)
    return {
        "estimate": float(chain_means.mean()),
        "se": float(chain_means.std(unbiased=True) / math.sqrt(n_chains)),
        "chain_means": chain_means.cpu(),
        "block_acceptance": (accepted / ((burn_in + draws) * n_chains)).cpu(),
        "initialization": init,
        "values": values.cpu(),
    }


def self_test():
    """Analytic solution, independent FEM solve, refinement, support/Jacobian."""
    problem = BeskosElliptic(dimension=50, y=[20.0, 30.0])
    positions = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    exact = 100.0 / (6.0 * 0.15) * (positions - positions.pow(3))
    zero = torch.zeros(50, dtype=torch.float64)
    errors = []
    for level in range(6):
        errors.append(float((problem.fem_values(zero, level) - exact).abs().max()))
    torch.testing.assert_close(problem.continuum_values(zero), exact,
                               rtol=1e-12, atol=1e-11)
    generator = torch.Generator().manual_seed(7)
    u = 1.8 * torch.rand((5, 50), generator=generator, dtype=torch.float64) - 0.9
    dense_errors = []
    for level in (0, 1, 2):
        n = problem._mesh_size(level)
        basis, _ = problem._element_data(n)
        a = problem.base_diffusion + u @ basis.T
        conductance = n * a
        main = conductance[:, :-1] + conductance[:, 1:]
        off = -conductance[:, 1:-1]
        matrix = torch.diag_embed(main) + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
        nodes = torch.arange(1, n, dtype=torch.float64) / n
        rhs = (100.0 * nodes / n).expand(len(u), -1)
        p = torch.linalg.solve(matrix, rhs)
        at_quarters = p[:, [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]]
        fast = problem.fem_values(u, level)
        dense_errors.append(float((fast - at_quarters).abs().max()))
        torch.testing.assert_close(fast, at_quarters, rtol=1e-11, atol=1e-10)
    reference = problem.continuum_values(u, order=64)
    ref128 = problem.continuum_values(u, order=128)
    torch.testing.assert_close(reference, ref128, rtol=1e-11, atol=1e-10)
    e4 = float((problem.fem_values(u, 4) - reference).abs().max())
    e5 = float((problem.fem_values(u, 5) - reference).abs().max())
    assert e5 < e4
    outside = torch.zeros(50, dtype=torch.float64)
    outside[0] = 1.01
    assert torch.isneginf(problem.log_density_u(outside, 0))
    z = torch.atanh(u)
    dz = problem.log_density_z(z, 1) - problem.log_density_z(z, 0)
    du = problem.log_density_u(u, 1) - problem.log_density_u(u, 0)
    torch.testing.assert_close(dz, du, rtol=1e-10, atol=1e-10)
    print("Analytic constant-coefficient FEM max error:", max(errors))
    print("Dense tridiagonal vs flux-FEM max error:", max(dense_errors))
    print("Continuum quadrature 64 vs 128 max difference:", float((reference - ref128).abs().max()))
    print("FEM refinement errors (levels 4,5):", e4, e5)
    print("Support and transformed-density ratio checks: passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    self_test()
