"""Checks for beskos_transport_mh.py. Run from this directory:
    python test_transport_mh.py
Execution, accounting and MH-correctness checks; not convergence experiments.
"""
import math
import torch
from threadpoolctl import threadpool_limits

import beskos_transport_mh as tm


class GaussianTiltProblem:
    """Fake problem: logistic(1/2) coordinates (u uniform) with a Gaussian tilt
    on z_0. Known moments by importance sampling; level is ignored."""
    def log_density_z(self, z, level):
        a = z.abs()
        base = (math.log(2.0) - 2.0 * a - 2.0 * torch.nn.functional.softplus(-2.0 * a)).sum(-1)
        return base - 0.5 * ((z[:, 0] - 0.4) / 0.3) ** 2


def reference_moments(d, n=2_000_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.atanh(2 * torch.rand(n, d, generator=g, dtype=torch.float64) - 1)
    lw = -0.5 * ((z[:, 0] - 0.4) / 0.3) ** 2
    w = (lw - lw.max()).exp()
    w = w / w.sum()
    mean = (w[:, None] * z).sum(0)
    return mean, (w[:, None] * z * z).sum(0) - mean ** 2


def check_mh_targets():
    torch.manual_seed(0)
    d, n = 3, 20000
    problem = GaussianTiltProblem()
    mean, var = reference_moments(d)
    clock = tm.Timers()
    # pCN with Gaussian reference refitted each block of sweeps.
    z = 0.5 * torch.randn(n, d, dtype=torch.float64) + 1.0
    lg = problem.log_density_z(z, 0)
    for _ in range(30):
        W = tm.fit_whitening(z, "full", 1e-3)
        z, lg, info = tm.pcn_mh(problem, z, lg, 0, 0.8, 10, W, clock)
    assert (z.mean(0) - mean).abs().max() < 0.03, (z.mean(0), mean)
    assert (z.var(0) - var).abs().max() < 0.04, (z.var(0), var)
    # LightSB-MH with a RANDOM (untrained) kernel must still target the right law.
    model = tm.LightSB(d, 3, 0.4, 0.5).double()
    with torch.no_grad():
        model.r.normal_(0, 0.3)
        model.log_s.normal_(-0.3, 0.2)
        model.log_alpha_raw.normal_()
    z = 0.5 * torch.randn(n, d, dtype=torch.float64) + 1.0
    lg = problem.log_density_z(z, 0)
    z, lg, info = tm.lightsb_mh(problem, z, lg, 0, model, None, 2000, clock)
    assert (z.mean(0) - mean).abs().max() < 0.03, (z.mean(0), mean)
    assert (z.var(0) - var).abs().max() < 0.04, (z.var(0), var)
    print("pCN-MH and LightSB-MH (random kernel) converge to the known target: passed")


def check_kernel_pairs():
    torch.manual_seed(1)
    model = tm.LightSB(4, 3, 0.3, 0.5).double()
    with torch.no_grad():
        model.r.normal_(); model.log_s.normal_(-0.5, 0.3); model.log_alpha_raw.normal_()
    x = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(6, 4, dtype=torch.float64)
    pairs = tm.log_kernel_pairs(model, x, y)
    single = torch.stack([tm.log_marginal_density(model, x[i:i + 1], y[i:i + 1])[0] for i in range(6)])
    torch.testing.assert_close(pairs, single, rtol=1e-10, atol=1e-10)
    print("Pairwise kernel density == one-parent mixture density: passed")


def check_runs():
    for d in (2, 50):
        problem = tm.MeasuredBeskos(dimension=d, y=[26.0, 40.0])
        g = torch.Generator().manual_seed(11)
        initial = torch.randn(20, d, generator=g, dtype=torch.float64) * 0.1
        light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3,
                     whiten="full", init="identity", dtype="float64")
        # sweeps=0 reproduces the LightSB paired method exactly.
        for N in ([20, 20, 20], [20, 14, 10]):
            a = tm.run_light_sb(problem, initial, N, light, seed=5)
            b = tm.run_transport_mh(problem, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=0), seed=5)
            assert a["estimate"] == b["estimate"], (a["estimate"], b["estimate"])
            assert a["fem_solves_by_level"] == b["fem_solves_by_level"]
        # sweeps=2: per level l -> N[l-1] ratio + N[l] child density + 2 N[l] MH solves.
        N = [20, 14, 10]
        ans = tm.run_transport_mh(problem, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=2), seed=6)
        assert ans["fem_solves_by_level"] == {"0": 20, "1": 20 + 3 * 14, "2": 14 + 3 * 10}, ans["fem_solves_by_level"]
        assert ans["continuum_forward_evals"] == sum(N)   # diagnostic phi is untracked
        for stage in ans["diagnostics"]:
            assert 0 <= stage["importance"]["mh_acceptance_fraction"] <= 1
        # MLSMC with pCN / LightSB mutation: 1 solve per particle per sweep, no final mutation.
        kcfg = dict(tm.MLSMC_KERNEL_DEFAULTS, sweeps=2, iterations=5)
        for kind in ("pcn", "lsb"):
            m = tm.run_mlsmc_kernel(problem, initial, N, light, kcfg, kind, seed=7)
            assert m["fem_solves_by_level"] == {"0": 20, "1": 20 + 2 * 14, "2": 14}, (kind, m["fem_solves_by_level"])
            assert [t["n_correction"] for t in m["contributions"]] == [20, 20, 14]
            assert m["continuum_forward_evals"] == sum(N[:-1])
            assert math.isfinite(m["estimate"])
        print(f"d={d}: TransportMH (sweeps=0 identity, accounting), MLSMC_pCN/LSB runs: passed")
    cfg = tm.merge_config({"configurations": [{"name": "A", "N": [4, 4]}],
                           "methods": ["LightSB", "MLSMC", "TransportMH", "MLSMC_pCN", "MLSMC_LSB"],
                           "transport_mh": {"sweeps": 3}})
    assert cfg["transport_mh"]["sweeps"] == 3 and cfg["transport_mh"]["rho"] == "auto"
    for bad in ({"transport_mh": {"rho": 1.0}}, {"mlsmc_kernel": {"sweeps": 0}}):
        try:
            tm.merge_config({"configurations": [{"name": "A", "N": [4, 4]}], **bad})
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print("Config defaults and validation: passed")


def check_noise_option():
    a = tm.MeasuredBeskos(dimension=50, y=None, seed=2026)
    b = tm.MeasuredBeskos(dimension=50, y=None, seed=2026, noise_sd=0.05)
    assert a.noise_sd == 0.25 and b.noise_sd == 0.05
    torch.testing.assert_close(a.noiseless_y, b.noiseless_y)
    torch.testing.assert_close((b.y - b.noiseless_y) * 5.0, a.y - a.noiseless_y)   # same draw, scaled
    u = 1.8 * torch.rand(9, 50, dtype=torch.float64) - 0.9
    base = tm.BeskosElliptic(dimension=50, y=b.y, noise_sd=0.05)
    torch.testing.assert_close(b.log_density_u(u, 2), base.log_density_u(u, 2), rtol=1e-9, atol=1e-8)
    # untimed initialisation uses the flux solver and leaves counters untouched
    b.solver = "dense"
    b.reset_work()
    samples, info = b.sample_posterior(5, level=0, seed=3, dtype=torch.float64, max_proposals=10_000_000)
    assert samples.shape == (5, 50) and b.work()["fem_solves_raw"] == 0
    print("noise_sd option, scaled data, flux-solver initialisation: passed")


def check_is_reference():
    """IS reference agrees with an iid rejection estimate; both SEs are sane."""
    problem = tm.MeasuredBeskos(dimension=50, y=None, seed=2026, noise_sd=0.25)
    spec = {"seed": 4, "max_proposals": 50_000_000, "is_samples": 400_000, "is_pilot": 1000,
            "is_batch": 20000, "is_batches_for_se": 20}
    ref = tm.importance_sampling_reference(problem, spec)
    u, _ = problem.sample_posterior(4000, level=None, seed=11, return_z=False, dtype=torch.float64,
                                    max_proposals=50_000_000)
    phi = problem.phi_u(u)
    rej, rej_se = float(phi.mean()), float(phi.std() / math.sqrt(len(phi)))
    z = abs(ref["value"] - rej) / math.sqrt(ref["se"] ** 2 + rej_se ** 2)
    assert z < 4, (ref["value"], ref["se"], rej, rej_se)
    assert 0.3 < ref["se"] / ref["batch_se"] < 3, (ref["se"], ref["batch_se"])
    assert 0 < ref["ess_fraction"] <= 1
    print(f"IS reference {ref['value']:.4f}+-{ref['se']:.4f} (ESS {ref['ess_fraction']:.3f}) vs rejection "
          f"{rej:.4f}+-{rej_se:.4f}: passed")


def check_source_problem():
    """Bimodal source-location problem: forward models, symmetry, accounting, methods, IS."""
    prob = tm.SourceMomentProblem(seed=2026)
    assert prob.dimension == 51 and len(prob.stored_y()) == 6
    g = torch.Generator().manual_seed(0)
    u = 2 * torch.rand(6, 51, generator=g, dtype=torch.float64) - 1
    c = prob.continuum_values(u)
    torch.testing.assert_close(prob.continuum_values(u, order=128), c, rtol=1e-12, atol=1e-12)
    assert (prob.fem_values(u, 9) - c).abs().max() < 1e-6
    assert (prob.fem_values(u, 2) - c).abs().max() > (prob.fem_values(u, 4) - c).abs().max()
    mirror = u.clone(); mirror[:, 0] = -mirror[:, 0]
    torch.testing.assert_close(prob.continuum_values(mirror)[:, :2], c[:, :2], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(prob.fem_values(mirror, 5)[:, :2], prob.fem_values(u, 5)[:, :2], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(prob.phi_u(u), (u[:, 0] + 1) / 2)
    prob.reset_work(); prob.log_density_z(torch.atanh(u * 0.99), 3)
    assert prob.work()["fem_solves_by_level"] == {"3": 6}
    # same y reproduces the same problem
    again = tm.SourceMomentProblem(y=prob.stored_y())
    torch.testing.assert_close(again.log_density_u(u, 4), prob.log_density_u(u, 4))
    # every method runs in d = 51 and records mode fractions
    initial, _ = prob.sample_posterior(60, level=0, seed=5, dtype=torch.float64, max_proposals=50_000_000)
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64")
    N = [60, 60, 60]
    ans = tm.run_light_sb(prob, initial, N, light, seed=1)
    assert "right_mode_fraction" in ans["diagnostics"][0]["importance"]
    ans = tm.run_transport_mh(prob, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=2), seed=2)
    assert "mh_mode_switches_per_particle_sweep" in ans["diagnostics"][0]["importance"]
    kcfg = dict(tm.MLSMC_KERNEL_DEFAULTS, sweeps=2, iterations=5)
    for kind in ("pcn", "lsb"):
        m = tm.run_mlsmc_kernel(prob, initial, N, light, kcfg, kind, seed=3)
        assert "right_mode_fraction" in m["diagnostics"][0]
    m = tm.run_smc(prob, initial, N, tm.DEFAULTS["rwm"], 4, True)
    assert math.isfinite(m["estimate"])
    # mixture IS reference agrees with rejection sampling
    spec = {"seed": 4, "max_proposals": 200_000_000, "is_samples": 400_000, "is_pilot": 1000,
            "is_batch": 20000, "is_batches_for_se": 20, "is_split_coord": 0}
    ref = tm.importance_sampling_reference(prob, spec)
    assert len(ref["proposal"]["component_weights"]) == 2
    uu, _ = prob.sample_posterior(3000, level=None, seed=11, return_z=False, dtype=torch.float64,
                                  max_proposals=200_000_000)
    phi = prob.phi_u(uu); rej, rej_se = float(phi.mean()), float(phi.std() / math.sqrt(len(phi)))
    right = float((phi > 0.5).double().mean())
    assert 0.1 < right < 0.45, f"mode weights not as designed (right-mode fraction {right:.3f})"
    assert abs(ref["value"] - rej) / math.sqrt(ref["se"] ** 2 + rej_se ** 2) < 4, (ref["value"], rej)
    print(f"source problem: FEM->continuum, mirror symmetry, accounting, 5 methods, mixture IS "
          f"({ref['value']:.4f}+-{ref['se']:.4f}, ESS {ref['ess_fraction']:.3f}; rejection {rej:.4f}+-{rej_se:.4f}): passed")


def check_mesh_sensitive_source():
    """Weak sensor at the source position: coarse meshes misjudge the mode weights."""
    base = tm.SourceMomentProblem(seed=2026)
    assert "weak_sensor_x" not in base.settings and base.description()["observations"].endswith("m0(1/4)")
    xw = 5.0 / 12.0
    prob = tm.SourceMomentProblem(seed=2026, xi_true=xw, weak_sensor_x=xw, start_elements=4, weak_separation=2.5)
    assert prob.settings["weak_sensor_x"] == xw and prob.start_elements == 4
    g = torch.Generator().manual_seed(1)
    u = 2 * torch.rand(8, 51, generator=g, dtype=torch.float64) - 1
    c = prob.continuum_values(u)
    torch.testing.assert_close(prob.continuum_values(u, order=128), c, rtol=1e-11, atol=1e-11)
    # strong sensors unchanged and still mirror invariant
    torch.testing.assert_close(c[:, :2], base.continuum_values(u)[:, :2], rtol=1e-12, atol=1e-12)
    mirror = u.clone(); mirror[:, 0] = -mirror[:, 0]
    torch.testing.assert_close(prob.fem_values(mirror, 3)[:, :2], prob.fem_values(u, 3)[:, :2], rtol=1e-12, atol=1e-12)
    # off-node weak sensor: FEM interpolation converges to the closed form
    errs = [float((prob.fem_values(u, l)[:, 2] - c[:, 2]).abs().max()) for l in (0, 3, 9)]
    assert errs[2] < 2e-3 and errs[2] < errs[1] < errs[0], errs
    # weak sensor at x_w via a closed-form check in the continuum: m0(x; xi) symmetric in (x, xi)
    v = u.clone(); v[:, 0] = 2 * xw - 1
    a = prob.continuum_values(v)[:, 2]
    assert (prob.fem_values(v, 0)[:, 2] - a).abs().min() > 0.05      # kink cut off on the coarsest mesh
    # mode weights move across levels (coarse levels overweight the mirror mode)
    fr = []
    for level in (0, 6):
        z, _ = prob.sample_posterior(1500, level=level, seed=21 + level, dtype=torch.float64,
                                     max_proposals=400_000_000)
        fr.append(float(prob.mode_label(z).double().mean()))
    assert fr[0] > 0.15 and fr[1] < 0.12 and fr[0] - fr[1] > 0.08, fr
    print(f"mesh-sensitive source variant: interpolation errors {errs}, right-mode fraction "
          f"level0 {fr[0]:.3f} -> level6 {fr[1]:.3f}: passed")


def check_lr_schedule():
    """light.lr_schedule='cosine' runs, is validated, and leaves 'constant' unchanged."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(300, 4, generator=g, dtype=torch.float64)
    lr = 0.3 * x[:, 0]
    base = dict(tm.DEFAULTS["light"], iterations=20, batch_size=32, K=3, epsilon=0.1, dtype="float64",
                target_batching="global_resample", whiten="full", init="identity")
    torch.manual_seed(0); a, _, _ = tm.fit_light_sb(x, lr, base)
    torch.manual_seed(0); b, _, _ = tm.fit_light_sb(x, lr, dict(base, lr_schedule="constant"))
    torch.testing.assert_close(a.r, b.r)
    torch.manual_seed(0); c, _, _ = tm.fit_light_sb(x, lr, dict(base, lr_schedule="cosine"))
    assert torch.isfinite(c.r).all() and not torch.equal(a.r, c.r)
    try:
        tm.merge_config({"configurations": [{"name": "A", "N": [4, 4]}], "light": {"lr_schedule": "step"}})
        raise AssertionError("invalid lr_schedule accepted")
    except ValueError:
        pass
    print("LightSB lr_schedule (constant unchanged, cosine runs, validation): passed")


def check_ess_rho():
    """transport_mh.rho = 'ess': ESS-adaptive pCN step for TransportMH."""
    mh = dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=3)
    eps = 3.125e-3
    r_lo, n_lo = tm.ess_adaptive_rho(0.003, eps, mh)
    r_hi, _ = tm.ess_adaptive_rho(0.9, eps, mh)
    r_mid, _ = tm.ess_adaptive_rho(0.091, eps, mh)
    assert abs(r_lo - 0.8) < 1e-12 and n_lo == 3
    assert abs(r_hi - tm.identity_scale(eps)) < 1e-12
    assert r_lo < r_mid < r_hi and abs(r_mid - 0.9465) < 2e-3, r_mid
    rs = [tm.ess_adaptive_rho(e, eps, mh)[0] for e in (0.01, 0.05, 0.1, 0.2, 0.5, 0.7)]
    assert all(a <= b + 1e-15 for a, b in zip(rs, rs[1:])), rs
    # If S(eps) < ess_rho_min (large eps), 'auto' already moves more: rho = S(eps) at every ESS.
    for e in (0.001, 0.2, 0.9):
        assert abs(tm.ess_adaptive_rho(e, 0.5, mh)[0] - tm.identity_scale(0.5)) < 1e-12
    m10 = dict(mh, ess_sweeps=10)
    assert tm.ess_adaptive_rho(0.01, eps, m10)[1] == 10 and tm.ess_adaptive_rho(0.6, eps, m10)[1] == 3
    assert tm.ess_adaptive_rho(0.097, eps, m10)[1] == 8      # t = 0.288 -> round(10 - 0.288*7) = 8
    ns = [tm.ess_adaptive_rho(e, eps, m10)[1] for e in (0.01, 0.05, 0.1, 0.2, 0.5, 0.9)]
    assert all(a >= b for a, b in zip(ns, ns[1:])), ns
    base = {"configurations": [{"name": "A", "N": [4, 4]}]}
    tm.merge_config(dict(base, transport_mh={"rho": "ess"}))
    for bad in ({"transport_mh": {"rho": "ess", "ess_low": 0.6}}, {"mlsmc_kernel": {"rho": "ess"}}):
        try:
            tm.merge_config(dict(base, **bad))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    problem = tm.MeasuredBeskos(dimension=50, y=[26.0, 40.0])
    g = torch.Generator().manual_seed(11)
    initial = torch.randn(20, 50, generator=g, dtype=torch.float64) * 0.1
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64")
    ans = tm.run_transport_mh(problem, initial, [20, 14, 10], light, dict(mh, rho="ess", sweeps=2), seed=6)
    for stage in ans["diagnostics"]:
        d = stage["importance"]
        assert d["mh_rho_rule"] == "ess" and 0 < d["mh_training_ess_fraction"] <= 1
        expected, _ = tm.ess_adaptive_rho(d["mh_training_ess_fraction"], d["epsilon_level"], dict(mh, sweeps=2))
        assert abs(d["pcn_rho"] - expected) < 1e-12, (d["pcn_rho"], expected)
    print("TransportMH ESS-adaptive rho (interpolation, sweeps, validation, run): passed")


def check_iterations_by_level():
    """light.iterations_by_level: per-level SGD budget, last entry reused; absent = unchanged."""
    base = dict(tm.DEFAULTS["light"], epsilon=0.05, lr=0.01, epsilon_schedule="geometric",
                epsilon_decay=0.25, lr_epsilon_scaling=True)
    assert tm.level_settings(base, 3)["iterations"] == base["iterations"]
    sched = dict(base, iterations_by_level=[1000, 1000, 1000, 1000, 100])
    got = [tm.level_settings(sched, l)["iterations"] for l in range(1, 10)]
    assert got == [1000] * 4 + [100] * 5, got
    flat = dict(tm.DEFAULTS["light"], iterations_by_level=[7])   # constant eps, no lr scaling
    assert tm.level_settings(flat, 2)["iterations"] == 7
    assert tm.level_settings(tm.DEFAULTS["light"], 2) is tm.DEFAULTS["light"]
    cfgbase = {"configurations": [{"name": "A", "N": [4, 4]}]}
    tm.merge_config(dict(cfgbase, light={"iterations_by_level": [50, 10]}))
    for bad in ([], [0], [10, "x"], 5):
        try:
            tm.merge_config(dict(cfgbase, light={"iterations_by_level": bad}))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    problem = tm.MeasuredBeskos(dimension=50, y=[26.0, 40.0])
    g = torch.Generator().manual_seed(11)
    initial = torch.randn(20, 50, generator=g, dtype=torch.float64) * 0.1
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64", lr_schedule="cosine", target_batching="global_resample",
                 iterations_by_level=[6, 3])
    ans = tm.run_transport_mh(problem, initial, [20, 14, 10], light, dict(tm.TRANSPORT_MH_DEFAULTS), seed=6)
    assert [s["importance"]["iterations_level"] for s in ans["diagnostics"]] == [6, 3]
    print("light.iterations_by_level (schedule, default unchanged, validation, run): passed")


def check_mlsmc_ess_sweeps():
    """mlsmc_kernel.ess_sweeps: same interpolation as TransportMH's ess_sweeps; default unchanged."""
    k = dict(tm.MLSMC_KERNEL_DEFAULTS, sweeps=3)
    assert tm.mlsmc_level_sweeps(0.01, k) == 3                       # unset: always sweeps
    k30 = dict(k, ess_sweeps=30)
    got = [tm.mlsmc_level_sweeps(e, k30) for e in (0.01, 0.05, 0.097, 0.6)]
    mh = dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=3, ess_sweeps=30)
    want = [tm.ess_adaptive_rho(e, 0.05, mh)[1] for e in (0.01, 0.05, 0.097, 0.6)]
    assert got == want and got[0] == 30 and got[-1] == 3, (got, want)
    base = {"configurations": [{"name": "A", "N": [4, 4]}]}
    tm.merge_config(dict(base, mlsmc_kernel={"ess_sweeps": 30}))
    for bad in ({"ess_sweeps": 0}, {"ess_sweeps": "x"}, {"ess_low": 0.6}, {"foo": 1}):
        try:
            tm.merge_config(dict(base, mlsmc_kernel=bad))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    problem = tm.MeasuredBeskos(dimension=50, y=[26.0, 40.0])
    g = torch.Generator().manual_seed(11)
    initial = torch.randn(20, 50, generator=g, dtype=torch.float64) * 0.1
    light = dict(tm.DEFAULTS["light"], dtype="float64")
    ans = tm.run_mlsmc_kernel(problem, initial, [20, 14, 10], light, k30, "pcn", seed=5)
    stage = ans["diagnostics"][0]
    assert stage["level_sweeps"] == tm.mlsmc_level_sweeps(stage["ratio_ess_fraction"], k30)
    assert stage["mh_sweeps"] == stage["level_sweeps"]
    # PDE accounting: level 1 = N0 ratio solves + level_sweeps * N1 MH solves
    assert ans["fem_solves_by_level"]["1"] == 20 + stage["level_sweeps"] * 14, ans["fem_solves_by_level"]
    print("MLSMC ess_sweeps (same rule as TransportMH, default unchanged, validation, run): passed")


def check_darcy2d():
    """2D Darcy problem: exact constant-kappa solve, h^2 convergence, densities/work units,
    level-0 pool, figure-4 rows, config validation, and short MLSMC_pCN / TransportMH runs."""
    import numpy as np
    import darcy2d
    core = darcy2d.DarcyCore(d=6, sensors=3, noise_sd=0.05, data_level=3, seed=1)
    z0 = np.zeros((1, 6))
    for level in (0, 1, 3):                       # constant kappa: p = 1 - x exactly
        exact = np.array([1.0 - x for x, _ in core.sensors_xy])
        assert np.abs(core.level(level).sensor_values(z0)[0] - exact).max() < 1e-12
    z = np.random.default_rng(3).standard_normal((2, 6))
    d = [np.abs(core.level(l).sensor_values(z) - core.level(l - 1).sensor_values(z)).max() for l in (2, 3, 4)]
    assert d[1] < 0.5 * d[0] and d[2] < 0.5 * d[1], d          # ~h^2: about 4x per level
    prob = tm.Darcy2DProblem(seed=1, d=6, sensors=3, noise_sd=0.05, data_level=3)
    Z = torch.randn(4, 6, dtype=torch.float64)
    lp = prob.log_density_z(Z, 2)
    assert lp.shape == (4,) and torch.isfinite(lp).all()
    w = prob.work()
    assert w["fem_solves_by_level"] == {"2": 4} and w["fem_element_work"] == 4 * 2 * 16 ** 2
    prob.reset_work()
    pool, info = prob.sample_posterior(200, level=0, seed=5, dtype=torch.float64)
    assert pool.shape == (200, 6) and info["stages"] >= 1 and prob.work()["fem_solves_raw"] == 0
    rows = prob.pde_convergence([1, 2, 3], 7)["rows"]
    assert rows[2]["sensor_diff_max"] < rows[0]["sensor_diff_max"]
    base = {"configurations": [{"name": "A", "N": [4, 4]}], "problem": "darcy2d"}
    cfg = tm.merge_config(dict(base, darcy={"d": 6, "noise_sd": 0.05}))
    assert cfg["dimension"] == 6 and cfg["noise_sd"] == 0.05
    try:
        tm.merge_config(dict(base, darcy={"foo": 1}))
        raise AssertionError("accepted unknown darcy key")
    except ValueError:
        pass
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64", lr_schedule="cosine", target_batching="global_resample")
    k = dict(tm.MLSMC_KERNEL_DEFAULTS, sweeps=2, ess_sweeps=4)
    a = tm.run_mlsmc_kernel(prob, pool, [60, 30, 20], light, k, "pcn", seed=2)
    b = tm.run_transport_mh(prob, pool, [60, 30, 20], light, dict(tm.TRANSPORT_MH_DEFAULTS, rho="ess"), seed=3)
    for ans in (a, b):
        assert math.isfinite(ans["estimate"]) and ans["fem_element_work"] > 0
    print("Darcy 2D (exact solve, h^2 convergence, densities, work, pool, figure 4, config, runs): passed")


def check_ns2d():
    """2D Navier--Stokes problem: exact single-mode decay (fd2 and spectral), h^2 convergence of the
    fd2 levels, densities/work units, level-0 pool, figure-4 rows, config validation, short runs."""
    import numpy as np
    import ns2d
    for scheme in ("fd2", "spectral"):
        core = ns2d.NSCore(d=6, f_amp=0.0, data_level=1, scheme=scheme, seed=1)
        z = np.zeros(6); z[2] = 1.0
        for level in (0, 2):
            L = core.level(level)
            w0 = L.initial(z[None])[0]
            k = 2 * math.pi * core.modes[0][2]
            if scheme == "fd2":
                h = 1.0 / L.N
                lam = sum((2 * math.sin(kk * h / 2) / h) ** 2 for kk in k)
            else:
                lam = float((k ** 2).sum())
            assert np.abs(L.final_field(z) - w0 * math.exp(-core.nu * lam * core.T)).max() < 1e-12
    core = ns2d.NSCore(d=6, data_level=2, seed=1)
    Z = np.random.default_rng(3).standard_normal((2, 6))
    d = [np.abs(core.sensor_values(Z, l) - core.sensor_values(Z, l - 1)).max() for l in (2, 3, 4)]
    assert d[1] < 0.4 * d[0] and d[2] < 0.4 * d[1], d          # ~h^2: about 4x per level
    prob = tm.NS2DProblem(seed=1, d=6, data_level=2)
    Zt = torch.randn(4, 6, dtype=torch.float64)
    lp = prob.log_density_z(Zt, 1)
    assert lp.shape == (4,) and torch.isfinite(lp).all()
    w = prob.work()
    assert w["fem_solves_by_level"] == {"1": 4} and w["fem_element_work"] == 4 * 16 ** 2 * 16
    assert abs(float(prob.phi_z(Zt[:1])) - float(Zt[0].numpy() @ prob.core.phi_vec)) < 1e-12
    prob.reset_work()
    pool, info = prob.sample_posterior(200, level=0, seed=5, dtype=torch.float64)
    assert pool.shape == (200, 6) and info["stages"] >= 1 and prob.work()["fem_solves_raw"] == 0
    rows = prob.pde_convergence([1, 2, 3], 7)["rows"]
    assert rows[2]["sensor_diff_max"] < rows[0]["sensor_diff_max"]
    base = {"configurations": [{"name": "A", "N": [4, 4]}], "problem": "ns2d"}
    cfg = tm.merge_config(dict(base, ns={"d": 6, "noise_sd": 0.3}))
    assert cfg["dimension"] == 6 and cfg["noise_sd"] == 0.3
    try:
        tm.merge_config(dict(base, ns={"foo": 1}))
        raise AssertionError("accepted unknown ns key")
    except ValueError:
        pass
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64", lr_schedule="cosine", target_batching="global_resample")
    k = dict(tm.MLSMC_KERNEL_DEFAULTS, sweeps=2, ess_sweeps=4)
    a = tm.run_mlsmc_kernel(prob, pool, [60, 30, 20], light, k, "pcn", seed=2)
    b = tm.run_transport_mh(prob, pool, [60, 30, 20], light, dict(tm.TRANSPORT_MH_DEFAULTS, rho="ess"), seed=3)
    for ans in (a, b):
        assert math.isfinite(ans["estimate"]) and ans["fem_element_work"] > 0
    print("Navier-Stokes 2D (exact decay, h^2 convergence, densities, work, pool, figure 4, config, runs): passed")


def check_initial_cache():
    """initial_pool: a cached pool is identical to a fresh one; the key separates seeds."""
    import tempfile
    prob = tm.Darcy2DProblem(seed=1, d=6, sensors=3, noise_sd=0.05, data_level=3)
    with tempfile.TemporaryDirectory() as tmp:
        a, ia = tm.initial_pool(prob, 120, 7, 10**6, tmp)
        b, ib = tm.initial_pool(prob, 120, 7, 10**6, tmp)
        c, _ = tm.initial_pool(prob, 120, 8, 10**6, tmp)
        fresh, _ = tm.initial_pool(prob, 120, 7, 10**6, None)
        assert not ia["from_cache"] and ib["from_cache"]
        assert torch.equal(a, b) and torch.equal(a, fresh) and not torch.equal(a, c)
    print("initial pool cache (identical to fresh, keyed by seed): passed")


def check_coupled_mh():
    """transport_mh.coupled = true: synchronously coupled pCN on (parent, child)."""
    clock = tm.Timers()
    prob = GaussianTiltProblem()
    d, n = 5, 400
    g = torch.Generator().manual_seed(3)
    zs = torch.randn(n, d, generator=g, dtype=torch.float64)
    zd = zs + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64)
    W = tm.fit_whitening(zd, "full", 1e-3)
    ls, ld = prob.log_density_z(zs, 0), prob.log_density_z(zd, 1)
    # 1. each coordinate is pathwise the plain pCN chain driven by the same random numbers
    out = tm.coupled_pcn_mh(prob, zs, ls, 0, zd, ld, 1, 0.7, 4, W, clock,
                            generator=torch.Generator().manual_seed(9))
    src_alone = tm.pcn_mh(prob, zs, ls, 0, 0.7, 4, W, clock, generator=torch.Generator().manual_seed(9))
    dst_alone = tm.pcn_mh(prob, zd, ld, 1, 0.7, 4, W, clock, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(out[0], src_alone[0], rtol=0, atol=1e-12)
    torch.testing.assert_close(out[2], dst_alone[0], rtol=0, atol=1e-12)
    diag = out[4]
    for k in ("mh_acceptance_fraction", "mh_source_acceptance_fraction", "mh_accept_disagreement_fraction",
              "mh_both_accept_fraction"):
        assert 0 <= diag[k] <= 1, (k, diag[k])
    # 2. identical starts and targets stay identical
    same = tm.coupled_pcn_mh(prob, zs, ls, 0, zs.clone(), ls.clone(), 0, 0.5, 5, W, clock,
                             generator=torch.Generator().manual_seed(1))
    assert torch.equal(same[0], same[2]) and same[4]["mh_accept_disagreement_fraction"] == 0
    # 3. the coupling keeps pairs much closer than driving the two chains with independent noise
    #    (the pair distance itself need not shrink: pairs separate whenever the accept decisions differ)
    dst_indep = tm.pcn_mh(prob, zd, ld, 1, 0.7, 4, W, clock, generator=torch.Generator().manual_seed(10))
    msd_indep = float((tm.whiten(dst_indep[0], W) - tm.whiten(src_alone[0], W)).square().sum(-1).mean())
    assert diag["pair_msd_whitened_after_mh"] < 0.5 * msd_indep, (diag, msd_indep)
    # 4. validation and accounting in a full run
    base = {"configurations": [{"name": "A", "N": [4, 4]}]}
    assert tm.merge_config(dict(base, transport_mh={"coupled": True}))["transport_mh"]["coupled"] is True
    try:
        tm.merge_config(dict(base, transport_mh={"coupled": "yes"}))
        raise AssertionError("accepted coupled='yes'")
    except ValueError:
        pass
    problem = tm.MeasuredBeskos(dimension=50, y=[26.0, 40.0])
    g = torch.Generator().manual_seed(11)
    initial = torch.randn(20, 50, generator=g, dtype=torch.float64) * 0.1
    light = dict(tm.DEFAULTS["light"], iterations=5, batch_size=16, K=3, whiten="full", init="identity",
                 dtype="float64")
    N = [20, 14, 10]
    ans = tm.run_transport_mh(problem, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=2, coupled=True), seed=6)
    # level l-1 -> l: N[l-1] ratio solves at l (and at l-1 for the first transition), N[l] child density,
    # 2 N[l] child MH solves at l and 2 N[l] parent MH solves at l-1
    assert ans["fem_solves_by_level"] == {"0": 20 + 2 * 14, "1": 20 + 3 * 14 + 2 * 10, "2": 14 + 3 * 10}, \
        ans["fem_solves_by_level"]
    assert ans["continuum_forward_evals"] == sum(N) + N[1] + N[2]   # phi of the moved parents
    assert math.isfinite(ans["estimate"])
    for stage in ans["diagnostics"]:
        assert stage["importance"]["mh_coupled"] is True and stage["importance"]["mh_kernel"] == "pcn_coupled"
    # coupled=false is unchanged: same estimate as the default dictionary
    a = tm.run_transport_mh(problem, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=2), seed=6)
    b = tm.run_transport_mh(problem, initial, N, light, dict(tm.TRANSPORT_MH_DEFAULTS, sweeps=2, coupled=False), seed=6)
    assert a["estimate"] == b["estimate"]
    print("TransportMH coupled pCN (pathwise marginals, identity, contraction, validation, accounting): passed")


def main():
    torch.set_num_threads(1)
    check_kernel_pairs()
    check_noise_option()
    check_is_reference()
    check_source_problem()
    check_mesh_sensitive_source()
    check_lr_schedule()
    check_ess_rho()
    check_iterations_by_level()
    check_mlsmc_ess_sweeps()
    check_darcy2d()
    check_ns2d()
    check_initial_cache()
    check_mh_targets()
    check_runs()
    check_coupled_mh()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
