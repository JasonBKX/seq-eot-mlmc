# Sequential entropic optimal transport for multilevel Monte Carlo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24361732.svg)](https://doi.org/10.5281/zenodo.24361732)

Code and saved results for

> J. Beh, P. Del Moral, A. Jasra. *Sequential entropic optimal transport for Bayesian inverse
> problems via multilevel Monte Carlo*, 2026.

The method couples consecutive levels of a multilevel Monte Carlo estimator with entropic
optimal transport maps learned as light Schrödinger bridges (LightSB). It then corrects each
transported sample with pCN Metropolis–Hastings sweeps that target the finer posterior and
keep the pairing (**TransportMH**). The comparison methods are multilevel SMC with pCN
mutation (MLSMC-pCN), with and without ESS-adaptive sweeps, and LightSB without correction.

## Requirements

Python 3.11 with

```
pip install -r requirements.txt
```

`requirements.txt` lists the versions used for the paper. Everything runs on the CPU, single-threaded.

## Layout

| path | contents |
|---|---|
| `beskos_transport_mh.py` | driver: all methods, timing and PDE-solve accounting, output files |
| `beskos_elliptic.py` | test case 1: 1D elliptic problem (Beskos et al., 2017), 50 parameters |
| `darcy2d.py` | test case 2: 2D Darcy flow, P1 finite elements, 25 parameters |
| `ns2d.py` | test case 3: 2D Navier–Stokes (vorticity form on the torus), 36 parameters |
| `make_reference_from_runs.py` | reference value from independent large-sample runs (test cases 2 and 3) |
| `make_beskos_allocation.py` | per-level sample sizes following Beskos et al. (2017), Sec. 3.3 |
| `test_transport_mh.py` | correctness checks: MH invariance, solve counts, solvers, I/O |
| `configs/` | the configuration file of every run in the paper, plus three quick smoke tests |
| `analysis/make_paper_figures.py` | figures of the paper from the saved runs |
| `analysis/make_paper_tables.py` | tables of the paper from the saved runs |
| `results/` | saved runs (`runs.csv`, `metadata.json`, `reference.json`) and reference values |

## Reproducing the figures and tables from the saved results (seconds)

```
python analysis/make_paper_tables.py results
python analysis/make_paper_figures.py results figures --tag paper
```

The first command prints three tables at the finest horizons:
- relative RMSE, relative bias and wall time;
- the wall time needed to reach each RMSE target;
- the cost breakdown.

The second command writes the figures and `figures/table_*.csv`. The paper's figures are
`figures/fig_siamgrid_error_vs_time.pdf` and `figures/fig_siamgrid_level_sd.pdf`.

## Re-running the experiments

Run all commands from the repository root. New runs are written to the current directory. To
score them against the saved references, use the `results/ref_*` folders.

Quick check (a few minutes):

```
python test_transport_mh.py
python beskos_transport_mh.py --config configs/config_compare_smoke.json --out results_smoke
```

**Test case 1 (1D elliptic).** The reference is self-normalized importance sampling on the
continuum model with 2×10⁸ samples:

```
python beskos_transport_mh.py --config configs/config_reference_is_noise0.05.json --reference-only --out ref_is_noise005
python beskos_transport_mh.py --config configs/config_paper_T1_noise005.json            --reference-file ref_is_noise005/reference.json --out results_paper_T1_noise005
python beskos_transport_mh.py --config configs/config_paper_T1_noise005_pcn_plain.json  --reference-file ref_is_noise005/reference.json --out results_paper_T1_noise005_pcn_plain
python beskos_transport_mh.py --config configs/config_paper_T1_noise005_coupled.json    --reference-file ref_is_noise005/reference.json --out results_paper_T1_noise005_coupled
```

The last run is the coupled variant of TransportMH discussed in the paper (a remark only; not in the figures).

**Test case 2 (Darcy flow).** The reference comes from 40 independent runs of MLSMC-pCN with ESS
sweeps, at ten times the L = 5 allocation:

```
python beskos_transport_mh.py --config configs/config_darcy2d_reference.json --out results_darcy_ref
python make_reference_from_runs.py results_darcy_ref REF MLSMC_pCN ref_darcy
python beskos_transport_mh.py --config configs/config_paper_darcy_L5.json           --reference-file ref_darcy/reference.json --out results_paper_darcy_L5
python beskos_transport_mh.py --config configs/config_paper_darcy_L6.json           --reference-file ref_darcy/reference.json --out results_paper_darcy_L6
python beskos_transport_mh.py --config configs/config_paper_darcy_L5_pcn_plain.json --reference-file ref_darcy/reference.json --out results_paper_darcy_L5_pcn_plain
python beskos_transport_mh.py --config configs/config_paper_darcy_L6_pcn_plain.json --reference-file ref_darcy/reference.json --out results_paper_darcy_L6_pcn_plain
```

**Test case 3 (Navier–Stokes).** The reference comes from 20 independent runs of MLSMC-pCN with
ESS sweeps:

```
python beskos_transport_mh.py --config configs/config_ns2d_reference.json --out results_ns2d_reference
python make_reference_from_runs.py results_ns2d_reference REF MLSMC_pCN ref_ns
python beskos_transport_mh.py --config configs/config_ns2d_L4.json           --reference-file ref_ns/reference.json --out results_ns2d_L4
python beskos_transport_mh.py --config configs/config_ns2d_L4_pcn_plain.json --reference-file ref_ns/reference.json --out results_ns2d_L4_pcn_plain
```

Each configuration runs 30 independent repeats of every method at every horizon L. The paper's
runs took about 7 hours of sampler time on one core of an Apple-silicon laptop:
- test case 1: about 0.7 h;
- test case 2: about 2.5 h;
- test case 3: about 3.3 h.

This excludes the level-0 pools and the references. The level-0 pool is shared by all methods
and repeats of a test case. It is generated on the first run and cached in the folder named by
`initial_cache` in the configuration (`pool_cache_*`, several hundred MB, not included). Wall
times depend on the machine; errors and solve counts do not, since the seeds are fixed in the
configuration files.

## Output

`runs.csv` has one row per (horizon, repeat, method). Its main columns are:
- `estimate`, and the level contributions `contributions`;
- `total_seconds`, the wall time excluding the shared level-0 initialization;
- `pde_seconds`, with the part inside MCMC in `pde_mcmc_seconds`;
- `fit_seconds`, the LightSB fits;
- `fem_solves_by_level` and `fem_element_work`.

`metadata.json` records the full configuration, the observed data and the software versions.

## How to cite

Please cite the paper and, for the code, the Zenodo archive:

> J. Beh, P. Del Moral, A. Jasra. Code for "Sequential entropic optimal transport for Bayesian
> inverse problems via multilevel Monte Carlo" [software], Zenodo, 2026.
> https://doi.org/10.5281/zenodo.24361732

## License

MIT; see `LICENSE`.
