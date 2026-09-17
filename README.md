# ATS-ILU: Code and Data

**A Framework for Parallel Incomplete LU Factorizations Based on Pattern-Norm
Minimization**

Marc A. Tunnell and Erik G. Boman

This repository contains the C/OpenMP implementations, recorded measurements,
settings and scripts needed to reproduce the paper's numerical figures and
tables. It also supports new experiments on downloaded matrices.

## Reproduce Figures and Tables

Requirements: Python 3.12 and Julia 1.12.4. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
python reproduce.py
```

`+1.12` is a juliaup selector. With standalone Julia, use that executable for
installation and pass `--julia /path/to/julia` to `reproduce.py`.
Dependencies are pinned in `requirements.txt`, `Project.toml` and `Manifest.toml`.
No matrix downloads or numerical reruns are required for this workflow.

Outputs appear in `generated/figures/` (PDF/SVG/PNG) and `generated/tables/`
(CSV and LaTeX tabular fragments requiring `booktabs`). Intermediate calculations
are in `generated/work/`; recorded inputs are unchanged. `--data-only` generates
the accounting and contraction tables without Julia. The marginal-sweep table
requires the full workflow because it depends on the Pareto calculation.

| Paper result | Generator |
|---|---|
| Solver quality relative to sequential ILU/IC | `plotting/plot_combined_solver_quality_profiles.jl` |
| Objective versus solver-quality case studies | `plotting/plot_combined_objective_solver_case_studies.jl` |
| Strong scaling, median and quantile bands | `results/publication_timing_2026-08-04/figure_01_scaling/plot.jl` |
| Speedup versus factor-pattern width | `results/publication_timing_2026-08-04/figure_02_work_scaling/plot.jl` |
| Construction-cost/quality Pareto comparison | `results/publication_timing_2026-08-04/figure_04_pareto/plot.jl` |
| Experiment accounting and objective contraction | `scripts/generate_tables.py` |
| Marginal benefit of additional sweeps | `results/publication_timing_2026-08-04/table_01_marginal_sweeps/generate.jl` |

The driver runs these in dependency order. Shared figure styling is defined
in `plotting/ATSILUPaperStyle.jl`.

## Data and Parameters

These CSVs contain individual observations, not just plotted averages:

| Directory under `results/` | Contents |
|---|---|
| `analysis_2026-07-31/unsymmetric_gmres/` | `combined_solver_outcomes.csv`: 17,680 ILU states, objectives and GMRES outcomes |
| `analysis_2026-08-02/spd_cg/` | `combined_solver_outcomes.csv`: 8,890 IC states, objectives and CG outcomes |
| `analysis_2026-08-03/unsymmetric_timing/` | Selected timing rows, pair coverage and eligibility |
| `analysis_2026-08-03/spd_timing/` | Selected timing rows, pair coverage and eligibility |

Solver CSVs contain per-configuration factorization objectives, Krylov iteration
counts and outcome statuses. Timing CSVs contain numerical and setup measurements
in seconds; accompanying coverage and eligibility files identify the cases
included in each comparison. Timing analysis normalizes measurements by their
recorded sweep counts before aggregating repeats.

Matrix identifiers and download URLs are in `data/suitesparse/`. Experiment
parameters are recorded in `config/experiments.json`. The scripts listed above
implement the filtering and aggregation for each figure and table.

Timing measurements were collected on Perlmutter nodes with two 64-core AMD
EPYC 7763 processors and 512 GB RAM. Solver-quality measurements were collected
separately.

## Build and Test

Requirements: a C11 compiler with OpenMP, GNU Make and the C math library.

```sh
make -C c_ilu -j4 all check
python scripts/small-test.py
```

`c_ilu/src/` contains the numerical kernels, symbolic factorization, preprocessing
and sparse matrix I/O. The tools in `c_ilu/tools/` measure construction, generate
factors and replay GMRES/CG. The test uses small bundled matrices.

### MC64

MC64 is **not bundled**. To reproduce the unsymmetric preprocessing, supply a
compatible external 32-bit `mc64ad_dist` implementation:

```sh
make -C c_ilu clean
make -C c_ilu -j4 MC64_SOURCE=/absolute/path/to/mc64ad_dist.c all check
python scripts/small-test.py --expect-mc64
```

Alternatively, set `MC64_LIBS` to link a library exporting that interface.
Clean before changing linkage. Without MC64, the code warns, skips matching
and scaling, and retains RCM; the CSV records the effective preprocessing.
That fallback does **not** reproduce the unsymmetric experiments. SPD needs no MC64.

## New Experiments

Download a named matrix using its manifest entry:

```sh
python scripts/download_matrix.py Janna/ML_Laplace --dest matrices
```

Large matrices are downloaded separately. The downloader checks dimensions and
records SHA-256 hashes; historical download hashes were not retained.

Run one matrix/fill pair in a new output directory:

```sh
python run_experiment.py unsymmetric quality --matrix matrices/Janna__ML_Laplace/ML_Laplace.mtx --k 0 --threads 2,4,8,16,32,64,128 --out generated/new-quality
python run_experiment.py unsymmetric timing --matrix matrices/Janna__ML_Laplace/ML_Laplace.mtx --k 0 --threads 1,2,4,8,16,32,64,128 --out generated/new-timing
```

Use `spd` for IC/CG and choose `--threads` and `--sync-threads` for your machine.
The runner records commands and execution settings in `run.json`; use `--help`
for options. Large cases can require substantial resources. The selection
cutoff is not an execution timeout, and hardware and asynchronous scheduling
can change timings and outcomes.

Original code is BSD-3-Clause. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
