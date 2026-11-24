# UPM modelling 2026

Python library for course exercises on **viral evolutionary dynamics**:

1. **SIR + deleterious mutations** — multi-class load, deterministic ODE and tau-leap stochastic versions
2. **SIR + life-history + antigen evolution** — resident/mutant types with partial cross-immunity
3. **Phylodynamic visualization** — MutAntiGen tree / timeseries readers and plots (antigenic type, Net R)

The notebook `03.viral-dynamics.ipynb` remains a worked example; the installable package is under `src/upm_modelling_2026/`.

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) recommended (also works with `pip`)

## Install

From the repository root:

```bash
# with uv (creates .venv and installs the package editable)
uv sync

# optional: Jupyter for the notebook
uv sync --extra dev

# or with pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

After install you get the `upm-modelling` CLI and several shortcuts (see below).

## Package layout

```text
src/upm_modelling_2026/
  sir/                 # evolutionary SIR models
    deleterious.py     # ODE + mutation-selection equilibrium
    stochastic.py      # tau-leap
    antigen.py         # life-history compartments + two antigenic types
  phylodynamics/       # MutAntiGen I/O + plots
    tree_io.py
    plot_antigenic.py
    plot_netR.py
    timeseries.py
  cli.py               # command-line entry points
```

Root scripts `tree_io.py`, `plot_tree_antigenic.py`, and `plot_tree_netR.py` are thin wrappers around the package (same CLI flags as before).

## Library usage

### 1. Deleterious-mutation SIR

```python
from upm_modelling_2026.sir import simulate_deleterious, compute_equilibrium_population

res = simulate_deleterious(k_max=20, beta_0=0.5625, s_d=0.08, lamb=0.10, N=1000, tfinal=100)
t, y = res.t, res.y   # y[0]=S, y[1:-1]=I_k, y[-1]=R

I_tot, p, k = compute_equilibrium_population(30, 0.565, 0.008, 0.10, 1/4, 0.0, 4e9)
```

Stochastic tau-leap:

```python
import numpy as np
from upm_modelling_2026.sir import prepare_population, model_base_rate_fun, integrate_tau_leap

y0 = prepare_population(5, 1, N=100)
ts, Y = integrate_tau_leap(
    model_base_rate_fun, y0, t_end=100, tau=1.0,
    beta_0=0.565, s_d=0.08, lamb=0.1, mu=1/(33*365), gamma=0.25, N=100,
    rng=np.random.default_rng(32),
)
```

### 2. Antigen evolution + life history

State layout with `K` load classes:

`[S0, I0r x K, I0m x K, Imr x K, Irm x K, Sr, Sm, Srm]`

```python
from upm_modelling_2026.sir import mutant_initial_state, simulate_antigen, unpack_antigen_state

K = 45
y0 = mutant_initial_state(K, i_bg=3, count=1, beta_0=0.5625, s_d=0.008,
                          lamb=0.10, mu=1/(30*365), gamma=0.25, N=4e9)
res = simulate_antigen(y0, sigma=0.045, tfinal=1500)
parts = unpack_antigen_state(res.y, res.K)
```

### 3. Phylodynamic plots (MutAntiGen outputs)

```python
from upm_modelling_2026.phylodynamics import (
    plot_antigenic_tree,
    plot_tree_netR,
    plot_timeseries_columns,
    plot_mutation_series,
)

plot_antigenic_tree("out.simmapAntigenic", "out.tips", "tree_antigenic.png")
plot_tree_netR("out.branches", "out.viralFitnessSeries", "tree_netR.png")
plot_timeseries_columns("out.timeseries", ["totalCases", "tmrca"], "cases.png", logy=True)
plot_mutation_series("out.mutationSeries", "mutations.png")
```

Expected simulation side-car files (from MutAntiGen / similar):

| File | Role |
|------|------|
| `out.simmapAntigenic` | SIMMAP Newick with antigenic states |
| `out.tips` / `out.branches` | tip geometry and edges |
| `out.viralFitnessSeries` | `date`, `meanR` (= E[log R]) |
| `out.timeseries` | population / diversity series |
| `out.mutationSeries` | load-class counts over time |

## CLI

```bash
upm-modelling --help

# ODE deleterious SIR -> PNG
upm-modelling sir-deleterious --sd 0.08 --lamb 0.10 --out sir_del.png
# shortcut:
upm-sir-deleterious --sd 0.08 --out sir_del.png

# stochastic tau-leap
upm-modelling sir-stochastic --N 100 --gamma 0.25 --seed 32 --out sir_stoch.png

# antigen + life history (large K is slower)
upm-modelling sir-antigen --sigma 0.045 --i-bg 3 --tfinal 1500 --out antigen.png
upm-sir-antigen --sigma 0.045 --out antigen.png

# phylogenies (run in a directory with MutAntiGen outputs)
upm-modelling plot-tree-antigenic --simmap out.simmapAntigenic --tips out.tips --out tree_ag.png
upm-modelling plot-tree-netr --branches out.branches --fitness out.viralFitnessSeries --out tree_R.png
upm-plot-tree-antigenic --simmap out.simmapAntigenic --out tree_ag.png
upm-plot-tree-netr --branches out.branches --fitness out.viralFitnessSeries --out tree_R.png

upm-modelling plot-timeseries --timeseries out.timeseries --columns totalCases tmrca --logy
upm-modelling plot-mutation-series --series out.mutationSeries --n-classes 10
```

Root wrappers (same flags as the package CLIs):

```bash
python plot_tree_antigenic.py --simmap out.simmapAntigenic --out tree_ag.png
python plot_tree_netR.py --branches out.branches --fitness out.viralFitnessSeries --out tree_R.png
```

## Model notes (short)

**Deleterious load.** On transmission, j ~ Poisson(lambda) mutations are added. Transmission of class k is beta_k = beta_0 (1 - s_d)^k. At mutation-selection balance the class frequencies are approximately Poisson(lambda / s_d).

**Antigen model.** Two antigenic types (resident / mutant) share the same load structure. Hosts remember prior type (Sr, Sm, Srm). Heterologous infection is scaled by sigma (escape / residual susceptibility).

**Net R.** Plots use Net R(t) = exp(meanR(t)) from the fitness series (population mean of log R).

## Notebook

```bash
uv sync --extra dev
uv run jupyter lab 03.viral-dynamics.ipynb
```

Prefer importing from `upm_modelling_2026` instead of copying notebook cells into new scripts.

## License / course

Course material for UPM modelling 2026 (Bruno Cuevas). Adjust as needed for your institution.
