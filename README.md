# STATS607 Project 04: Dendrogram of Mixing Measures

**Author:** An Pho  
**Course:** STATS 607  
**Date:** December 2025  

This project implements the *dendrogram of mixing measures* algorithm for summarizing
and selecting finite mixture models, based on the method proposed by:

> Do, D., Do, L., McKinley, S., Terhorst, J., & Nguyen, X. (2024).  
> *Dendrogram of mixing measures: Hierarchical clustering and model selection 
> for finite mixture models.*

---

## Overview

The goal of this project is to build reusable software for constructing
**dendrograms of mixing measures** in finite mixture models and to explore their use
for:

- visualizing hierarchical structure among mixture components,
- selecting the number of components,
- extending the approach to Bayesian mixture models.

Traditional mixture modeling and hierarchical clustering address clustering in
different ways:
- Mixture models are model-based but sensitive to overfitting.
- Hierarchical clustering is flexible and visual, but model-free.

This project bridges these two paradigms by constructing a dendrogram **from the
mixture components themselves**, rather than from the data.

---

## Background

### Finite mixture model

We assume data arise from:  

$ x_1, \dots, x_n \sim p_{G_0}(x) = \int f(x \mid \theta) \, dG_0(\theta) = \sum_{i=1}^{k_0} p_i^0 f(x \mid \theta_i^0), $

where:
- $ f(x \mid \theta) $ is a known kernel (e.g. Gaussian),
- $ G_0 = \sum p_i^0 \delta_{\theta_i^0} $ is the *mixing measure*,
- the true number of components $ k_0 $ is unknown.

In practice, an **overfitted model** with $ k > k_0 $ components is estimated
frequentistically (MLE) or via Bayesian inference to obtain an estimated mixing
measure $ G $.

---

## Dendrogram of Mixing Measures

Instead of clustering data points, we cluster the **atoms of the mixing measure**:
weights and parameters.

### High-level algorithm

Given an overfitted mixing measure

$ G^{(k)} = \sum_{i=1}^{k} p_i \delta_{\theta_i}, $

we construct a dendrogram as follows:

1. Start with all $ k $ atoms as leaves.
2. Repeatedly:
   - find the closest pair of atoms under a merging dissimilarity,
   - merge them into a new atom,
   - record the merge height.
3. Continue until a single atom remains.

This produces:
- a full dendrogram over mixture components,
- a sequence of reduced measures $ G^{(k)}, G^{(k-1)}, \dots, G^{(1)} $.

---

## Merging rule

For two atoms $ p_i \delta_{\theta_i} $ and $ p_j \delta_{\theta_j} $, define:

$ d(p_i \delta_{\theta_i}, p_j \delta_{\theta_j}) = \frac{\|\theta_i - \theta_j\|^2}{p_i^{-1} + p_j^{-1}}. $

This encourages merging:
- components close in parameter space,
- components with small weights.

When merging:

$ p^* = p_i + p_j, \quad \theta^* = \frac{p_i \theta_i + p_j \theta_j}{p^*}. $

For Gaussian mixtures, covariance matrices are updated using a moment-preserving
formula that accounts for within- and between-cluster variance.

---

## Algorithms implemented

### Algorithm 1 — Merging of atoms
- Picks closest pair under $ d(\cdot,\cdot) $
- Merges two atoms
- Records merge height

### Algorithm 2 — Dendrogram Inferred Clustering (DIC)
- Iterates Algorithm 1 from $ k \to 1 $
- Returns:
  - all intermediate mixing measures,
  - all dendrogram heights.

---

## Model Selection: DIC

The paper introduces the **Dendrogram Information Criterion (DIC)**:

$ \text{DIC}(\kappa) = - \big( d^{(\kappa)} + \omega_n \ell^{(\kappa)} \big), $

where:
- $ d^{(\kappa)} $ is the dendrogram height at level $ \kappa $,
- $ \ell^{(\kappa)} $ is average log-likelihood,
- $ \omega_n $ grows slowly (e.g. $ \log n $).

The selected number of components is:

$ \hat{k} = \arg\min_\kappa \mathrm{DIC}(\kappa). $

---

## What this project builds

This repository implements:

- `MixingMeasure`: class for representing a discrete mixing measure
- `MixingDendrogram`: performs dendrogram construction (Algorithms 1–2)
- DIC utilities:
  - likelihood computation,
  - automatic selection of $ k $
- Visualization tools for:
  - dendrograms,
  - DIC curves
- Simulation and evaluation scripts for:
  - convergence of DIC-based selection in strongly/weakly identifiable regimes,
  - model selection under well-specified, ε-contamination, and skew-normal mixtures,
  - comparing AIC, BIC, and DIC in both serial and parallel implementations.

### Intended workflow

1. Fit overfitted mixture model (frequentist or Bayesian)
2. Extract mixing measure
3. Build dendrogram
4. Compute DIC along tree
5. Select $ k $
6. Visualize hierarchical structure

---

## Bayesian extension

Instead of using a single MLE mixing measure, one can:

- fit a Bayesian mixture model with a large number of components,
- sample posterior mixing measures,
- apply the dendrogram algorithm to posterior draws, and
- obtain a distribution over selected $ k $ and over hierarchical structures.

A full Bayesian implementation is left as future work; this project focuses primarily on the frequentist (MLE-based) setting but is designed so that posterior mixing measures could be plugged in with minimal changes.

---

## Installation

```bash
git clone https://github.com/anpho26/STATS607_Project04
cd STATS607_Project04

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick Demo

After installation and activating the virtual environment, the main demo for this project is the Makefile target:

```bash
make figs
```

This single command orchestrates the full workflow:

- runs the synthetic dendrogram + DIC example,
- runs the convergence experiments (strongly and weakly identifiable regimes),
- runs the model-selection experiments (well-specified, $\epsilon$-contamination, and skew-normal) in both serial and parallel implementations,
- saves all figures under `out/figures/`.

If you only want the small synthetic example, you can run the lightweight demo either directly:

```bash
python examples/demo_synthetic.py
```

or via the Makefile:

```bash
make figs-demo
```

This script demonstrates the basic workflow on a single finite Gaussian mixture and saves a figure with the dendrogram and DIC curve to:

- `out/figures/demo_synthetic_dendrogram_dic.png`.
---

## Reproducing Experiments

### Convergence analysis

The convergence experiments study how the DIC-based selected number of components behaves across sample sizes in:

- a strongly identifiable regime, and
- a weakly identifiable regime.

They are implemented in:

```bash
python examples/experiment_convergence_strong.py
python examples/experiment_convergence_weak.py
```

Each script saves a figure under `out/figures/`:

- `convergence_strong_all_four.png`
- `convergence_weak_all_four.png`

### Model selection and misspecification

The model-selection experiments compare AIC, BIC, and DIC in three regimes:

- well-specified Gaussian mixture (true $k_0 = 2$),
- ε-contamination of a Gaussian mixture,
- skew-normal mixture.

Serial implementation:

```bash
python examples/experiment_model_selection.py
```

Parallel implementation (joblib):

```bash
python examples/experiment_model_selection_parallel.py
```

For each regime, the scripts produce:

- panel (a): histogram of the data with the true density,
- panel (b): dendrogram of an overfitted Gaussian mixture with 10 atoms,
- panel (c): proportion of correctly selecting $k_0$ as a function of $n$,
- panel (d): average selected number of components as a function of $n$.

Output figures are saved under `out/figures/` with names such as:

- `well_hist_true_density.png`, `well_dendrogram_k10.png`,
- `model_selection_well_specified_k2.png`,
- and analogous files for the ε-contamination and skew-normal cases, as well as their `_parallel` counterparts.

### Profiling serial vs parallel

To compare the runtime of the serial and parallel implementations, use:

```bash
python examples/profile_model_selection.py
```

This script runs a reduced grid of sample sizes and replications, and reports:

- wall-clock time for the serial vs parallel experiments, and
- simple checks that their Monte Carlo estimates are numerically consistent.

---

## Using the Makefile

For convenience, a `Makefile` is provided in the project root. Common targets:

```bash
make install                      # create .venv and install dependencies
make figs                         # run all figure-generating scripts
make figs-demo                    # run the synthetic demo only
make figs-convergence             # run convergence experiments (strong + weak)
make figs-model-selection         # run serial model-selection experiments
make figs-model-selection-parallel  # run parallel model-selection experiments
make profile                      # profile serial vs parallel runtimes
make clean                        # remove generated outputs and caches
```

---

## Repository Structure

```
STATS607_Project04/
├── src/dmm/
│   ├── mixing_measure.py            # representation of a discrete mixing measure
│   ├── dendrogram.py                # Algorithms 1–2 (merging + dendrogram)
│   ├── dic.py                       # likelihood + DIC utilities
│   └── __init__.py
├── examples/
│   ├── demo_synthetic.py            # self-contained demo script
│   ├── experiment_convergence_strong.py  # convergence in strongly identifiable regime
│   ├── experiment_convergence_weak.py    # convergence in weakly identifiable regime
│   ├── experiment_model_selection.py     # serial AIC/BIC/DIC experiments
│   ├── experiment_model_selection_parallel.py  # parallel AIC/BIC/DIC experiments
│   └── profile_model_selection.py         # serial vs parallel profiling
├── out/
│   └── figures/                     # generated figures (not tracked by git)
├── Makefile
├── requirements.txt
├── README.md
└── report-PHO.pdf                   # final project report
```
