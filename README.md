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
  - DIC curves.

### Intended workflow

1. Fit overfitted mixture model (frequentist or Bayesian)
2. Extract mixing measure
3. Build dendrogram
4. Compute DIC along tree
5. Select $ k $
6. Visualize hierarchical structure

---

## Bayesian extension

Instead of using a single MLE mixing measure:

- Fit a Bayesian mixture model with large $ k $
- Sample posterior mixing measures
- Apply the dendrogram algorithm to posterior draws
- Obtain:
  - distribution over selected $ k $,
  - posterior uncertainty over hierarchies.

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

```bash
python examples/demo_synthetic.py
```

This script demonstrates the full workflow:

- simulate data from a finite Gaussian mixture,
- fit an overfitted mixture model,
- extract the estimated mixing measure,
- construct the dendrogram of mixing measures,
- compute DIC along the dendrogram,
- report a suggested number of components,
- visualize the dendrogram and DIC curve.

The demo is designed to run in under 30 minutes on a standard laptop.

---

## Repository Structure

```
STATS607_Project04/
├── src/dmm/
│   ├── mixing_measure.py   # representation of a discrete mixing measure
│   ├── dendrogram.py       # Algorithms 1–2 (merging + dendrogram)
│   ├── dic.py              # likelihood + DIC utilities
│   └── __init__.py
├── examples/
│   └── demo_synthetic.py   # self-contained example script
├── notebooks/              # exploratory analysis
├── tests/                  # (optional) unit tests
├── requirements.txt
├── README.md
└── report-PHO.pdf          # final project report
```

---

## Course Concepts Used

This project uses tools and ideas from STAT 607, including:

- version control with Git and GitHub,
- modular software design in Python,
- numerical linear algebra (covariance operations),
- likelihood-based model comparison,
- reproducible research workflows,
- simulation-based validation.

---

## Lessons Learned (to be updated at submission)

Planned reflection points:

- challenges in implementing numerically stable merging rules,
- computational complexity of quadratic pairwise distance evaluation,
- careful handling of covariance structure in Gaussian merging,
- differences between frequentist and Bayesian overfitting behavior,
- tradeoffs between model flexibility and interpretability.

---

## Citation

If you use or reference this repository, please cite:

Dat Do, Linh Do, Scott McKinley, Jonathan Terhorst, XuanLong Nguyen (2024).  
*Dendrogram of mixing measures: Hierarchical clustering and model selection for finite mixture models.*