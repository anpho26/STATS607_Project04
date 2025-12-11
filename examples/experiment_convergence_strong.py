from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

# ---------------------------------------------------------------------
# Make src/ visible so we can import dmm
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dmm import MixingMeasure, MixingDendrogram  # type: ignore


# ---------------------------------------------------------------------
# True mixing measure (strongly identifiable setting)
# ---------------------------------------------------------------------
def true_mixing_measure_strong() -> MixingMeasure:
    """
    True G0 for the strongly identifiable setting in the paper:
    - 3 components in R^2
    - equal weights (1/3, 1/3, 1/3)
    - means: (2,1), (0,6), (-2,1)
    - covariances: identity (known)
    """
    weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    means = np.array(
        [
            [2.0, 1.0],
            [0.0, 6.0],
            [-2.0, 1.0],
        ],
        dtype=float,
    )
    covs = np.array([np.eye(2) for _ in range(3)], dtype=float)
    return MixingMeasure.from_arrays(weights, means, covs)


def simulate_from_G0_strong(
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw n samples from G0 in the strong setting."""
    if rng is None:
        rng = np.random.default_rng()

    G0 = true_mixing_measure_strong()
    weights = G0.weights
    means = G0.means
    covs = G0.covs
    assert covs is not None

    k = G0.n_components
    zs = rng.choice(k, size=n, p=weights)

    X_blocks = []
    for j in range(k):
        nj = (zs == j).sum()
        if nj > 0:
            X_blocks.append(
                rng.multivariate_normal(mean=means[j], cov=covs[j], size=nj)
            )
    X = np.vstack(X_blocks)
    return X


# ---------------------------------------------------------------------
# Wasserstein distance between discrete mixing measures
# ---------------------------------------------------------------------
def _param_vectors_strong(mm: MixingMeasure) -> np.ndarray:
    """
    For the strong setting, we only use means (location model) as parameters.
    """
    return mm.means.astype(float)


def wasserstein_p_mixing(
    mm_true: MixingMeasure,
    mm_est: MixingMeasure,
    p: float,
) -> float:
    """
    Compute p-Wasserstein distance between two discrete mixing measures
    on parameter space (here: means only).

    Uses an optimal transport linear program:
      minimize <C, P> s.t. P 1 = w_true, P^T 1 = w_est, P >= 0.
    """
    w1 = mm_true.weights
    w2 = mm_est.weights
    theta1 = _param_vectors_strong(mm_true)
    theta2 = _param_vectors_strong(mm_est)

    m, d1 = theta1.shape
    n, d2 = theta2.shape
    assert d1 == d2

    # cost matrix: ||theta_i - theta'_j||^p
    C = cdist(theta1, theta2, metric="euclidean") ** p  # (m, n)

    c = C.ravel()  # objective coefficients

    # Equality constraints: rows & columns
    A_eq = []
    b_eq = []

    # Row sums
    for i in range(m):
        row = np.zeros(m * n)
        row[i * n : (i + 1) * n] = 1.0
        A_eq.append(row)
        b_eq.append(w1[i])

    # Column sums
    for j in range(n):
        col = np.zeros(m * n)
        col[j::n] = 1.0
        A_eq.append(col)
        b_eq.append(w2[j])

    A_eq = np.asarray(A_eq)
    b_eq = np.asarray(b_eq)

    bounds = [(0.0, None)] * (m * n)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    cost = res.fun  # this is W_p^p
    return cost ** (1.0 / p)


# ---------------------------------------------------------------------
# Main experiment: all four curves for strong setting
# ---------------------------------------------------------------------
def run_experiment_strong_all_four(
    n_rep: int = 32,
) -> Tuple[np.ndarray, dict]:
    """
    Run convergence experiment in the strongly identifiable setting for
    all four combinations from Figure 2(a):

      1. log W1(G_n^e, G0)   (exact-fitted, p=1)
      2. log W2(G_n^o, G0)   (overfitted,  p=2)
      3. log W1(G_n^o, G0)   (overfitted,  p=1)
      4. log W1(G_n^m, G0)   (merged,     p=1)

    Returns
    -------
    logn : np.ndarray, shape (T,)
        log sample sizes.
    log_errors : dict[str, np.ndarray]
        For each key, a matrix of shape (T, n_rep) with log-errors.
    """
    rng = np.random.default_rng(0)
    G0 = true_mixing_measure_strong()

    # n from 10^2 to 10^4
    logn = np.linspace(2.0, 4.0, 10)
    n_grid = np.round(np.exp(logn)).astype(int)  # natural log version

    # Store log-errors for each combination
    combos = {
        "W1_exact": np.zeros((len(n_grid), n_rep)),
        "W2_over": np.zeros((len(n_grid), n_rep)),
        "W1_over": np.zeros((len(n_grid), n_rep)),
        "W1_merged": np.zeros((len(n_grid), n_rep)),
    }

    for t, n in enumerate(n_grid):
        for r in range(n_rep):
            X = simulate_from_G0_strong(n=n, rng=rng)

            # Exact-fitted MLE (k=3)
            gmm_exact = GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=rng.integers(1, 10_000_000),
                reg_covar=1e-6,
                n_init=3,
            ).fit(X)
            mm_exact = MixingMeasure.from_sklearn_gmm(gmm_exact)

            # Overfitted MLE (k=5)
            gmm_over = GaussianMixture(
                n_components=5,
                covariance_type="full",
                random_state=rng.integers(1, 10_000_000),
                reg_covar=1e-6,
                n_init=3,
            ).fit(X)
            mm_over = MixingMeasure.from_sklearn_gmm(gmm_over)

            # Merged from dendrogram at k=3
            dend = MixingDendrogram(mm_over).build()
            mm_merged = dend.get_measure_at_k(3)

            # Distances
            e_W1_exact = wasserstein_p_mixing(G0, mm_exact, p=1.0)
            e_W2_over = wasserstein_p_mixing(G0, mm_over, p=2.0)
            e_W1_over = wasserstein_p_mixing(G0, mm_over, p=1.0)
            e_W1_merged = wasserstein_p_mixing(G0, mm_merged, p=1.0)

            combos["W1_exact"][t, r] = np.log(e_W1_exact)
            combos["W2_over"][t, r] = np.log(e_W2_over)
            combos["W1_over"][t, r] = np.log(e_W1_over)
            combos["W1_merged"][t, r] = np.log(e_W1_merged)

        print(
            f"n = {n:5d} | "
            f"log W1_exact: {combos['W1_exact'][t].mean():.3f}, "
            f"log W2_over:  {combos['W2_over'][t].mean():.3f}, "
            f"log W1_over:  {combos['W1_over'][t].mean():.3f}, "
            f"log W1_merged:{combos['W1_merged'][t].mean():.3f}"
        )

    return logn, combos


def main() -> None:
    logn, log_errors = run_experiment_strong_all_four(n_rep=32)

    # Compute means and std devs across replications
    stats = {}
    for key, mat in log_errors.items():
        mean = mat.mean(axis=1)
        std = mat.std(axis=1)
        stats[key] = (mean, std)

    # Fit regression lines: log(error) ~ a * log(n) + b
    fits = {}
    for key, mat in log_errors.items():
        # Flatten all reps
        x = np.tile(logn, mat.shape[1])  # repeat logn for each rep
        y = mat.T.ravel()               # shape (T * n_rep,)
        a, b = np.polyfit(x, y, 1)
        fits[key] = (a, b)
        print(f"{key}: log(error) ≈ {a:.2f} log(n) + {b:.2f}")

    # Plot
    plt.figure(figsize=(6, 4))

    colors = {
        "W1_exact": "tab:red",
        "W2_over": "tab:blue",
        "W1_over": "tab:purple",
        "W1_merged": "tab:gray",
    }
    markers = {
        "W1_exact": "o",
        "W2_over": "s",
        "W1_over": "D",
        "W1_merged": "^",
    }
    linestyles = {
        "W1_exact": "--",
        "W2_over": "-.",
        "W1_over": ":",
        "W1_merged": "-",
    }

    labels = {
        "W1_exact": "log(W1(G_n^e, G0))",
        "W2_over": "log(W2(G_n^o, G0))",
        "W1_over": "log(W1(G_n^o, G0))",
        "W1_merged": "log(W1(G_n^m, G0))",
    }

    for key in ["W1_exact", "W2_over", "W1_over", "W1_merged"]:
        mean, std = stats[key]
        a, b = fits[key]
        c = colors[key]

        # Optional: tiny horizontal jitter so bars don't sit exactly on top of each other
        jitter_map = {
            "W1_exact": -0.015,
            "W2_over": -0.005,
            "W1_over": 0.005,
            "W1_merged": 0.015,
        }
        x = logn + jitter_map[key]

        # Light, thin error bars without markers
        plt.errorbar(
            x,
            mean,
            yerr=std,
            fmt="none",
            ecolor=c,
            elinewidth=0.8,
            capsize=3,
            alpha=0.4,
            zorder=1,
        )

        # Solid markers for the means
        plt.scatter(
            x,
            mean,
            marker=markers[key],
            s=30,
            color=c,
            edgecolor="white",
            linewidths=0.7,
            zorder=3,
        )

        # Regression line (no jitter, so it stays on the true logn grid)
        y_fit = a * logn + b
        plt.plot(
            logn,
            y_fit,
            linestyle=linestyles[key],
            color=c,
            linewidth=1.8,
            label=f"{labels[key]} = {a:.2f} log(n) + {b:.2f}",
            zorder=2,
        )

    plt.xlabel("log(n)")
    plt.ylabel("log(error)")
    plt.title("(a) Strongly identifiable setting")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()