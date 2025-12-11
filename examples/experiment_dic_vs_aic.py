from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

import sys
from pathlib import Path

# Make src/ visible so we can import dmm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dmm import (
    MixingMeasure,
    MixingDendrogram,
    dic_curve,
    select_k_dic,
)


def simulate_paper_gaussian(
    n: int = 300,
    setting: str = "strong",  # "strong" or "weak"
    random_state: int | None = None,
) -> np.ndarray:
    """
    Simulate data from the 3-component 2D Gaussian mixture used in the paper.

    Means:
        (2, 1), (0, 6), (-2, 1)
    Equal weights: 1/3 each.

    setting = "strong": identity covariances (strongly identifiable).
    setting = "weak"  : location-scale covariances (weakly identifiable).
    """
    rng = np.random.default_rng(random_state)

    means = np.array([
        [2.0, 1.0],
        [0.0, 6.0],
        [-2.0, 1.0],
    ])
    pis = np.array([1/3, 1/3, 1/3])

    if setting == "strong":
        covs = np.array([np.eye(2) for _ in range(3)])
    else:
        # Weakly identifiable setting: rough match to paper
        covs = np.array([
            [[0.5,  0.5], [0.5,  0.1]],
            [[0.5, -0.1], [-0.1, 0.1]],
            [[0.25, 0.5], [0.5,  2.0]],
        ])

    # Sample component labels
    zs = rng.choice(3, size=n, p=pis)
    X_blocks = []
    for k in range(3):
        nk = (zs == k).sum()
        if nk > 0:
            X_blocks.append(
                rng.multivariate_normal(mean=means[k], cov=covs[k], size=nk)
            )
    X = np.vstack(X_blocks)
    return X


def run_experiment(
    n: int = 300,
    setting: str = "strong",
    k_max: int = 10,
    n_rep: int = 30,
    random_state: int = 0,
) -> None:
    """
    Compare how often AIC, BIC, and DIC recover k0 = 3 on the paper mixture.
    """
    rng = np.random.default_rng(random_state)
    true_k = 3

    counts_aic = {k: 0 for k in range(1, k_max + 1)}
    counts_bic = {k: 0 for k in range(1, k_max + 1)}
    counts_dic = {k: 0 for k in range(1, k_max + 1)}

    for r in range(n_rep):
        X = simulate_paper_gaussian(
            n=n, setting=setting, random_state=rng.integers(1, 10_000_000)
        )

        # Fit models for k = 1..k_max
        aic_vals = []
        bic_vals = []
        gmms = []

        for k in range(1, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=rng.integers(1, 10_000_000),
                reg_covar=1e-6,
                n_init=5,
            ).fit(X)
            gmms.append(gmm)
            aic_vals.append(gmm.aic(X))
            bic_vals.append(gmm.bic(X))

        ks_grid = np.arange(1, k_max + 1)

        # AIC/BIC selection
        k_hat_aic = int(ks_grid[np.argmin(aic_vals)])
        k_hat_bic = int(ks_grid[np.argmin(bic_vals)])
        counts_aic[k_hat_aic] += 1
        counts_bic[k_hat_bic] += 1

        # DIC: use the overfitted model with k_max components -> dendrogram -> DIC
        gmm_over = gmms[-1]  # k = k_max
        mm_over = MixingMeasure.from_sklearn_gmm(gmm_over)
        dend = MixingDendrogram(mm_over).build()
        ks_dic, dic_vals = dic_curve(X, dend)  # omega = log n by default
        k_hat_dic = select_k_dic(ks_dic, dic_vals)
        counts_dic[k_hat_dic] += 1

        print(
            f"rep {r+1}/{n_rep}: "
            f"k_hat (AIC,BIC,DIC) = ({k_hat_aic}, {k_hat_bic}, {k_hat_dic})"
        )

    print("\n=== Summary over", n_rep, "replications ===")
    print(f"True k0 = {true_k}, k_max = {k_max}, n = {n}, setting = {setting}\n")

    def fmt_counts(label, counts):
        print(label)
        for k in range(1, k_max + 1):
            if counts[k] > 0:
                print(f"  k = {k}: {counts[k]} times")
        print()

    fmt_counts("AIC selections:", counts_aic)
    fmt_counts("BIC selections:", counts_bic)
    fmt_counts("DIC selections:", counts_dic)


def main() -> None:
    run_experiment(
        n=300,
        setting="strong",  # or "weak"
        k_max=10,
        n_rep=30,
        random_state=0,
    )


if __name__ == "__main__":
    main()