from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

import sys
from pathlib import Path

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


def simulate_gaussian_mixture(
    n_per: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    """
    Simulate data from a simple 2D Gaussian mixture with 3 components.
    """
    rng = np.random.default_rng(random_state)

    means = np.array([
        [2.0, 1.0],
        [0.0, 6.0],
        [-2.0, 1.0],
    ])
    cov = np.array([[0.3, 0.0], [0.0, 0.3]])

    Xs = []
    for m in means:
        Xs.append(rng.multivariate_normal(mean=m, cov=cov, size=n_per))
    X = np.vstack(Xs)
    return X

def simulate_paper_gaussian(
    n: int = 300,
    setting: str = "strong",  # "strong" or "weak"
    random_state: int = 0,
) -> np.ndarray:
    """
    Simulate data from the 3-component 2D Gaussian mixture used in the paper.

    Means:
        (2, 1), (0, 6), (-2, 1)
    Equal weights: 1/3 each.

    setting = "strong": identity covariances (strongly identifiable)
    setting = "weak"  : location-scale covariances (weakly identifiable)
    """
    rng = np.random.default_rng(random_state)

    means = np.array([
        [ 2.0, 1.0],
        [ 0.0, 6.0],
        [-2.0, 1.0],
    ])
    pis = np.array([1/3, 1/3, 1/3])

    if setting == "strong":
        covs = np.array([np.eye(2) for _ in range(3)])
    else:
        # Weakly identifiable setting: covariances inspired by the paper.
        covs = np.array([
            [[0.5 ,  0.5], [0.5 ,  0.1]],
            [[0.5 , -0.1], [-0.1, 0.1]],
            [[0.25,  0.5], [0.5 ,  2.0]],
        ])

    # Sample n points from the mixture
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

def main() -> None:
    # 1. Simulate data
    # X = simulate_gaussian_mixture(n_per=200, random_state=0)
    X = simulate_paper_gaussian(n=300, setting="strong", random_state=0)
    true_k = 3
    over_k = 10  # slightly larger overfitting bound, like in the paper
    n, d = X.shape
    print(f"Simulated data: n={n}, d={d}")

    # 2. Fit overfitted Gaussian mixture with sklearn
    gmm = GaussianMixture(
        n_components=over_k,
        covariance_type="full",
        random_state=0,
        reg_covar=1e-6,
    ).fit(X)

    print(f"Fitted overfitted GaussianMixture with k={over_k} components.")

    # 3. Build mixing measure and dendrogram
    mm = MixingMeasure.from_sklearn_gmm(gmm)
    dend = MixingDendrogram(mm).build()

    print("Dendrogram built.")
    print(f"Initial components: {dend.n_initial_components}")
    print(f"Number of merges: {len(dend.heights)}")

    # 4. Compute DIC curve
    ks, dic_vals = dic_curve(X, dend, omega="logn")
    k_hat = select_k_dic(ks, dic_vals)

    print("DIC curve computed.")
    print(f"true k0 = {true_k}, overfitted k = {over_k}, selected k_hat = {k_hat}")

    # 5. Plot dendrogram + DIC curve
    Z = dend.to_linkage()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: dendrogram of mixing measures
    ax0 = axes[0]
    if Z.shape[0] > 0:
        scipy_dendrogram(Z, ax=ax0)
        ax0.set_title("Dendrogram of mixing measures")
        ax0.set_xlabel("Initial component index")
        ax0.set_ylabel("Merge height")
    else:
        ax0.text(0.5, 0.5, "Single component (no dendrogram)", ha="center")

    # Right: DIC curve
    ax1 = axes[1]
    # For readability, sort ks ascending:
    order = np.argsort(ks)
    ks_sorted = ks[order]
    dic_sorted = dic_vals[order]

    ax1.plot(ks_sorted, dic_sorted, marker="o")
    ax1.set_xlabel("Number of components k")
    ax1.set_ylabel("DIC(k)")
    ax1.set_title("DIC along dendrogram")
    ax1.invert_xaxis()  # optional: show large k on the left, small on the right

    ax1.axvline(k_hat, color="gray", linestyle="--", label=f"Selected k = {k_hat}")
    ax1.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()