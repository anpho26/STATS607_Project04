from __future__ import annotations

from pathlib import Path
import sys

# Make src/ visible so we can import dmm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from scipy.stats import laplace, norm
from scipy.cluster.hierarchy import linkage, dendrogram as h_dendrogram
# ============================================================
# 2.4. ILLUSTRATIONS FOR WELL-SPECIFIED AND SKEW-NORMAL REGIMES
# ============================================================

def plot_well_hist_and_dendrogram(
    cfg: WellSpecifiedConfig,
    n: int = 200,
    seed: int = 123,
) -> None:
    """Panels (a) and (b) for the well-specified Gaussian mixture."""
    rng = np.random.default_rng(seed)

    # sample from the true well-specified Gaussian mixture
    X = sample_gaussian_mixture_1d(n, cfg.weights, cfg.means, cfg.sigmas, rng)
    x_flat = X.ravel()

    # grid for true density
    x_min, x_max = x_flat.min() - 1.0, x_flat.max() + 1.0
    xx = np.linspace(x_min, x_max, 400)

    # true Gaussian mixture density
    base_pdf = np.zeros_like(xx)
    for w, m, s in zip(cfg.weights, cfg.means, cfg.sigmas):
        base_pdf += w * norm.pdf(xx, loc=m, scale=s)

    out_dir = PROJECT_ROOT / "out" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) histogram + true density
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(x_flat, bins=40, density=True, alpha=0.5,
            color="tab:red", edgecolor="none")
    ax.plot(xx, base_pdf, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("(a) Well-specified: histogram with true density")
    fig.tight_layout()
    fig.savefig(out_dir / "well_hist_true_density_parallel.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) dendrogram of overfitted GMM with K_MAX atoms
    gmm = fit_gmm_for_k(X, k=K_MAX, rng=rng)
    means = gmm.means_.ravel()
    Z = linkage(means[:, None], method="average")

    fig, ax = plt.subplots(figsize=(4, 3))
    h_dendrogram(Z, ax=ax, labels=np.arange(K_MAX))
    ax.set_xlabel("Initial component index")
    ax.set_ylabel("Merge height")
    ax.set_title("(b) Well-specified: dendrogram of mixing measure with 10 atoms")
    fig.tight_layout()
    fig.savefig(out_dir / "well_dendrogram_k10_parallel.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def _skew_pdf(x: np.ndarray, xi: float, omega: float, alpha: float) -> np.ndarray:
    """Skew-normal density SN(xi, omega, alpha) at x."""
    z = (x - xi) / omega
    return (2.0 / omega) * norm.pdf(z) * norm.cdf(alpha * z)


def plot_skew_hist_and_dendrogram(
    cfg: SkewNormalConfig,
    n: int = 200,
    seed: int = 123,
) -> None:
    """Panels (a) and (b) for the skew-normal mixture."""
    rng = np.random.default_rng(seed)

    # sample from the true skew-normal mixture
    X = sample_skew_mixture_1d(n, cfg, rng)
    x_flat = X.ravel()

    # grid for true density
    x_min, x_max = x_flat.min() - 1.0, x_flat.max() + 1.0
    xx = np.linspace(x_min, x_max, 400)

    # true skew-normal mixture density
    base_pdf = np.zeros_like(xx)
    for w, xi, om, al in zip(cfg.weights, cfg.xi, cfg.omega, cfg.alpha):
        base_pdf += w * _skew_pdf(xx, xi, om, al)

    out_dir = PROJECT_ROOT / "out" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) histogram + true density
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(x_flat, bins=40, density=True, alpha=0.5,
            color="tab:red", edgecolor="none")
    ax.plot(xx, base_pdf, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("(a) Skew-normal mixture: histogram with true density")
    fig.tight_layout()
    fig.savefig(out_dir / "skew_hist_true_density_parallel.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) dendrogram of overfitted GMM with K_MAX atoms
    gmm = fit_gmm_for_k(X, k=K_MAX, rng=rng)
    means = gmm.means_.ravel()
    Z = linkage(means[:, None], method="average")

    fig, ax = plt.subplots(figsize=(4, 3))
    h_dendrogram(Z, ax=ax, labels=np.arange(K_MAX))
    ax.set_xlabel("Initial component index")
    ax.set_ylabel("Merge height")
    ax.set_title("(b) Skew-normal mixture: dendrogram of mixing measure with 10 atoms")
    fig.tight_layout()
    fig.savefig(out_dir / "skew_dendrogram_k10_parallel.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 2.5. ILLUSTRATION FOR EPS-CONTAMINATION (PANELS (a) AND (b))
# ============================================================

def plot_eps_contam_hist_and_dendrogram(
    cfg: EpsContamConfig,
    n: int = 200,
    seed: int = 123,
) -> None:
    """Draw panels (a) and (b) for the ε-contamination experiment.

    (a) Histogram with true density of the contaminated distribution.
    (b) Dendrogram of the overfitted mixing measure with k_max components.

    The figures are saved under out/figures/ as PNG files.
    """
    rng = np.random.default_rng(seed)

    # --- sample a dataset from the ε-contamination model ---
    X = sample_eps_contamination(n, cfg, rng)
    x_flat = X.ravel()

    # grid for true density
    x_min, x_max = x_flat.min() - 1.0, x_flat.max() + 1.0
    xx = np.linspace(x_min, x_max, 400)

    # base Gaussian mixture density
    base_pdf = (
        cfg.weights[0] * norm.pdf(xx, loc=cfg.means[0], scale=cfg.sigmas[0])
        + cfg.weights[1] * norm.pdf(xx, loc=cfg.means[1], scale=cfg.sigmas[1])
    )
    # contaminated density: (1-ε) * base + ε * Laplace
    p0_pdf = (1 - cfg.eps) * base_pdf + cfg.eps * laplace.pdf(
        xx, loc=cfg.laplace_loc, scale=cfg.laplace_scale
    )

    out_dir = PROJECT_ROOT / "out" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) Histogram with true density ---
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(x_flat, bins=40, density=True, alpha=0.5, color="tab:red", edgecolor="none")
    ax.plot(xx, p0_pdf, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("(a) ε-contamination: histogram with true density")
    fig.tight_layout()
    fig.savefig(out_dir / "eps_contam_hist_true_density_parallel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (b) Dendrogram of mixing measure with k_max atoms ---
    gmm = fit_gmm_for_k(X, k=K_MAX, rng=rng)
    means = gmm.means_.ravel()  # shape (K_MAX,), 1D component locations

    # Perform agglomerative clustering on component means
    Z = linkage(means[:, None], method="average")

    fig, ax = plt.subplots(figsize=(4, 3))
    h_dendrogram(Z, ax=ax, labels=np.arange(K_MAX))
    ax.set_xlabel("Initial component index")
    ax.set_ylabel("Merge height")
    ax.set_title("(b) ε-contamination: dendrogram of mixing measure with 10 atoms")
    fig.tight_layout()
    fig.savefig(out_dir / "eps_contam_dendrogram_k10_parallel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

from sklearn.mixture import GaussianMixture

from dmm.dendrogram import MixingDendrogram
from dmm.mixing_measure import MixingMeasure
from dmm.dic import dic_curve, select_k_dic

import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# ------------------------------------------------------------
# Plot helper (same as in your current file)
# ------------------------------------------------------------
def plot_selection_results(
    results: Dict[str, np.ndarray],
    scenario_label: str,
    filename: str,
) -> None:
    """Plot proportion of choosing k0 and average chosen k for AIC/BIC/DIC,
    and save the figure to disk.
    """
    log10_n = results["log10_n"]
    k0 = results["k0"]

    methods = ["AIC", "BIC", "DIC"]
    colors = {"AIC": "tab:red", "BIC": "tab:green", "DIC": "tab:blue"}
    linestyles = {"AIC": "--", "BIC": "-", "DIC": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)

    # Panel (c): proportion of choosing k0
    ax = axes[0]
    ax.set_title(r"(c) Proportion of choosing $k_0 = %d$" % k0)
    for m in methods:
        prop = results[f"prop_correct_{m.lower()}"]
        ax.plot(log10_n, prop, linestyle=linestyles[m], color=colors[m], label=m)
    ax.set_xlabel(r"log$_{10}(n)$")
    ax.set_ylabel("Correct proportion")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # Panel (d): average chosen k
    ax = axes[1]
    ax.set_title("(d) Average chosen number of components")
    for m in methods:
        avg_k = results[f"avg_k_{m.lower()}"]
        ax.plot(log10_n, avg_k, linestyle=linestyles[m], color=colors[m], label=m)
    ax.set_xlabel(r"log$_{10}(n)$")
    ax.set_ylabel("Average number of components")

    fig.suptitle(scenario_label)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])

    # Save figure to out/figures directory under the project root
    out_dir = PROJECT_ROOT / "out" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    # Only save; no GUI
    plt.close(fig)


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class WellSpecifiedConfig:
    # true k0 and mixture parameters for the *Gaussian* location–scale mixture
    k0: int = 2
    # Simple symmetric 2-component mixture
    weights: tuple = (0.5, 0.5)
    means: tuple = (-3.0, 3.0)
    sigmas: tuple = (0.7, 0.7)


@dataclass
class EpsContamConfig:
    # true base Gaussian mixture (before contamination), k0 = 2
    # p_G0(x) = 0.4 N(5, 1^2) + 0.6 N(10, 1.5^2)
    k0: int = 2
    weights: tuple = (0.4, 0.6)
    means: tuple = (5.0, 10.0)
    sigmas: tuple = (1.0, 1.5)
    # contamination level and Laplace parameters
    eps: float = 0.01
    laplace_loc: float = 0.0
    laplace_scale: float = 1.0


@dataclass
class SkewNormalConfig:
    # mixture of two skew–normal components, k0 = 2
    # p0(x) = 0.4 * Skew(5, 1, 20) + 0.6 * Skew(8, 1.5, 20)
    k0: int = 2
    weights: tuple = (0.4, 0.6)
    xi:    tuple = (5.0, 8.0)   # locations
    omega: tuple = (1.0, 1.5)   # scales
    alpha: tuple = (20.0, 20.0) # skewness


CFG_WELL = WellSpecifiedConfig()
CFG_EPS = EpsContamConfig()
CFG_SKEW = SkewNormalConfig()

# sample-size grid: log10 n from 2 to 4 in steps of 0.25 (as in figs)
LOG10_NS = np.arange(2.0, 4.01, 0.25)
N_GRID = (10 ** LOG10_NS).astype(int)

K_MAX = 10   # upper bound on number of components we fit
N_REP = 100  # number of replications per n


# ============================================================
# 2. BASIC SAMPLERS
# ============================================================

def sample_gaussian_mixture_1d(
    n: int,
    weights: tuple,
    means: tuple,
    sigmas: tuple,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    k = len(weights)
    assert means.shape == (k,)
    assert sigmas.shape == (k,)

    z = rng.choice(k, size=n, p=weights)
    x = rng.normal(loc=means[z], scale=sigmas[z])
    return x[:, None]  # shape (n,1) for sklearn


def rskewnorm(
    n: int,
    xi: float,
    omega: float,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate from skew-normal SN(xi, omega, alpha)."""
    if rng is None:
        rng = np.random.default_rng()

    delta = alpha / np.sqrt(1 + alpha**2)

    z0 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    x = xi + omega * (delta * np.abs(z0) + np.sqrt(1 - delta**2) * z1)
    return x


def sample_skew_mixture_1d(
    n: int,
    cfg: SkewNormalConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    weights = np.asarray(cfg.weights, dtype=float)
    xi = np.asarray(cfg.xi, dtype=float)
    omega = np.asarray(cfg.omega, dtype=float)
    alpha = np.asarray(cfg.alpha, dtype=float)

    k = len(weights)
    z = rng.choice(k, size=n, p=weights)
    x = np.empty(n)
    for j in range(k):
        idx = (z == j)
        m = idx.sum()
        if m > 0:
            x[idx] = rskewnorm(m, xi[j], omega[j], alpha[j], rng=rng)
    return x[:, None]


def sample_eps_contamination(
    n: int,
    cfg: EpsContamConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from p0 = (1 - eps) * p_G + eps * Laplace(loc, scale)."""
    if rng is None:
        rng = np.random.default_rng()

    is_contam = rng.random(n) < cfg.eps

    # base Gaussian mixture for non-contaminated points
    n_base = (~is_contam).sum()
    x = np.empty(n)

    if n_base > 0:
        x[~is_contam] = sample_gaussian_mixture_1d(
            n_base, cfg.weights, cfg.means, cfg.sigmas, rng
        ).ravel()

    # Laplace contamination
    n_contam = is_contam.sum()
    if n_contam > 0:
        x[is_contam] = laplace.rvs(
            loc=cfg.laplace_loc,
            scale=cfg.laplace_scale,
            size=n_contam,
            random_state=rng,
        )
    return x[:, None]


# ============================================================
# 3. FITTING GMMs AND DIC ALONG DENDROGRAM
# ============================================================

def fit_gmm_for_k(X: ArrayLike, k: int, rng: np.random.Generator) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=3,
        max_iter=500,
        random_state=rng.integers(1_000_000),
    )
    gmm.fit(X)
    return gmm


def aic_bic_for_k_range(
    X: ArrayLike,
    k_range: range,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    aics = []
    bics = []
    for k in k_range:
        gmm = fit_gmm_for_k(X, k, rng)
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))
    return np.array(aics), np.array(bics)


def fit_overfitted_and_dendrogram(
    X: ArrayLike,
    k_max: int,
    rng: np.random.Generator,
) -> MixingDendrogram:
    gmm = fit_gmm_for_k(X, k_max, rng)
    mm = MixingMeasure.from_sklearn_gmm(gmm)
    dendro = MixingDendrogram(mm).build()
    return dendro


def select_k_all_criteria(
    X: ArrayLike,
    k_max: int,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    k_range = range(1, k_max + 1)

    aics, bics = aic_bic_for_k_range(X, k_range, rng)
    ks = np.array(list(k_range))

    k_aic = int(ks[np.argmin(aics)])
    k_bic = int(ks[np.argmin(bics)])

    dendro = fit_overfitted_and_dendrogram(X, k_max, rng)
    ks_dic, dic_vals = dic_curve(X, dendro, omega="logn")
    k_dic = select_k_dic(ks_dic, dic_vals)

    return k_aic, k_bic, k_dic


# ============================================================
# 4. PARALLELIZED SINGLE-REP WORKER
# ============================================================

def _one_replication(
    setting: str,
    n: int,
    k_max: int,
    seed: int,
) -> Tuple[int, int, int]:
    """Run one replication for a given setting and sample size.

    This function is designed to be called in parallel (picklable, top-level).
    """
    rng = np.random.default_rng(seed)

    if setting == "well":
        cfg = CFG_WELL
        X = sample_gaussian_mixture_1d(
            n, cfg.weights, cfg.means, cfg.sigmas, rng
        )
    elif setting == "eps":
        cfg = CFG_EPS
        X = sample_eps_contamination(n, cfg, rng)
    elif setting == "skew":
        cfg = CFG_SKEW
        X = sample_skew_mixture_1d(n, cfg, rng)
    else:
        raise ValueError(f"Unknown setting '{setting}'")

    k_aic, k_bic, k_dic = select_k_all_criteria(X, k_max, rng)
    return k_aic, k_bic, k_dic


# ============================================================
# 5. MAIN SIMULATION WRAPPER (PARALLEL)
# ============================================================

def run_experiment_parallel(
    setting: str,
    n_grid: np.ndarray = N_GRID,
    n_rep: int = N_REP,
    k_max: int = K_MAX,
    seed: int = 123,
    n_jobs: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Parallel version of run_experiment.

    setting in {"well", "eps", "skew"}.
    """
    rng = np.random.default_rng(seed)

    if setting == "well":
        k0 = CFG_WELL.k0
    elif setting == "eps":
        k0 = CFG_EPS.k0
    elif setting == "skew":
        k0 = CFG_SKEW.k0
    else:
        raise ValueError(f"Unknown setting '{setting}'")

    n_grid = np.asarray(n_grid, dtype=int)
    n_n = len(n_grid)

    prop_correct_aic = np.zeros(n_n)
    prop_correct_bic = np.zeros(n_n)
    prop_correct_dic = np.zeros(n_n)

    avg_k_aic = np.zeros(n_n)
    avg_k_bic = np.zeros(n_n)
    avg_k_dic = np.zeros(n_n)

    for i, n in enumerate(n_grid):
        # Generate independent seeds for each replication at this n
        seeds = rng.integers(1_000_000_000, size=n_rep, dtype=np.int64)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_replication)(setting, int(n), k_max, int(s))
            for s in seeds
        )

        results = np.array(results)  # shape (n_rep, 3)
        k_aic_all = results[:, 0]
        k_bic_all = results[:, 1]
        k_dic_all = results[:, 2]

        prop_correct_aic[i] = np.mean(k_aic_all == k0)
        prop_correct_bic[i] = np.mean(k_bic_all == k0)
        prop_correct_dic[i] = np.mean(k_dic_all == k0)

        avg_k_aic[i] = k_aic_all.mean()
        avg_k_bic[i] = k_bic_all.mean()
        avg_k_dic[i] = k_dic_all.mean()

        print(
            f"[{setting}] n={n:5d}:  "
            f"AIC: {prop_correct_aic[i]:.2f}, "
            f"BIC: {prop_correct_bic[i]:.2f}, "
            f"DIC: {prop_correct_dic[i]:.2f}"
        )

    return dict(
        n_grid=n_grid,
        log10_n=np.log10(n_grid.astype(float)),
        prop_correct_aic=prop_correct_aic,
        prop_correct_bic=prop_correct_bic,
        prop_correct_dic=prop_correct_dic,
        avg_k_aic=avg_k_aic,
        avg_k_bic=avg_k_bic,
        avg_k_dic=avg_k_dic,
        k0=k0,
    )


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    # Well-specified Gaussian mixture (k0 = 2): (a),(b),(c),(d)
    plot_well_hist_and_dendrogram(CFG_WELL, n=200, seed=123)
    res_well = run_experiment_parallel("well")
    plot_selection_results(
        res_well,
        "Well-specified Gaussian mixture (k0 = 2)",
        "model_selection_well_specified_k2_parallel.png",
    )

    # ε-contamination: (a),(b),(c),(d)
    plot_eps_contam_hist_and_dendrogram(CFG_EPS, n=200, seed=456)
    res_eps = run_experiment_parallel("eps")
    plot_selection_results(
        res_eps,
        r"Experiments with $\epsilon$-contamination data",
        "model_selection_eps_contamination_parallel.png",
    )

    # Skew-normal mixture: (a),(b),(c),(d)
    plot_skew_hist_and_dendrogram(CFG_SKEW, n=200, seed=789)
    res_skew = run_experiment_parallel("skew")
    plot_selection_results(
        res_skew,
        "Experiments with mixture of skew-normal data",
        "model_selection_skew_normal_parallel.png",
    )