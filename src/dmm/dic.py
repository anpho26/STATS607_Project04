from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from .mixing_measure import MixingMeasure
from .dendrogram import MixingDendrogram


def log_likelihood_gaussian(
    X: ArrayLike,
    mm: MixingMeasure,
    average: bool = True,
) -> float:
    """
    Log-likelihood of data under a Gaussian mixture mixing measure.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data matrix.
    mm : MixingMeasure
        Mixing measure containing weights, means, and covariances.
    average : bool, default True
        If True, return average log-likelihood (1/n sum log p(x_i)).
        If False, return total log-likelihood (sum log p(x_i)).

    Returns
    -------
    float
        Average or total log-likelihood.

    Notes
    -----
    - Assumes covs is not None and has shape (k, d, d).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    weights = mm.weights  # (k,)
    means = mm.means      # (k, d)
    covs = mm.covs

    if covs is None:
        raise ValueError(
            "log_likelihood_gaussian requires covariances; got covs=None."
        )

    k = mm.n_components
    if means.shape[1] != d:
        raise ValueError(
            f"Data dimension d={d} does not match means shape {means.shape}."
        )

    # log p(x | component j)
    # shape: (k, n)
    log_pdf_components = np.empty((k, n), dtype=float)
    for j in range(k):
        log_pdf_components[j, :] = multivariate_normal.logpdf(
            X, mean=means[j], cov=covs[j], allow_singular=True
        )

    # log (sum_j p_j * N_j(x))
    # = logsumexp_j (log p_j + log N_j(x))
    log_weights = np.log(weights)
    log_mix = logsumexp(log_weights[:, None] + log_pdf_components, axis=0)  # (n,)

    if average:
        return float(log_mix.mean())
    else:
        return float(log_mix.sum())


def dic_curve(
    X: ArrayLike,
    dendrogram: MixingDendrogram,
    omega: str | float = "logn",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DIC(kappa) along the dendrogram.

    For each stage (each number of components k), we compute:

        DIC(kappa) = - ( d^(kappa) + omega_n * ell^(kappa) ),

    where:
      - d^(kappa) is the merge height at that stage (0 for the initial level),
      - ell^(kappa) is the average log-likelihood at that stage,
      - omega_n = log(n) by default.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data matrix.
    dendrogram : MixingDendrogram
        A built dendrogram (call build() first).
    omega : {"logn"} or float, default "logn"
        Weight on the log-likelihood term.
        - "logn": use log(n).
        - float: use that constant value.

    Returns
    -------
    ks : np.ndarray, shape (L,)
        Numbers of components at each stage (from k0 down to 1).
    dic_values : np.ndarray, shape (L,)
        DIC values corresponding to each k in ks.
    """
    if not dendrogram.built:
        raise ValueError("dic_curve: dendrogram must be built first.")

    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if omega == "logn":
        omega_n = np.log(n)
    elif isinstance(omega, (int, float)):
        omega_n = float(omega)
    else:
        raise ValueError(
            f"Unsupported omega={omega!r}. Use 'logn' or a float."
        )

    k0 = dendrogram.n_initial_components

    ks = []
    dic_vals = []

    # measures[0] has k0 components, measures[t] has k0 - t components
    # heights[t-1] is the merge height that produced measures[t]
    for idx, mm in enumerate(dendrogram.measures):
        k = mm.n_components
        ks.append(k)

        if idx == 0:
            height = 0.0  # no merge yet
        else:
            height = dendrogram.heights[idx - 1]

        ell = log_likelihood_gaussian(X, mm, average=True)
        dic = - (height + omega_n * ell)
        dic_vals.append(dic)

    return np.array(ks, dtype=int), np.array(dic_vals, dtype=float)


def select_k_dic(ks: np.ndarray, dic_values: np.ndarray) -> int:
    """
    Select the number of components k that minimizes DIC.

    Parameters
    ----------
    ks : np.ndarray, shape (L,)
        Numbers of components.
    dic_values : np.ndarray, shape (L,)
        DIC values for each k.

    Returns
    -------
    k_hat : int
        Selected number of components.
    """
    if ks.shape != dic_values.shape:
        raise ValueError(
            f"ks and dic_values must have same shape, got {ks.shape} and {dic_values.shape}."
        )
    idx = int(np.argmin(dic_values))
    return int(ks[idx])