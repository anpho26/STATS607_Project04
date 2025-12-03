from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike


Number = Union[float, int]


@dataclass
class MixingMeasure:
    """
    Representation of a discrete mixing measure

        G = sum_{i=1}^k p_i δ_{θ_i},

    specialized (for now) to Gaussian mixture components with
    mean vectors and covariance matrices.

    Attributes
    ----------
    weights : np.ndarray, shape (k,)
        Nonnegative mixture weights (normalized to sum to 1).
    means : np.ndarray, shape (k, d)
        Mean vectors for each component.
    covs : Optional[np.ndarray], shape (k, d, d), optional
        Covariance matrices for each component. May be None if
        we are working in a “strongly identifiable” setting that
        only uses locations.
    """

    weights: np.ndarray
    means: np.ndarray
    covs: Optional[np.ndarray] = None

    # -------------------------- constructors -------------------------- #
    def __post_init__(self) -> None:
        # Cast to arrays
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)

        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {self.weights.shape}")
        if self.means.ndim != 2:
            raise ValueError(f"means must be 2D (k, d), got shape {self.means.shape}")

        k = self.weights.shape[0]
        if self.means.shape[0] != k:
            raise ValueError(
                f"weights and means must have same number of components, "
                f"got {k} and {self.means.shape[0]}"
            )

        if self.covs is not None:
            self.covs = np.asarray(self.covs, dtype=float)
            if self.covs.ndim != 3:
                raise ValueError(
                    f"covs must be 3D (k, d, d), got shape {self.covs.shape}"
                )
            if self.covs.shape[0] != k:
                raise ValueError(
                    f"weights and covs must have same number of components, "
                    f"got {k} and {self.covs.shape[0]}"
                )
            d = self.means.shape[1]
            if self.covs.shape[1:] != (d, d):
                raise ValueError(
                    f"covs must have shape (k, d, d) with d={d}, "
                    f"got {self.covs.shape}"
                )

        self._normalize_in_place()

    @classmethod
    def from_arrays(
        cls,
        weights: ArrayLike,
        means: ArrayLike,
        covs: Optional[ArrayLike] = None,
        normalize: bool = True,
    ) -> "MixingMeasure":
        """
        Construct a MixingMeasure from raw arrays.

        Parameters
        ----------
        weights : array-like, shape (k,)
            Mixture weights (do not need to be normalized).
        means : array-like, shape (k, d)
            Component mean vectors.
        covs : array-like, shape (k, d, d), optional
            Component covariance matrices.
        normalize : bool, default True
            If True, re-normalize weights to sum to 1.

        Returns
        -------
        MixingMeasure
        """
        mm = cls(weights=np.asarray(weights), means=np.asarray(means), covs=covs)
        if normalize:
            mm._normalize_in_place()
        return mm

    @classmethod
    def from_sklearn_gmm(cls, gmm, normalize: bool = True) -> "MixingMeasure":
        """
        Construct a MixingMeasure from a fitted sklearn.mixture.GaussianMixture.

        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            A fitted GaussianMixture instance.
        normalize : bool, default True
            If True, re-normalize weights (usually already normalized by sklearn).

        Notes
        -----
        - Supports covariance_type="full" and "diag".
        - For "diag", we convert diagonal variances to full covariance matrices.
        - For "tied" or "spherical", we currently raise NotImplementedError.
        """
        weights = np.asarray(gmm.weights_, dtype=float)
        means = np.asarray(gmm.means_, dtype=float)

        cov_type = getattr(gmm, "covariance_type", "full")
        covs_raw = np.asarray(gmm.covariances_, dtype=float)

        if cov_type == "full":
            covs = covs_raw
        elif cov_type == "diag":
            # Each row is the diagonal of the covariance
            covs = np.array([np.diag(v) for v in covs_raw])
        elif cov_type == "tied":
            # One shared covariance matrix for all components
            # Expand to (k, d, d)
            k = weights.shape[0]
            covs = np.repeat(covs_raw[None, :, :], k, axis=0)
        elif cov_type == "spherical":
            # Spherical: each component has variance * I
            k = weights.shape[0]
            d = means.shape[1]
            covs = np.array([np.eye(d) * v for v in covs_raw])
        else:
            raise NotImplementedError(
                f"Unsupported covariance_type={cov_type!r} for from_sklearn_gmm."
            )

        mm = cls(weights=weights, means=means, covs=covs)
        if normalize:
            mm._normalize_in_place()
        return mm

    # ---------------------------- properties -------------------------- #
    @property
    def n_components(self) -> int:
        """Number of mixture components k."""
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        """Dimension of the observation space d."""
        return int(self.means.shape[1])

    # ---------------------------- utilities --------------------------- #
    def _normalize_in_place(self) -> None:
        """Normalize weights so they sum to 1, in-place."""
        total = float(self.weights.sum())
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.weights /= total

    def copy(self) -> "MixingMeasure":
        """Deep copy of the mixing measure."""
        covs_copy = None if self.covs is None else self.covs.copy()
        return MixingMeasure(
            weights=self.weights.copy(),
            means=self.means.copy(),
            covs=covs_copy,
        )

    def __repr__(self) -> str:
        return (
            f"MixingMeasure(k={self.n_components}, "
            f"d={self.dim}, covs={'yes' if self.covs is not None else 'no'})"
        )