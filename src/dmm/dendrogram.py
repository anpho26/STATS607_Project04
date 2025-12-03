from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np

from .mixing_measure import MixingMeasure


@dataclass
class MergeRecord:
    """One merge step in the dendrogram (SciPy-style linkage row)."""

    left_id: int      # cluster id (global) of left child
    right_id: int     # cluster id (global) of right child
    height: float     # merge dissimilarity
    size: int         # number of original components in the new cluster


class MixingDendrogram:
    """
    Greedy dendrogram construction over the atoms of a MixingMeasure.

    Given an overfitted mixing measure G^(k) with k components, this class:

    - repeatedly merges the closest pair of components under a dissimilarity
      d(p_i, θ_i; p_j, θ_j),
    - records the sequence of mixing measures as k decreases,
    - stores merge heights and cluster membership history,
    - exposes a SciPy-compatible linkage matrix for plotting.

    Notes
    -----
    - For now, the dissimilarity uses only means and weights:

          d(i, j) = ||μ_i - μ_j||^2 / (p_i^{-1} + p_j^{-1}),

      consistent with the formula used in your README.
    - Covariances are updated via a moment-preserving formula when present.
    """

    def __init__(self, mixing_measure: MixingMeasure) -> None:
        # Initial overfitted measure (copied to avoid side effects)
        self.initial_measure: MixingMeasure = mixing_measure.copy()

        # Sequence of measures as we merge (index 0 = initial)
        self.measures: List[MixingMeasure] = [self.initial_measure]

        # Dendrogram heights, one per merge (length k0 - 1)
        self.heights: List[float] = []

        # Merge history, in cluster-id space (for SciPy linkage)
        self.merge_history: List[MergeRecord] = []

        # Groups per level: at each stage, a list of sets of original indices
        # groups_per_level[0] = [{0}, {1}, ..., {k0-1}]
        self.groups_per_level: List[List[Set[int]]] = []

        # Built flag
        self._built: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self) -> "MixingDendrogram":
        """
        Run the greedy merging algorithm (Algorithms 1–2).

        After calling this method, the attributes
        - measures
        - heights
        - merge_history
        - groups_per_level

        are populated.

        Returns
        -------
        self : MixingDendrogram
        """
        # Reset state in case build() is called twice
        self.measures = [self.initial_measure.copy()]
        self.heights = []
        self.merge_history = []
        self.groups_per_level = []

        mm = self.initial_measure.copy()
        k0 = mm.n_components

        if k0 < 1:
            raise ValueError("Mixing measure must have at least one component.")
        if k0 == 1:
            # Trivial dendrogram: nothing to merge
            self.groups_per_level = [[{0}]]
            self._built = True
            return self

        # Initial groups: each component is its own group
        current_groups: List[Set[int]] = [set([i]) for i in range(k0)]
        self.groups_per_level.append([g.copy() for g in current_groups])

        # Cluster IDs used to build a SciPy-style linkage matrix.
        # Original components: ids 0..k0-1
        cluster_ids: List[int] = list(range(k0))
        next_cluster_id: int = k0  # new clusters will be k0, k0+1, ...

        # Greedy merging
        while mm.n_components > 1:
            # Compute pairwise dissimilarities
            D = _pairwise_dissimilarity(mm)

            # Find closest pair (upper triangular, excluding diagonal)
            i, j = _argmin_upper_triangle(D)
            if i is None or j is None:
                raise RuntimeError("Failed to find a valid pair to merge.")

            # Ensure i < j for stable indexing
            if i > j:
                i, j = j, i

            height = float(D[i, j])

            # Record merge in global cluster-id space
            left_id = cluster_ids[i]
            right_id = cluster_ids[j]
            new_group = current_groups[i] | current_groups[j]
            new_size = len(new_group)

            self.merge_history.append(
                MergeRecord(
                    left_id=left_id,
                    right_id=right_id,
                    height=height,
                    size=new_size,
                )
            )
            self.heights.append(height)

            # Merge mixing measure components i and j
            mm = _merge_components(mm, i, j)

            # Update groups: replace group i with merged group, remove group j
            current_groups[i] = new_group
            del current_groups[j]

            # Update cluster IDs: new cluster gets next_cluster_id
            cluster_ids[i] = next_cluster_id
            del cluster_ids[j]
            next_cluster_id += 1

            # Store current state
            self.measures.append(mm.copy())
            self.groups_per_level.append([g.copy() for g in current_groups])

        self._built = True
        return self

    @property
    def built(self) -> bool:
        """Whether `build()` has been successfully run."""
        return self._built

    @property
    def n_initial_components(self) -> int:
        """Number of components in the initial overfitted measure."""
        return self.initial_measure.n_components

    def get_measure_at_k(self, k: int) -> MixingMeasure:
        """
        Return the MixingMeasure at the stage with k components.

        Parameters
        ----------
        k : int
            Desired number of components, 1 <= k <= k0.

        Returns
        -------
        MixingMeasure

        Raises
        ------
        ValueError
            If k is out of range, or if build() has not been called.
        """
        if not self._built:
            raise ValueError("Dendrogram not built yet. Call build() first.")
        k0 = self.n_initial_components
        if not (1 <= k <= k0):
            raise ValueError(f"k must be between 1 and {k0}, got {k}.")

        # measures[0] has k0 components, measures[t] has k0 - t components
        # so index = k0 - k
        idx = k0 - k
        return self.measures[idx].copy()

    def to_linkage(self) -> np.ndarray:
        """
        Return a SciPy-compatible linkage matrix.

        The result is an array Z with shape (k0 - 1, 4), where each row is

            [left_id, right_id, height, size]

        following the convention of scipy.cluster.hierarchy.linkage.

        Returns
        -------
        Z : np.ndarray, shape (k0 - 1, 4)

        Raises
        ------
        ValueError
            If build() has not been called.
        """
        if not self._built:
            raise ValueError("Dendrogram not built yet. Call build() first.")
        if self.n_initial_components <= 1:
            return np.zeros((0, 4), dtype=float)

        Z = np.zeros((len(self.merge_history), 4), dtype=float)
        for t, rec in enumerate(self.merge_history):
            Z[t, 0] = rec.left_id
            Z[t, 1] = rec.right_id
            Z[t, 2] = rec.height
            Z[t, 3] = rec.size
        return Z


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #
def _pairwise_dissimilarity(mm: MixingMeasure) -> np.ndarray:
    """
    Compute the pairwise dissimilarity matrix between components.

    d(i, j) = ||μ_i - μ_j||^2 / (p_i^{-1} + p_j^{-1})

    Parameters
    ----------
    mm : MixingMeasure

    Returns
    -------
    D : np.ndarray, shape (k, k)
        Symmetric matrix with zeros on the diagonal.
    """
    weights = mm.weights  # shape (k,)
    means = mm.means      # shape (k, d)
    k = mm.n_components

    # Compute squared Euclidean distances between all pairs of means
    # Efficiently: ||μ_i - μ_j||^2 = ||μ_i||^2 + ||μ_j||^2 - 2 μ_i·μ_j
    sq_norms = np.sum(means ** 2, axis=1)  # (k,)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * means @ means.T  # (k, k)

    # Harmonic-like denominator term
    inv_w = 1.0 / weights  # (k,)
    denom = inv_w[:, None] + inv_w[None, :]  # (k, k)

    D = D2 / denom
    np.fill_diagonal(D, np.inf)  # never merge a component with itself
    return D


def _argmin_upper_triangle(D: np.ndarray) -> Tuple[int | None, int | None]:
    """
    Find (i, j) with minimal D[i, j] over the strict upper triangle.

    Parameters
    ----------
    D : np.ndarray, shape (k, k)

    Returns
    -------
    i, j : int or (None, None)
        Indices of the minimal off-diagonal entry, or (None, None) if k < 2.
    """
    k = D.shape[0]
    if k < 2:
        return None, None

    # Use upper triangle indices (i < j)
    iu, ju = np.triu_indices(k, k=1)
    flat_idx = np.argmin(D[iu, ju])
    i = int(iu[flat_idx])
    j = int(ju[flat_idx])
    return i, j


def _merge_components(mm: MixingMeasure, i: int, j: int) -> MixingMeasure:
    """
    Merge components i and j in the given mixing measure.

    Parameters
    ----------
    mm : MixingMeasure
        Current mixing measure (k components).
    i, j : int
        Indices of components to merge (i < j).

    Returns
    -------
    new_mm : MixingMeasure
        New mixing measure with k - 1 components.

    Notes
    -----
    - We keep total weight fixed (weights remain normalized).
    - If covariances are present, we use a moment-preserving update:

        Σ* = sum_c (w_c / w*) [ Σ_c + (μ_c - μ*)(μ_c - μ*)^T ]

      where c runs over {i, j} and w* = w_i + w_j.
    """
    if i == j:
        raise ValueError("Cannot merge a component with itself.")
    if i > j:
        i, j = j, i

    weights = mm.weights.copy()
    means = mm.means.copy()
    covs = None if mm.covs is None else mm.covs.copy()

    wi, wj = weights[i], weights[j]
    w_star = wi + wj

    # New mean
    mu_i = means[i]
    mu_j = means[j]
    mu_star = (wi * mu_i + wj * mu_j) / w_star

    # New covariance, if present
    if covs is not None:
        Sigma_i = covs[i]
        Sigma_j = covs[j]
        # Between-component adjustment
        diff_i = (mu_i - mu_star).reshape(-1, 1)
        diff_j = (mu_j - mu_star).reshape(-1, 1)
        Sigma_star = (
            (wi / w_star) * (Sigma_i + diff_i @ diff_i.T)
            + (wj / w_star) * (Sigma_j + diff_j @ diff_j.T)
        )
    else:
        Sigma_star = None

    # Build new arrays with k - 1 components
    # Replace row i with merged, delete row j
    new_weights = np.delete(weights, j)
    new_means = np.delete(means, j, axis=0)
    new_weights[i] = w_star
    new_means[i] = mu_star

    if Sigma_star is not None:
        new_covs = np.delete(covs, j, axis=0)
        new_covs[i] = Sigma_star
    else:
        new_covs = None

    # Construct new MixingMeasure (normalization happens inside)
    return MixingMeasure(weights=new_weights, means=new_means, covs=new_covs)