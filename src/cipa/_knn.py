"""Shared k-NN infrastructure. See §3.1 of García Rodríguez et al. (2026).

This module is part of the CIPA software package, companion implementation to:

    García Rodríguez, L., Neme Castillo, J. A., Gómez Adorno, H. M., & Fuentes Pineda, G. (2026).
    CIPA: A Multi-Domain Statistical Framework for Characterizing Imbalanced
    Datasets and Computing a Difficulty Score.
    COMIA 2026 — XVIII Congreso Mexicano de Inteligencia Artificial.
    DOI: TODO (pending publication)

Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Universidad Nacional Autónoma de México (UNAM)

Development supported by SECIHTI (researcher ID (CVU) 905206, Luis García Rodríguez).

License: MIT — see LICENSE file for full terms.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from cipa.dataset import CIPADataset

logger = logging.getLogger(__name__)


class _KNNCache:
    """Pre-fitted NearestNeighbors shared by D2 (kDN), D3 (NS-typology), D7 (N2).

    Fits once; caches query results for the default k.
    Querying includes self-exclusion (training point excluded from its own
    neighborhood by requesting k+1 neighbors and dropping the first result).
    """

    def __init__(
        self,
        dataset: CIPADataset,
        k: int,
        algorithm: str = "ball_tree",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= dataset.N:
            k = dataset.N - 1
            logger.warning("_KNNCache: k reduced to %d (= N-1)", k)

        self._dataset = dataset
        self._k = k
        # Request k+1 to allow self-exclusion; cap at N
        self._n_fit = min(k + 1, dataset.N)

        self._nn = NearestNeighbors(n_neighbors=self._n_fit, algorithm=algorithm)
        self._nn.fit(dataset.X)

        self._cache_all: tuple[np.ndarray, np.ndarray] | None = None
        self._cache_minority: tuple[np.ndarray, np.ndarray] | None = None

    def query_all(self) -> tuple[np.ndarray, np.ndarray]:
        """k nearest neighbors for all N instances, excluding self.

        Returns
        -------
        distances : (N, k)
        indices   : (N, k)  — indices into dataset.X
        """
        if self._cache_all is None:
            dists, idxs = self._nn.kneighbors(self._dataset.X, n_neighbors=self._n_fit)
            # Remove self (column 0 is always self with distance ≈ 0)
            self._cache_all = (dists[:, 1:], idxs[:, 1:])
        return self._cache_all

    def query_minority(self) -> tuple[np.ndarray, np.ndarray]:
        """k nearest neighbors for minority instances, excluding self.

        Neighbors are drawn from the full dataset (both classes).

        Returns
        -------
        distances : (n_minority, k)
        indices   : (n_minority, k)  — indices into dataset.X
        """
        if self._cache_minority is None:
            dists, idxs = self._nn.kneighbors(
                self._dataset.X_minority, n_neighbors=self._n_fit
            )
            self._cache_minority = (dists[:, 1:], idxs[:, 1:])
        return self._cache_minority


def _maybe_subsample(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int | bool,
    majority_label: int | bool,
    max_exact: int,
    subsample_size: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Stratified subsample for large datasets.

    Returns (X_out, y_out, was_subsampled). If len(X) <= max_exact, returns
    the original arrays unchanged. Otherwise returns a stratified subsample
    of size subsample_size preserving the minority/majority ratio.
    """
    if len(X) <= max_exact:
        return X, y, False

    rng = np.random.default_rng(random_state)
    min_mask = y == minority_label
    maj_mask = y == majority_label
    n_min = int(min_mask.sum())
    n_maj = int(maj_mask.sum())
    n_total = len(X)

    # Proportional allocation; ensure at least 2 minority
    n_min_sample = max(2, int(subsample_size * n_min / n_total))
    n_maj_sample = subsample_size - n_min_sample

    n_min_sample = min(n_min_sample, n_min)
    n_maj_sample = min(n_maj_sample, n_maj)

    min_idx = np.where(min_mask)[0]
    maj_idx = np.where(maj_mask)[0]

    sampled_min = rng.choice(min_idx, size=n_min_sample, replace=False)
    sampled_maj = rng.choice(maj_idx, size=n_maj_sample, replace=False)

    indices = np.concatenate([sampled_min, sampled_maj])
    rng.shuffle(indices)

    logger.info(
        "_maybe_subsample: N=%d → %d (min=%d, maj=%d)",
        n_total, len(indices), n_min_sample, n_maj_sample,
    )
    return X[indices], y[indices], True
