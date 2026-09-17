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

from cipa._constants import DEFAULT_QUERY_CHUNK_SIZE
from cipa.dataset import CIPADataset

logger = logging.getLogger(__name__)


def kneighbors_excluding_self(
    nn: NearestNeighbors,
    X_query: np.ndarray,
    query_indices: np.ndarray,
    k: int,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours of fitted points, excluding each point by index (C3).

    Requests k+1 neighbours and removes the column whose index equals the
    query's own index. Exact duplicates of the query (other rows at distance
    0) are kept as neighbours: a twin with another label is real overlap.
    If the query's own index is not among the k+1 results, which only happens
    when more than k other rows share its distance 0, the last column is
    dropped instead.

    Among neighbours at exactly the same distance, the order is the one
    returned by the scikit-learn tree; it is deterministic for a given X.

    Parameters
    ----------
    nn : NearestNeighbors
        Estimator fitted on the reference matrix that contains the queries.
    X_query : np.ndarray, shape (m, d)
        Query rows.
    query_indices : np.ndarray, shape (m,)
        Row index of each query in the fitted reference matrix.
    k : int
        Neighbours to return per query. Must satisfy k + 1 <= n_fitted.
    chunk_size : int
        Maximum number of queries sent to the tree at once.

    Returns
    -------
    distances : np.ndarray, shape (m, k)
    indices : np.ndarray, shape (m, k)
        Indices into the fitted reference matrix, sorted by distance.
    """
    m = len(X_query)
    distances = np.empty((m, k), dtype=np.float64)
    indices = np.empty((m, k), dtype=np.intp)
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        dist, idx = nn.kneighbors(X_query[start:stop], n_neighbors=k + 1)
        drop = idx == np.asarray(query_indices[start:stop])[:, None]
        drop[~drop.any(axis=1), -1] = True
        # Keep the first occurrence only, in case of repeated indices
        drop &= np.cumsum(drop, axis=1) == 1
        keep = ~drop
        distances[start:stop] = dist[keep].reshape(stop - start, k)
        indices[start:stop] = idx[keep].reshape(stop - start, k)
    return distances, indices


class _KNNCache:
    """Nearest-neighbour index shared by D2 (kDN) and D3 (NS-typology).

    The index is fitted lazily on first query and results are cached for the
    default k. Each instance is excluded from its own neighbourhood by index,
    not by position (see ``kneighbors_excluding_self``).
    """

    def __init__(
        self,
        dataset: CIPADataset,
        k: int,
        algorithm: str = "ball_tree",
        n_jobs: int | None = None,
        chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    ) -> None:
        """Prepare a lazily fitted NearestNeighbors model on the dataset.

        Parameters
        ----------
        dataset : CIPADataset
            Dataset whose feature matrix X is used for fitting.
        k : int
            Number of neighbors to retrieve (excluding self). Clamped to N-1
            if k >= dataset.N.
        algorithm : str
            Neighbor search algorithm passed to sklearn NearestNeighbors.
        n_jobs : int or None
            Parallel jobs for neighbour queries (sklearn convention).
        chunk_size : int
            Maximum number of queries sent to the tree at once.
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= dataset.N:
            k = dataset.N - 1
            logger.warning("_KNNCache: k reduced to %d (= N-1)", k)

        self._dataset = dataset
        self._k = k
        self._algorithm = algorithm
        self._n_jobs = n_jobs
        self._chunk_size = chunk_size
        self._nn: NearestNeighbors | None = None

        self._cache_all: tuple[np.ndarray, np.ndarray] | None = None
        self._cache_minority: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def k(self) -> int:
        """Number of neighbours returned per query (after clamping to N-1)."""
        return self._k

    def _fitted(self) -> NearestNeighbors:
        """Fit the neighbour index on first use and return it."""
        if self._nn is None:
            self._nn = NearestNeighbors(algorithm=self._algorithm, n_jobs=self._n_jobs)
            self._nn.fit(self._dataset.X)
        return self._nn

    def query_all(self) -> tuple[np.ndarray, np.ndarray]:
        """k nearest neighbors for all N instances, excluding self.

        Returns
        -------
        distances : (N, k)
        indices   : (N, k)  — indices into dataset.X
        """
        if self._cache_all is None:
            X = self._dataset.X
            self._cache_all = kneighbors_excluding_self(
                self._fitted(), X, np.arange(len(X)), self._k, self._chunk_size
            )
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
            minority_idx = np.flatnonzero(self._dataset.minority_mask)
            self._cache_minority = kneighbors_excluding_self(
                self._fitted(), self._dataset.X[minority_idx], minority_idx,
                self._k, self._chunk_size,
            )
        return self._cache_minority
