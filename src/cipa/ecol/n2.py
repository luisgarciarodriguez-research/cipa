"""ECoL N2: Intra/Inter-class Distance Ratio. See §3.1 of García Rodríguez et al. (2026).

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

logger = logging.getLogger(__name__)


def compute_n2(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> float:
    """Compute N2: Intra/Inter-class Distance Ratio, normalised (ECoL measure).

    For each instance, computes its distance to the nearest same-class neighbor
    (intra) and to the nearest opposite-class neighbor (inter). Aggregates and
    normalises via:

        N2_raw  = Σ intra_dist / Σ inter_dist
        N2norm  = N2_raw / (1 + N2_raw)  ∈ [0, 1)

    Low N2norm indicates well-separated classes (inter >> intra, easy boundary);
    high N2norm indicates tangled classes (intra ≈ inter or intra > inter).
    Returns 0.0 when all inter-class distances are zero.

    The instance itself is excluded from its intra-class search by index, so
    an exact duplicate in the same class counts as a neighbour at distance 0
    (C3). No subsampling happens here: the pipeline decides which rows N2
    sees (C6).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Binary label vector.
    n_jobs : int or None
        Parallel jobs for the neighbour queries.
    chunk_size : int
        Maximum number of queries sent to the tree at once.

    Returns
    -------
    float
        N2norm ∈ [0, 1). Higher = more class overlap = harder boundary.
    """
    from cipa._knn import kneighbors_excluding_self

    y = np.asarray(y)
    labels = np.unique(y)
    intra = np.full(len(X), np.inf)
    inter = np.zeros(len(X))

    for own, other in ((labels[0], labels[1]), (labels[1], labels[0])):
        own_idx = np.flatnonzero(y == own)
        X_own = X[own_idx]
        X_other = X[y == other]
        if len(X_own) > 1:
            nn = NearestNeighbors(algorithm="ball_tree", n_jobs=n_jobs).fit(X_own)
            d, _ = kneighbors_excluding_self(
                nn, X_own, np.arange(len(X_own)), 1, chunk_size
            )
            intra[own_idx] = d[:, 0]
        nn_other = NearestNeighbors(algorithm="ball_tree", n_jobs=n_jobs).fit(X_other)
        for start in range(0, len(X_own), chunk_size):
            d, _ = nn_other.kneighbors(X_own[start:start + chunk_size], n_neighbors=1)
            inter[own_idx[start:start + chunk_size]] = d[:, 0]

    finite = np.isfinite(intra)
    sum_intra = float(np.sum(intra[finite]))
    sum_inter = float(np.sum(inter))

    if sum_inter == 0.0:
        logger.warning("N2: sum(inter_dists) = 0 (all instances overlap exactly). Returning N2norm=0.")
        return 0.0

    N2_raw = sum_intra / sum_inter
    # Difficulty-oriented: separated classes → N2_raw small → N2norm small (easy)
    # Formula: N2norm = N2_raw / (1 + N2_raw)
    # Note: N2_raw / (1 + N2_raw) maps [0, ∞) → [0, 1); higher values indicate harder boundaries.
    N2norm = N2_raw / (1.0 + N2_raw)
    return float(N2norm)
