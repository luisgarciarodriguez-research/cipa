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

logger = logging.getLogger(__name__)


def compute_n2(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int | bool | None = None,
    majority_label: int | bool | None = None,
    max_exact: int = 50_000,
    subsample_size: int = 10_000,
    random_state: int | None = None,
) -> tuple[float, bool]:
    """N2norm ∈ (0, 1]. Higher = more overlap = more difficult.

    N2_raw  = sum(intra_dists) / sum(inter_dists)
    N2norm  = 1 / (1 + N2_raw)

    Returns (n2norm, was_subsampled).
    """
    was_subsampled = False
    if len(X) > max_exact:
        from cipa._knn import _maybe_subsample
        labels = np.unique(y)
        min_lbl = minority_label if minority_label is not None else labels[np.argmin([np.sum(y == lbl) for lbl in labels])]
        maj_lbl = majority_label if majority_label is not None else labels[np.argmax([np.sum(y == lbl) for lbl in labels])]
        X, y, was_subsampled = _maybe_subsample(
            X, y, min_lbl, maj_lbl, max_exact, subsample_size, random_state
        )

    labels = np.unique(y)
    mask0 = y == labels[0]
    mask1 = y == labels[1]
    X0, X1 = X[mask0], X[mask1]

    intra = np.full(len(X), np.inf)
    inter = np.zeros(len(X))

    # Class 0: intra = nearest in X0 (excl. self), inter = nearest in X1
    if len(X0) > 1:
        nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X0)
        d, _ = nn.kneighbors(X0)
        intra[mask0] = d[:, 1]  # skip self
    nn_x1 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X1)
    d, _ = nn_x1.kneighbors(X0)
    inter[mask0] = d[:, 0]

    # Class 1: intra = nearest in X1 (excl. self), inter = nearest in X0
    if len(X1) > 1:
        nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X1)
        d, _ = nn.kneighbors(X1)
        intra[mask1] = d[:, 1]
    nn_x0 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X0)
    d, _ = nn_x0.kneighbors(X1)
    inter[mask1] = d[:, 0]

    finite = np.isfinite(intra)
    sum_intra = float(np.sum(intra[finite]))
    sum_inter = float(np.sum(inter))

    if sum_inter == 0.0:
        logger.warning("N2: sum(inter_dists) = 0 (perfect separation). Returning N2norm=0.")
        return 0.0, was_subsampled

    N2_raw = sum_intra / sum_inter
    # Difficulty-oriented: separated classes → N2_raw small → N2norm small (easy)
    # Formula: N2norm = N2_raw / (1 + N2_raw)
    # Note: N2_raw / (1 + N2_raw) maps [0, ∞) → [0, 1); higher values indicate harder boundaries.
    N2norm = N2_raw / (1.0 + N2_raw)
    return float(N2norm), was_subsampled
