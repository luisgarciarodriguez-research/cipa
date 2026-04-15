"""ECoL N1: Fraction of Borderline Points (MST-based). See §3.1 of García Rodríguez et al. (2026).

This module is part of the CIPA software package, companion implementation to:

    García Rodríguez, L., Neme Castillo, J. A., & Gómez Adorno, H. M. (2026).
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def compute_n1(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int | bool | None = None,
    majority_label: int | bool | None = None,
    max_exact: int = 50_000,
    subsample_size: int = 10_000,
    random_state: int | None = None,
) -> tuple[float, bool]:
    """N1 ∈ [0, 1]. Higher = more borderline instances = more overlap.

    Returns (n1, was_subsampled).
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
        logger.info("N1: subsampled to N=%d", len(X))

    dist_sq = squareform(pdist(X, metric="euclidean"))
    mst = minimum_spanning_tree(csr_matrix(dist_sq)).tocoo()

    borderline: set[int] = set()
    for i, j in zip(mst.row, mst.col, strict=False):
        if y[i] != y[j]:
            borderline.add(i)
            borderline.add(j)

    return float(len(borderline) / len(X)), was_subsampled
