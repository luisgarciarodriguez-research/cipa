"""ECoL F3: Maximum Fisher Discriminant Ratio. See §3.1 of García Rodríguez et al. (2026).

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

logger = logging.getLogger(__name__)


def compute_f3(X: np.ndarray, y: np.ndarray) -> float:
    """F3 ∈ [0, 1]. Lower = more separable on best feature."""
    labels = np.unique(y)
    X0, X1 = X[y == labels[0]], X[y == labels[1]]
    N = len(X)
    min_overlap_count = N

    for j in range(X.shape[1]):
        lo = max(X0[:, j].min(), X1[:, j].min())
        hi = min(X0[:, j].max(), X1[:, j].max())
        if hi < lo:
            return 0.0  # perfect separation on feature j
        count = int(np.sum((X[:, j] >= lo) & (X[:, j] <= hi)))
        min_overlap_count = min(min_overlap_count, count)

    return float(min_overlap_count) / N
