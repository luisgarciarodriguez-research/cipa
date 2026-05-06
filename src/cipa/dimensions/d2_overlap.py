"""D2 — Class Overlap. See §3.1 of García Rodríguez et al. (2026).

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

from cipa._constants import (
    DEFAULT_D2_WEIGHTS,
    DEFAULT_K,
    DEFAULT_LARGE_N_SUBSAMPLE,
    DEFAULT_N1_MAX_EXACT,
)
from cipa.dataset import CIPADataset
from cipa.ecol.f3 import compute_f3
from cipa.ecol.n1 import compute_n1
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d2(
    dataset: CIPADataset,
    knn_cache: object | None = None,
    k: int = DEFAULT_K,
    weights: tuple[float, float, float] = DEFAULT_D2_WEIGHTS,
    n1_max_exact: int = DEFAULT_N1_MAX_EXACT,
    n1_subsample_size: int = DEFAULT_LARGE_N_SUBSAMPLE,
    random_state: int | None = None,
) -> DimensionResult:
    """Compute D2: Class Overlap = alpha·F3 + beta·N1 + gamma·kDN.

    Combines three complementary overlap measures:
    - F3  (Fisher discriminant ratio): feature-range overlap on the most
           discriminative feature.
    - N1  (MST boundary fraction): proportion of instances adjacent to a
           class boundary in the minimum spanning tree.
    - kDN (k-NN disagreement): fraction of k nearest neighbors with a
           different class label, averaged over all instances.

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse.
    knn_cache : _KNNCache or None
        Pre-fitted k-NN cache shared with D3 and D7. If None, a fresh
        cache is built from dataset using k.
    k : int
        Number of neighbors for kDN. Ignored when knn_cache is provided.
    weights : tuple of 3 floats (alpha, beta, gamma)
        Weights for (F3, N1, kDN). Must sum to 1.
    n1_max_exact : int
        Maximum N for exact MST computation. Larger datasets are subsampled.
    n1_subsample_size : int
        Subsample size used when N > n1_max_exact.
    random_state : int or None
        Seed for subsampling reproducibility.

    Returns
    -------
    DimensionResult
        value      : D2 ∈ [0, 1]. Higher = more class overlap.
        components : {"F3", "N1", "kDN", "alpha", "beta", "gamma"}
        metadata   : {"k", "n1_subsampled"}
    """
    alpha, beta, gamma = weights
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"D2 weights must sum to 1.0, got {sum(weights):.10f}")

    # F3
    F3 = compute_f3(dataset.X, dataset.y)

    # N1
    N1, n1_sub = compute_n1(
        dataset.X, dataset.y,
        minority_label=dataset.minority_label,
        majority_label=dataset.majority_label,
        max_exact=n1_max_exact,
        subsample_size=n1_subsample_size,
        random_state=random_state,
    )

    # kDN via k-NN cache
    if knn_cache is None:
        from cipa._knn import _KNNCache
        knn_cache = _KNNCache(dataset, k=k)

    _, indices = knn_cache.query_all()
    # indices shape: (N, k_actual); use min(k, available) columns
    k_actual = min(k, indices.shape[1])
    neighbor_labels = dataset.y[indices[:, :k_actual]]
    kDN = float(np.mean(neighbor_labels != dataset.y[:, None]))

    raw = alpha * F3 + beta * N1 + gamma * kDN
    value = float(np.clip(raw, 0.0, 1.0))
    if abs(raw - value) > 1e-6:
        logger.warning("D2: clipped value %.6f to [0, 1]", raw)

    return DimensionResult(
        value=value,
        dimension_id="D2",
        components={"F3": F3, "N1": N1, "kDN": kDN, "alpha": alpha, "beta": beta, "gamma": gamma},
        metadata={"k": k, "n1_subsampled": n1_sub},
    )
