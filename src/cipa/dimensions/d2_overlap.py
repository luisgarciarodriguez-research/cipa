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
from time import perf_counter

import numpy as np

from cipa._constants import (
    DEFAULT_D2_WEIGHTS,
    DEFAULT_K,
    DEFAULT_KNN_ALGORITHM,
    DEFAULT_QUERY_CHUNK_SIZE,
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
    n_jobs: int | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    algorithm: str = DEFAULT_KNN_ALGORITHM,
) -> DimensionResult:
    """Compute D2: Class Overlap = alpha·F3 + beta·N1 + gamma·kDN.

    Combines three complementary overlap measures:
    - F3  (Fisher discriminant ratio): feature-range overlap on the most
           discriminative feature.
    - N1  (MST boundary fraction): proportion of instances adjacent to a
           class boundary in the exact Euclidean minimum spanning tree.
    - kDN (k-NN disagreement): fraction of k nearest neighbors with a
           different class label, averaged over all instances. Each instance
           is excluded from its own neighbourhood by index; exact duplicates
           with another label count as disagreeing neighbours.

    All three are computed on the rows of ``dataset``; subsampling, if any, is
    decided by ``CIPAPipeline`` (C6).

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
    n_jobs : int or None
        Parallel jobs for the neighbour queries of N1 and kDN.

    Returns
    -------
    DimensionResult
        value      : D2 ∈ [0, 1]. Higher = more class overlap.
        components : {"F3", "N1", "kDN", "alpha", "beta", "gamma"}
        metadata   : {"k", "component_seconds"}

    ``component_seconds`` times F3, N1 and kDN separately (2.0.0rc3), so a
    bottleneck can be found without instrumenting from outside.

    **The cost of building the neighbour index is charged to kDN**, the first
    component here that touches it, and only when this call is what fits it.
    Under ``CIPAPipeline`` the cache is usually already warm because D3 queried
    it first, so kDN looks much cheaper there than when ``compute_d2`` is
    called on its own. Compare the two figures only within one call path.
    """
    alpha, beta, gamma = weights
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"D2 weights must sum to 1.0, got {sum(weights):.10f}")

    timings: dict[str, float] = {}

    # F3
    t0 = perf_counter()
    F3 = compute_f3(dataset.X, dataset.y)
    timings["F3"] = perf_counter() - t0

    # N1
    t0 = perf_counter()
    N1 = compute_n1(
        dataset.X, dataset.y, n_jobs=n_jobs, chunk_size=chunk_size, algorithm=algorithm
    )
    timings["N1"] = perf_counter() - t0

    # kDN via k-NN cache. Fitting the index, if this call is what triggers it,
    # is charged here; see the note in the docstring.
    t0 = perf_counter()
    if knn_cache is None:
        from cipa._knn import _KNNCache
        knn_cache = _KNNCache(dataset, k=k, n_jobs=n_jobs)

    _, indices = knn_cache.query_all()
    # indices shape: (N, k_actual); use min(k, available) columns
    k_actual = min(k, indices.shape[1])
    neighbor_labels = dataset.y[indices[:, :k_actual]]
    kDN = float(np.mean(neighbor_labels != dataset.y[:, None]))
    timings["kDN"] = perf_counter() - t0

    raw = alpha * F3 + beta * N1 + gamma * kDN
    value = float(np.clip(raw, 0.0, 1.0))
    if abs(raw - value) > 1e-6:
        logger.warning("D2: clipped value %.6f to [0, 1]", raw)

    return DimensionResult(
        value=value,
        dimension_id="D2",
        components={"F3": F3, "N1": N1, "kDN": kDN, "alpha": alpha, "beta": beta, "gamma": gamma},
        metadata={"k": k, "component_seconds": {k_: round(v, 4) for k_, v in timings.items()}},
    )
