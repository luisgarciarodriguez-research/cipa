"""D3 — Instance Hardness (Napierała-Stefanowski typology). See §3.1 of García Rodríguez et al. (2026).

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

from cipa._constants import DEFAULT_K
from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d3(
    dataset: CIPADataset,
    knn_cache: object | None = None,
    k: int = DEFAULT_K,
) -> DimensionResult:
    """Compute D3: Instance Hardness via the Napierała-Stefanowski typology.

    Classifies each minority instance into one of four types based on the
    composition of its k nearest neighbors (drawn from the full dataset):
    - Safe      : majority of neighbors share the minority label (n_same > k/2).
    - Borderline: more than one but not a majority of neighbors are minority.
    - Rare      : exactly one neighbor is minority.
    - Outlier   : no neighbor is minority.

    D3 = (n_borderline + 2·n_rare + 3·n_outlier) / (3·|C+|)

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse.
    knn_cache : _KNNCache or None
        Pre-fitted k-NN cache shared with D2 and D7. If None, a fresh
        cache is built from dataset using k.
    k : int
        Number of neighbors used for type classification. Reduced
        automatically when k >= n_minority.

    Returns
    -------
    DimensionResult
        value      : D3 ∈ [0, 1]. Higher = more hard/outlier minority instances.
        components : {"n_safe", "n_borderline", "n_rare", "n_outlier",
                      "pct_safe", "pct_borderline", "pct_rare", "pct_outlier"}
        metadata   : {"k"}
    """
    n_min = dataset.n_minority

    effective_k = k
    if k >= n_min:
        effective_k = max(1, n_min - 1)
        logger.warning("D3: k=%d >= n_minority=%d; using k=%d", k, n_min, effective_k)

    if knn_cache is None:
        from cipa._knn import _KNNCache
        knn_cache = _KNNCache(dataset, k=effective_k)

    _, indices = knn_cache.query_minority()
    k_actual = min(effective_k, indices.shape[1])

    n_safe = n_borderline = n_rare = n_outlier = 0
    for i in range(n_min):
        neighbor_labels = dataset.y[indices[i, :k_actual]]
        n_same = int(np.sum(neighbor_labels == dataset.minority_label))
        if n_same > k_actual / 2:
            n_safe += 1
        elif n_same > 1:
            n_borderline += 1
        elif n_same == 1:
            n_rare += 1
        else:
            n_outlier += 1

    raw = (n_borderline + 2 * n_rare + 3 * n_outlier) / (3 * n_min)
    value = float(np.clip(raw, 0.0, 1.0))

    return DimensionResult(
        value=value,
        dimension_id="D3",
        components={
            "n_safe": n_safe,
            "n_borderline": n_borderline,
            "n_rare": n_rare,
            "n_outlier": n_outlier,
            "pct_safe": n_safe / n_min,
            "pct_borderline": n_borderline / n_min,
            "pct_rare": n_rare / n_min,
            "pct_outlier": n_outlier / n_min,
        },
        metadata={"k": k_actual},
    )
