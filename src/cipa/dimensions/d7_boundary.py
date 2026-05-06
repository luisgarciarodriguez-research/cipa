"""D7 — Boundary Complexity. See §3.1 of García Rodríguez et al. (2026).

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
    DEFAULT_LARGE_N_SUBSAMPLE,
    DEFAULT_N1_MAX_EXACT,
    DEFAULT_SVC_MAX_ITER,
)
from cipa.dataset import CIPADataset
from cipa.ecol.l1 import compute_l1
from cipa.ecol.n2 import compute_n2
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d7(
    dataset: CIPADataset,
    knn_cache: object | None = None,
    svc_max_iter: int = DEFAULT_SVC_MAX_ITER,
    n2_max_exact: int = DEFAULT_N1_MAX_EXACT,
    n2_subsample_size: int = DEFAULT_LARGE_N_SUBSAMPLE,
    random_state: int | None = None,
) -> DimensionResult:
    """Compute D7: Boundary Complexity = (L1 + N2norm) / 2.

    Combines two complementary boundary difficulty measures:
    - L1  (LinearSVC training error): fraction of training instances
           misclassified by the best linear separator. High L1 indicates
           a non-linear or highly overlapping boundary.
    - N2norm (intra/inter-class distance ratio, normalised):
           N2_raw = Σ intra_dist / Σ inter_dist;
           N2norm = N2_raw / (1 + N2_raw) ∈ [0, 1).
           High N2norm means same-class instances are farther from each other
           than from instances of the opposite class (tangled boundary).

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse.
    knn_cache : _KNNCache or None
        Accepted for API consistency with D2/D3; N2 performs its own
        nearest-neighbor queries and does not use this cache.
    svc_max_iter : int
        Maximum iterations for LinearSVC. Non-convergence triggers a fallback
        L1 = 0.5.
    n2_max_exact : int
        Maximum N for exact N2 computation. Larger datasets are subsampled.
    n2_subsample_size : int
        Subsample size used when N > n2_max_exact.
    random_state : int or None
        Seed for LinearSVC and N2 subsampling.

    Returns
    -------
    DimensionResult
        value      : D7 ∈ [0, 1]. Higher = more complex boundary.
        components : {"L1", "N2norm", "N2_raw"}
        metadata   : {"svc_converged", "random_state"}
    """
    l1_val, converged = compute_l1(dataset.X, dataset.y,
                                   max_iter=svc_max_iter, random_state=random_state)

    n2norm, _n2_sub = compute_n2(
        dataset.X, dataset.y,
        minority_label=dataset.minority_label,
        majority_label=dataset.majority_label,
        max_exact=n2_max_exact,
        subsample_size=n2_subsample_size,
        random_state=random_state,
    )

    raw = (l1_val + n2norm) / 2.0
    value = float(np.clip(raw, 0.0, 1.0))
    if abs(raw - value) > 1e-6:
        logger.warning("D7: clipped value %.6f to [0, 1]", raw)

    N2_raw = n2norm / (1.0 - n2norm) if n2norm < 1.0 else float("inf")

    return DimensionResult(
        value=value, dimension_id="D7",
        components={"L1": l1_val, "N2norm": n2norm, "N2_raw": N2_raw},
        metadata={"svc_converged": converged, "random_state": random_state},
    )
