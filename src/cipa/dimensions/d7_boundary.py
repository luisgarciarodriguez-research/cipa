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
from time import perf_counter

import numpy as np

from cipa._constants import (
    DEFAULT_KNN_ALGORITHM,
    DEFAULT_QUERY_CHUNK_SIZE,
    DEFAULT_SVC_MAX_ITER,
    DEFAULT_SVC_TOL,
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
    random_state: int | None = None,
    n_jobs: int | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    algorithm: str = DEFAULT_KNN_ALGORITHM,
    svc_tol: float = DEFAULT_SVC_TOL,
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
        Maximum iterations for LinearSVC. If it is reached, L1 is still the
        training error of the last iterate and ``converged`` is False (C5).
    random_state : int or None
        Seed for LinearSVC.
    n_jobs : int or None
        Parallel jobs for the N2 neighbour queries.
    svc_tol : float
        Stopping tolerance for LinearSVC (2.0.0rc3).

    Returns
    -------
    DimensionResult
        value      : D7 ∈ [0, 1]. Higher = more complex boundary.
        components : {"L1", "N2norm", "N2_raw", "converged", "n_iter"}
        metadata   : {"svc_converged", "svc_max_iter", "svc_tol",
                      "random_state", "component_seconds"}

    ``component_seconds`` times L1 and N2 separately (2.0.0rc3). N2 builds its
    own neighbour indexes and does not use ``knn_cache``, so its figure always
    includes that cost; there is no warm-cache case to confuse it with.
    """
    timings: dict[str, float] = {}

    t0 = perf_counter()
    l1_val, converged, n_iter = compute_l1(
        dataset.X, dataset.y, max_iter=svc_max_iter, random_state=random_state,
        tol=svc_tol,
    )
    timings["L1"] = perf_counter() - t0

    t0 = perf_counter()
    n2norm = compute_n2(
        dataset.X, dataset.y, n_jobs=n_jobs, chunk_size=chunk_size, algorithm=algorithm
    )
    timings["N2"] = perf_counter() - t0

    raw = (l1_val + n2norm) / 2.0
    value = float(np.clip(raw, 0.0, 1.0))
    if abs(raw - value) > 1e-6:
        logger.warning("D7: clipped value %.6f to [0, 1]", raw)

    N2_raw = n2norm / (1.0 - n2norm) if n2norm < 1.0 else float("inf")

    return DimensionResult(
        value=value, dimension_id="D7",
        components={"L1": l1_val, "N2norm": n2norm, "N2_raw": N2_raw,
                    "converged": converged, "n_iter": n_iter},
        metadata={"svc_converged": converged, "svc_max_iter": svc_max_iter,
                  "svc_tol": svc_tol, "random_state": random_state,
                  "component_seconds": {k: round(v, 4) for k, v in timings.items()}},
    )
