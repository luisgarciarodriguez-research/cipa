"""D4 — Sub-concept Fragmentation. See §3.1 of García Rodríguez et al. (2026).

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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from cipa._constants import DEFAULT_DBSCAN_MIN_SAMPLES
from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d4(
    dataset: CIPADataset,
    dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    dbscan_eps: float | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> DimensionResult:
    """Compute D4: Sub-concept Fragmentation.

    Runs DBSCAN on the minority class to identify sub-regions (clusters) and
    noise points. Combines the Error Concentration Index (ECindex) with the
    cluster count relative to minority class size:

        D4 = ECindex · sqrt(n_clusters / |C+|)

    ECindex = 1 − sqrt(Σ pᵢ²), where pᵢ is the proportion of each cluster;
    high values indicate uneven size distribution across sub-concepts.
    Noise points are each treated as a singleton cluster of size 1.

    Adaptive eps: when dbscan_eps is None, eps is set to the median distance
    from each minority instance to its min_samples-th neighbor. If all
    instances are classified as noise at that eps, it is doubled once.

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse. Only the minority class X_minority is clustered.
    dbscan_min_samples : int
        DBSCAN min_samples parameter.
    dbscan_eps : float or None
        DBSCAN eps. None triggers adaptive eps estimation from the minority
        k-distance distribution.
    random_state : int or None
        Currently unused; DBSCAN is deterministic. Kept for API symmetry.
    n_jobs : int or None
        Parallel jobs for the neighbour queries of the eps estimate and DBSCAN.

    Returns
    -------
    DimensionResult
        value      : D4 ∈ [0, 1]. Higher = more fragmented minority concept.
        components : {"ECindex", "n_clusters", "n_true_clusters",
                      "n_noise_points", "cluster_sizes"}
        metadata   : {"eps_used", "min_samples", "eps_adaptive"}
    """
    return compute_d4_from_minority(
        dataset.X_minority,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_eps=dbscan_eps,
        n_jobs=n_jobs,
    )


def compute_d4_from_minority(
    X_min: np.ndarray,
    dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    dbscan_eps: float | None = None,
    n_jobs: int | None = None,
) -> DimensionResult:
    """Compute D4 directly on minority-class rows.

    Same computation as ``compute_d4``; used by ``CIPAPipeline`` to run D4 on
    subsamples of the minority class when |C+| > n_max (C6).

    Parameters
    ----------
    X_min : np.ndarray, shape (n_minority, d)
        Minority-class feature rows.
    dbscan_min_samples : int
        DBSCAN min_samples parameter.
    dbscan_eps : float or None
        DBSCAN eps. None triggers adaptive eps estimation.
    n_jobs : int or None
        Parallel jobs for the neighbour queries of the eps estimate and DBSCAN.

    Returns
    -------
    DimensionResult
        See ``compute_d4``.
    """
    n_min = len(X_min)
    eps_adaptive = dbscan_eps is None

    if n_min < dbscan_min_samples:
        logger.warning("D4: n_minority=%d < min_samples=%d. Returning D4=0.", n_min, dbscan_min_samples)
        return DimensionResult(
            value=0.0, dimension_id="D4",
            components={"ECindex": 0.0, "n_clusters": 1, "n_true_clusters": 1,
                        "n_noise_points": 0, "cluster_sizes": [n_min]},
            metadata={"eps_used": 0.0, "min_samples": dbscan_min_samples,
                      "eps_adaptive": eps_adaptive, "degenerate": True},
        )

    # Adaptive eps: median distance to min_samples-th neighbor
    if eps_adaptive:
        nn = NearestNeighbors(
            n_neighbors=dbscan_min_samples, algorithm="ball_tree", n_jobs=n_jobs
        ).fit(X_min)
        dists, _ = nn.kneighbors(X_min)
        eps = float(np.median(dists[:, -1]))
        if eps == 0.0:
            eps = float(np.mean(dists[:, -1])) + 1e-10
    else:
        eps = float(dbscan_eps)  # type: ignore[arg-type]

    def _run_dbscan(e: float) -> np.ndarray:
        """Run DBSCAN on the minority feature matrix with the given eps; return cluster labels."""
        return DBSCAN(eps=e, min_samples=dbscan_min_samples, n_jobs=n_jobs).fit_predict(X_min)

    labels = _run_dbscan(eps)

    if np.all(labels == -1):
        eps_doubled = eps * 2.0
        logger.warning("D4: all minority points noise at eps=%.4f. Retrying eps=%.4f.", eps, eps_doubled)
        labels = _run_dbscan(eps_doubled)
        eps = eps_doubled

        if np.all(labels == -1):
            logger.warning("D4: still all noise. Treating all as one cluster.")
            return DimensionResult(
                value=0.0, dimension_id="D4",
                components={"ECindex": 0.0, "n_clusters": 1, "n_true_clusters": 0,
                            "n_noise_points": n_min, "cluster_sizes": [n_min]},
                metadata={"eps_used": eps, "min_samples": dbscan_min_samples, "eps_adaptive": eps_adaptive},
            )

    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    cluster_ids = [lb for lb in np.unique(labels) if lb >= 0]
    n_true = len(cluster_ids)
    true_sizes = [int(np.sum(labels == c)) for c in cluster_ids]
    all_sizes = true_sizes + [1] * n_noise
    n_clusters = n_true + n_noise

    proportions = [s / n_min for s in all_sizes]
    ECindex = float(np.clip(1.0 - np.sqrt(sum(p * p for p in proportions)), 0.0, 1.0))

    raw = ECindex * float(np.sqrt(n_clusters / n_min))
    value = float(np.clip(raw, 0.0, 1.0))
    if abs(raw - value) > 1e-6:
        logger.warning("D4: clipped value %.6f to [0, 1]", raw)

    return DimensionResult(
        value=value, dimension_id="D4",
        components={"ECindex": ECindex, "n_clusters": n_clusters,
                    "n_true_clusters": n_true, "n_noise_points": n_noise,
                    "cluster_sizes": sorted(all_sizes, reverse=True)},
        metadata={"eps_used": eps, "min_samples": dbscan_min_samples, "eps_adaptive": eps_adaptive},
    )
