"""D4 — Sub-concept Fragmentation. See §3.1 of García Rodríguez et al. (2026).

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
) -> DimensionResult:
    """Compute D4 = ECindex · sqrt(n_clusters / |C+|)."""
    X_min = dataset.X_minority
    n_min = dataset.n_minority
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
        nn = NearestNeighbors(n_neighbors=dbscan_min_samples, algorithm="ball_tree").fit(X_min)
        dists, _ = nn.kneighbors(X_min)
        eps = float(np.median(dists[:, -1]))
        if eps == 0.0:
            eps = float(np.mean(dists[:, -1])) + 1e-10
    else:
        eps = float(dbscan_eps)  # type: ignore[arg-type]

    def _run_dbscan(e: float) -> np.ndarray:
        return DBSCAN(eps=e, min_samples=dbscan_min_samples).fit_predict(X_min)

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
