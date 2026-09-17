"""D5 — Effective Dimensionality via spectral entropy of PCA eigenvalues.

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
from sklearn.decomposition import PCA

from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d5(dataset: CIPADataset, random_state: int | None = None) -> DimensionResult:
    """Compute D5: Effective Dimensionality via spectral entropy.

    Formula
    -------
    Let λ₁, …, λₖ be the non-zero eigenvalues of the covariance matrix of X
    (k = min(N−1, d)). Define the normalised spectrum:

        pᵢ = λᵢ / Σλⱼ   (explained variance ratios from PCA)

    Then:

        H     = −Σ pᵢ ln(pᵢ)      [Shannon entropy, nats]
        H_max = ln(k)
        D5    = H / H_max          [normalised to [0, 1]]

    Interpretation
    --------------
    - D5 ≈ 0 : virtually all variance in one principal component
               (low effective dimensionality).
    - D5 ≈ 1 : variance spread uniformly across all k components
               (high effective dimensionality, curse-of-dimensionality regime).

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse.
    random_state : int or None
        Forwarded to PCA; only randomized or ARPACK solvers use it, so the
        value does not change the result of the exact solvers chosen here.

    Degenerate cases
    ----------------
    - d = 1 or n ≤ 2 : returns 0.0.
    - k ≤ 1 after filtering near-zero eigenvalues : returns 0.0.

    Returns
    -------
    DimensionResult
        value      : D5 ∈ [0, 1].
        components : {"H_nats", "H_max_nats", "n_components_fit"}
        metadata   : {"d", "n", "top5_explained_variance_ratio"}
    """
    X = dataset.X
    n, d = X.shape

    if d == 1 or n <= 2:
        logger.warning("D5: d=%d, n=%d — returning 0.0 (degenerate).", d, n)
        return DimensionResult(
            value=0.0,
            dimension_id="D5",
            components={"H_nats": 0.0, "H_max_nats": 0.0, "n_components_fit": 0},
            metadata={"d": d, "n": n, "top5_explained_variance_ratio": []},
        )

    n_components = min(n - 1, d)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X - X.mean(axis=0))

    evr     = pca.explained_variance_ratio_
    evr_pos = evr[evr > 1e-12]
    k       = len(evr_pos)

    if k <= 1:
        H = H_max = 0.0
        value = 0.0
    else:
        H     = float(-np.sum(evr_pos * np.log(evr_pos)))
        H_max = float(np.log(k))
        value = float(np.clip(H / H_max, 0.0, 1.0))

    top5 = evr[:5].tolist()
    logger.debug("D5=%.4f  H=%.4f nats  H_max=%.4f nats  k=%d  d=%d",
                 value, H, H_max, k, d)

    return DimensionResult(
        value=value,
        dimension_id="D5",
        components={
            "H_nats":            H,
            "H_max_nats":        H_max,
            "n_components_fit":  k,
        },
        metadata={"d": d, "n": n, "top5_explained_variance_ratio": top5},
    )
