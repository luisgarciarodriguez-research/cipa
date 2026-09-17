"""D5 — Effective Dimensionality relative to the minority class.

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

from cipa._constants import D5_VARIANCE_THRESHOLD
from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d5(dataset: CIPADataset, random_state: int | None = None) -> DimensionResult:
    """Compute D5: effective dimensionality relative to the minority class size.

    Formula
    -------
    Fit PCA with k = min(N−1, d) components and let p₁ ≥ … ≥ pₖ be the
    explained variance ratios. Then:

        r_95 = min{ r : p₁ + … + p_r ≥ 0.95 }   [effective dimensionality]
        ρ    = r_95 / |C₊|
        D5   = ρ / (1 + ρ)                        [∈ [0, 1); 0.5 when r_95 = |C₊|]

    ρ/(1+ρ) is the same transformation D7 applies to N2. r_95 is bounded by
    min(N−1, d), so with d > N (e.g. gene expression) the cap is N−1.

    ``r_95`` is ``searchsorted(cumsum(p), 0.95) + 1`` with no tolerance: a
    cumulative sum that rounding leaves just below 0.95 counts one more
    component. If the cumulative sum never reaches 0.95 (the ratios sum to
    slightly less than 1), r_95 is capped at the number of fitted components.

    Informative component (does not enter the value)
    ------------------------------------------------
    The D5 of cipa 1.x, the normalised spectral entropy of the same spectrum,
    is kept for comparison with COMIA 2026:

        H = −Σ pᵢ ln(pᵢ),  H_max = ln(k'),  spectral_entropy_norm = H / H_max

    over the k' ratios above 1e-12 (0 when k' ≤ 1).

    Interpretation
    --------------
    - D5 ≈ 0 : few effective dimensions per minority instance.
    - D5 > 0.5 : more effective dimensions than minority instances; the
                 minority concept is under-sampled in the feature space.

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to analyse. The pipeline passes the preprocessed matrix
        (constant columns dropped, scaled) with all N rows.
    random_state : int or None
        Forwarded to PCA; only randomized or ARPACK solvers use it, so the
        value does not change the result of the exact solvers chosen here.

    Degenerate cases
    ----------------
    - d = 1 or n ≤ 2 : returns 0.0.
    - A single non-zero component is not a special case: r_95 = 1 and
      D5 = 1/(1 + |C₊|).

    Returns
    -------
    DimensionResult
        value      : D5 ∈ [0, 1).
        components : {"r_95", "n_minority", "rho", "H_nats", "H_max_nats",
                      "spectral_entropy_norm", "n_components_fit"}
        metadata   : {"d", "n", "top5_explained_variance_ratio",
                      "variance_threshold"}
    """
    X = dataset.X
    n, d = X.shape
    n_minority = int(dataset.n_minority)
    metadata = {"d": d, "n": n, "variance_threshold": D5_VARIANCE_THRESHOLD}

    if d == 1 or n <= 2:
        logger.warning("D5: d=%d, n=%d — returning 0.0 (degenerate).", d, n)
        return DimensionResult(
            value=0.0,
            dimension_id="D5",
            components={
                "r_95": 0, "n_minority": n_minority, "rho": 0.0,
                "H_nats": 0.0, "H_max_nats": 0.0, "spectral_entropy_norm": 0.0,
                "n_components_fit": 0,
            },
            metadata={**metadata, "top5_explained_variance_ratio": []},
        )

    n_components = min(n - 1, d)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X - X.mean(axis=0))

    evr = pca.explained_variance_ratio_
    r_95 = min(int(np.searchsorted(np.cumsum(evr), D5_VARIANCE_THRESHOLD)) + 1, len(evr))
    rho = r_95 / n_minority
    value = float(np.clip(rho / (1.0 + rho), 0.0, 1.0))

    # Spectral entropy (the 1.x D5), reported as an informative component
    evr_pos = evr[evr > 1e-12]
    k       = len(evr_pos)
    if k <= 1:
        H = H_max = 0.0
        entropy_norm = 0.0
    else:
        H            = float(-np.sum(evr_pos * np.log(evr_pos)))
        H_max        = float(np.log(k))
        entropy_norm = float(np.clip(H / H_max, 0.0, 1.0))

    top5 = evr[:5].tolist()
    logger.debug("D5=%.4f  r_95=%d  |C+|=%d  spectral_entropy_norm=%.4f  d=%d",
                 value, r_95, n_minority, entropy_norm, d)

    return DimensionResult(
        value=value,
        dimension_id="D5",
        components={
            "r_95":                  r_95,
            "n_minority":            n_minority,
            "rho":                   float(rho),
            "H_nats":                H,
            "H_max_nats":            H_max,
            "spectral_entropy_norm": entropy_norm,
            "n_components_fit":      k,
        },
        metadata={**metadata, "top5_explained_variance_ratio": top5},
    )
