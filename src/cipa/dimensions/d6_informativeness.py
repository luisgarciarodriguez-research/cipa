"""D6 — Feature Informativeness. See §3.1 of García Rodríguez et al. (2026).

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
from sklearn.feature_selection import mutual_info_classif

from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d6(
    dataset: CIPADataset,
    random_state: int | None = None,
) -> DimensionResult:
    """Compute D6: Feature Informativeness.

    Formula:
        D6 = 1 - Ī(X; Y) / H(Y)    [all in nats]

    Returns
    -------
    DimensionResult
        value      : D6 ∈ [0, 1]. Higher = less informative features.
        components : {"H_Y_nats", "I_mean_nats", "mi_scores", "top_3_features"}
        metadata   : {"random_state"}
    """
    p_plus = dataset.n_minority / dataset.N
    p_minus = dataset.n_majority / dataset.N
    # Entropy in nats
    H_Y = float(-(p_plus * np.log(p_plus) + p_minus * np.log(p_minus)))

    if H_Y < 1e-10:
        logger.warning("D6: H(Y) near zero — degenerate label distribution. Returning 1.0.")
        return DimensionResult(
            value=1.0,
            dimension_id="D6",
            components={"H_Y_nats": H_Y, "I_mean_nats": 0.0, "mi_scores": [], "top_3_features": []},
            metadata={"random_state": random_state},
        )

    if H_Y < 0.05:
        logger.warning(
            "D6: H(Y) very small (%.4f nats) — D6 may be unreliable for this dataset.", H_Y
        )

    mi_scores = mutual_info_classif(
        dataset.X, dataset.y, discrete_features=False, random_state=random_state
    )

    I_mean = float(np.clip(np.mean(mi_scores), 0.0, H_Y))
    raw = 1.0 - I_mean / H_Y
    value = float(np.clip(raw, 0.0, 1.0))

    if abs(raw - value) > 1e-6:
        logger.warning("D6: clipped value %.6f to [0, 1]", raw)

    top_3 = np.argsort(mi_scores)[-3:][::-1].tolist()

    return DimensionResult(
        value=value,
        dimension_id="D6",
        components={
            "H_Y_nats": H_Y,
            "I_mean_nats": I_mean,
            "mi_scores": mi_scores.tolist(),
            "top_3_features": top_3,
        },
        metadata={"random_state": random_state},
    )
