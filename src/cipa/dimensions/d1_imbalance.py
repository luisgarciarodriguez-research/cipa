"""D1 — Imbalance Distribution. See §3.1 of García Rodríguez et al. (2026).

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

from cipa.dataset import CIPADataset
from cipa.types import DimensionResult

logger = logging.getLogger(__name__)


def compute_d1(dataset: CIPADataset) -> DimensionResult:
    """Compute D1: Imbalance Distribution via normalised binary entropy.

    Formula
    -------
        p⁺ = n_min / N,   p⁻ = n_maj / N
        H(Y) = −p⁺ log₂(p⁺) − p⁻ log₂(p⁻)   [bits, ∈ [0, 1] for binary]
        D1   = 1 − H(Y)

    Interpretation
    --------------
    - D1 = 0  : perfectly balanced classes (H(Y) = 1 bit, maximum entropy).
    - D1 → 1  : extreme imbalance (H(Y) → 0, almost all probability on one class).

    This replaces the log(IR+1) formula, which overestimated D1 for IR < 5 and
    was not grounded in information theory.

    Returns
    -------
    DimensionResult
        value      : D1 ∈ [0, 1]. Higher = more imbalanced.
        components : {"IR", "H_Y_bits"}
        metadata   : {"n_minority", "n_majority", "N"}
    """
    p_plus  = dataset.n_minority / dataset.N
    p_minus = dataset.n_majority / dataset.N

    H_Y = float(-(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus)))
    raw   = 1.0 - H_Y
    value = float(np.clip(raw, 0.0, 1.0))

    if abs(raw - value) > 1e-6:
        logger.warning("D1: clipped value %.6f to [0, 1]", raw)

    logger.debug("D1=%.4f  H_Y=%.4f bits  IR=%.4f", value, H_Y, dataset.IR)

    return DimensionResult(
        value=value,
        dimension_id="D1",
        components={"IR": dataset.IR, "H_Y_bits": H_Y},
        metadata={
            "n_minority": dataset.n_minority,
            "n_majority": dataset.n_majority,
            "N": dataset.N,
        },
    )
