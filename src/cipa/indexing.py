"""Stage I: Difficulty Score computation.

Aggregates D1-D7 into a single normalized Difficulty Score DS in [0, 1]
and assigns an interpretation band.

See §3.2 of García Rodríguez et al. (2026).

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

from cipa._constants import DEFAULT_WEIGHTS
from cipa.types import DifficultyScore, DimensionResult

logger = logging.getLogger(__name__)


def compute_difficulty_score(
    dimension_results: tuple[DimensionResult, ...],
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> DifficultyScore:
    """Compute the Difficulty Score DS = Σ w_i · D_i.

    Parameters
    ----------
    dimension_results : tuple of 7 DimensionResult
        Must be ordered D1, D2, ..., D7.
    weights : tuple of 7 floats
        Must sum to 1.0 ± 1e-9. All values ≥ 0.

    Returns
    -------
    DifficultyScore
        value   : DS ∈ [0, 1]
        band    : "Low" | "Moderate" | "High" | "Extreme"
        weights : the weights used
        dimensions : the input dimension_results

    Raises
    ------
    ValueError
        If dimension_results has wrong length, wrong order, or weights
        are invalid.
    """
    if len(dimension_results) != 7:
        raise ValueError(
            f"dimension_results must have 7 elements, got {len(dimension_results)}"
        )
    for i, dr in enumerate(dimension_results):
        expected = f"D{i + 1}"
        if dr.dimension_id != expected:
            raise ValueError(
                f"dimension_results[{i}].dimension_id must be {expected!r}, "
                f"got {dr.dimension_id!r}"
            )
    if len(weights) != 7:
        raise ValueError(f"weights must have length 7, got {len(weights)}")
    w_sum = sum(weights)
    if abs(w_sum - 1.0) >= 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {w_sum}")
    if any(w < 0 for w in weights):
        raise ValueError("all weights must be >= 0")

    ds = float(np.clip(sum(w * d.value for w, d in zip(weights, dimension_results, strict=False)), 0.0, 1.0))
    logger.debug("DS = %.4f (%s)", ds, classify_band(ds))

    return DifficultyScore(
        value=ds,
        band=classify_band(ds),
        weights=tuple(weights),
        dimensions=tuple(dimension_results),
    )


def classify_band(ds: float) -> str:
    """Map a DS value to its interpretation band.

    Parameters
    ----------
    ds : float
        Difficulty Score value in [0, 1].

    Returns
    -------
    str
        "Low"      if ds < 0.25  — unlikely to be significantly challenging.
        "Moderate" if 0.25 ≤ ds < 0.50  — standard techniques likely sufficient.
        "High"     if 0.50 ≤ ds < 0.75  — specialized strategies needed.
        "Extreme"  if ds ≥ 0.75  — fundamental learning challenges.
    """
    if ds < 0.25:
        return "Low"
    if ds < 0.50:
        return "Moderate"
    if ds < 0.75:
        return "High"
    return "Extreme"
