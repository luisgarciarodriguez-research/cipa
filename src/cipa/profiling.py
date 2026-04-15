"""Stage P: Complexity Profile and Signature classification.

Converts the seven dimension scores into a structural fingerprint
and classifies it into one of five named signatures.

See §3.3 of García Rodríguez et al. (2026).

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

from cipa._constants import (
    DOMINANCE_MARGIN,
    ELEVATION_THRESHOLD,
    LOW_THRESHOLD,
    SIGNATURE_NAMES,
)
from cipa.types import ComplexityProfile, DifficultyScore

logger = logging.getLogger(__name__)


def compute_profile(
    difficulty_score: DifficultyScore,
    elevation_threshold: float = ELEVATION_THRESHOLD,
    low_threshold: float = LOW_THRESHOLD,
    dominance_margin: float = DOMINANCE_MARGIN,
) -> ComplexityProfile:
    """Classify the Complexity Profile and assign a Signature.

    Parameters
    ----------
    difficulty_score : DifficultyScore
        Must contain exactly 7 dimension results (D1-D7).
    elevation_threshold : float
        A dimension Di is "elevated" when Di >= elevation_threshold.
    low_threshold : float
        A dimension Di is "low" when Di < low_threshold (used for Sig. I).
    dominance_margin : float
        Minimum gap between D5 and D2 for Sig. IV to be assigned.

    Returns
    -------
    ComplexityProfile
        vector             : (D1, ..., D7)
        signature          : "I" | "II" | "III" | "IV" | "V"
        signature_name     : human-readable name
        dominant_dimensions: dimensions with Di >= elevation_threshold
    """
    vector = tuple(d.value for d in difficulty_score.dimensions)
    d1, d2, d3, d4, d5, d6, d7 = vector

    # Priority 0 — Signature I: uniformly low, all non-imbalance dims below low_threshold
    if all(v < low_threshold for v in (d2, d3, d4, d5, d6, d7)):
        sig = "I"

    # Priority 1 - Signature IV: D5 dominates D1-D5, high enough, and leads D2 by margin
    elif (
        d5 == max(d1, d2, d3, d4, d5)
        and d5 > elevation_threshold
        and d5 > d2 + dominance_margin
    ):
        sig = "IV"

    # Priority 2 - Signature III: D4 dominates D1-D5 and exceeds 0.50
    elif d4 == max(d1, d2, d3, d4, d5) and d4 > 0.50:
        sig = "III"

    # Priority 3 — Signature II: D2 clearly elevated and at least as large as D1
    elif d2 > elevation_threshold and d2 >= d1:
        sig = "II"

    # Priority 4 — Signature V: compound (default)
    else:
        sig = "V"

    # Dominant dimensions: all Di >= elevation_threshold, sorted descending
    indexed = [
        (v, f"D{i + 1}") for i, v in enumerate(vector) if v >= elevation_threshold
    ]
    indexed.sort(key=lambda x: -x[0])
    dominant_dims = [name for _, name in indexed]

    logger.debug("Signature %s (%s)", sig, SIGNATURE_NAMES[sig])

    return ComplexityProfile(
        vector=vector,
        signature=sig,
        signature_name=SIGNATURE_NAMES[sig],
        dominant_dimensions=dominant_dims,
    )
