"""Stage P: Complexity Profile and Signature classification.

Converts the seven dimension scores into a structural fingerprint
and classifies it into one of five named signatures.

See §3.3 of García Rodríguez et al. (2026).

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

from cipa._constants import (
    SIGNATURE_CANDIDATES,
    SIGNATURE_NAMES,
    SIGNATURE_TAU,
    SIGNATURE_TAU_PRIME,
)
from cipa.types import ComplexityProfile, DifficultyScore

logger = logging.getLogger(__name__)


def compute_profile(
    difficulty_score: DifficultyScore,
    tau: float = SIGNATURE_TAU,
    tau_prime: float = SIGNATURE_TAU_PRIME,
) -> ComplexityProfile:
    """Classify the Complexity Profile and assign a Signature (§3.3, C7).

    Dominance rule (paper, §3.3)
    ----------------------------
    D_i dominates P if D_i > tau and D_i = max{D1, D2, D4, D5}.

    - Signature I:   D1 dominates.
    - Signature II:  D2 dominates.
    - Signature III: D4 dominates.
    - Signature IV:  D5 dominates.
    - Signature V:   no dimension dominates.

    If several candidates share the maximum and it exceeds tau, the first in
    the order D1 > D2 > D4 > D5 dominates. A value equal to tau does not
    dominate. D3, D6 and D7 never dominate, whatever their value.

    Signature V qualifier (proposal, pending confirmation by the author)
    --------------------------------------------------------------------
    The paper defines V as "no D_i dominates and ≥ 2 dimensions exceed
    tau_prime" or "all D_i < tau_prime", leaving uncovered the case in which
    no dimension dominates and exactly one exceeds tau_prime. Here V is the
    residual signature (no dimension dominates) and carries a qualifier that
    never changes the signature, counted over all seven dimensions:

    - ``"compound"``: ≥ 2 dimensions > tau_prime;
    - ``"single"``: exactly one dimension > tau_prime;
    - ``"low"``: every dimension ≤ tau_prime.

    Parameters
    ----------
    difficulty_score : DifficultyScore
        Must contain exactly 7 dimension results (D1-D7).
    tau : float
        Dominance threshold (paper: 0.50).
    tau_prime : float
        Threshold for the Signature V qualifier (paper: 0.35).

    Returns
    -------
    ComplexityProfile
        vector              : (D1, ..., D7)
        signature           : "I" | "II" | "III" | "IV" | "V"
        signature_name      : human-readable name
        dominant_dimension  : dominating dimension, or None for V
        qualifier           : "compound" | "single" | "low" for V, else None
        active_dimensions   : dimensions with Di > tau, descending
        elevated_dimensions : dimensions with Di > tau_prime, descending
    """
    vector = tuple(d.value for d in difficulty_score.dimensions)
    values = {f"D{i + 1}": v for i, v in enumerate(vector)}

    peak = max(values[dim] for dim, _ in SIGNATURE_CANDIDATES)
    dominant = None
    sig = "V"
    if peak > tau:
        for dim, candidate_sig in SIGNATURE_CANDIDATES:
            if values[dim] == peak:
                dominant, sig = dim, candidate_sig
                break

    def _above(threshold: float) -> list[str]:
        """Dimension IDs with value > threshold, by descending value (stable by ID)."""
        hits = [dim for dim, v in values.items() if v > threshold]
        return sorted(hits, key=lambda dim: -values[dim])

    active = _above(tau)
    elevated = _above(tau_prime)

    qualifier = None
    if sig == "V":
        qualifier = "compound" if len(elevated) >= 2 else "single" if elevated else "low"

    logger.debug("Signature %s (%s), qualifier=%s", sig, SIGNATURE_NAMES[sig], qualifier)

    return ComplexityProfile(
        vector=vector,
        signature=sig,
        signature_name=SIGNATURE_NAMES[sig],
        dominant_dimension=dominant,
        qualifier=qualifier,
        active_dimensions=active,
        elevated_dimensions=elevated,
        tau=tau,
        tau_prime=tau_prime,
    )
