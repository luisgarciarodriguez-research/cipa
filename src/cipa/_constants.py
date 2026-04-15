"""Shared constants for the CIPA framework.

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

# Default dimension weights (from paper, Section 3.2)
# w = (w1, w2, w3, w4, w5, w6, w7)
DEFAULT_WEIGHTS: tuple[float, ...] = (0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13)

# Difficulty Score interpretation bands (§3.2)
BAND_LOW = "Low"
BAND_MODERATE = "Moderate"
BAND_HIGH = "High"
BAND_EXTREME = "Extreme"

BAND_THRESHOLDS: tuple[float, float, float] = (0.25, 0.50, 0.75)

# Complexity signatures (§3.3)
SIGNATURE_NAMES: dict[str, str] = {
    "I": "Imbalance-dominated",
    "II": "Overlap-dominated",
    "III": "Fragmented",
    "IV": "Dimensionality-dominated",
    "V": "Compound",
}

# Profiling thresholds (§3.3)
ELEVATION_THRESHOLD: float = 0.55
LOW_THRESHOLD: float = 0.25
DOMINANCE_MARGIN: float = 0.10

# k-NN defaults (§3.1)
DEFAULT_K: int = 5
DEFAULT_N1_MAX_EXACT: int = 50_000
DEFAULT_LARGE_N_SUBSAMPLE: int = 10_000

# DBSCAN defaults (§3.1, D4)
DEFAULT_DBSCAN_MIN_SAMPLES: int = 3

# SVC defaults (§3.1, D7)
DEFAULT_SVC_MAX_ITER: int = 2_000

# D2 sub-weights: alpha (F3), beta (N1), gamma (kDN)
DEFAULT_D2_WEIGHTS: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
