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

# Profiling rule (§3.3): D_i dominates if D_i > tau and D_i = max{D1, D2, D4, D5}.
# The order of this tuple is the tie-break order when several share the maximum.
SIGNATURE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("D1", "I"),
    ("D2", "II"),
    ("D4", "III"),
    ("D5", "IV"),
)
SIGNATURE_TAU: float = 0.50
SIGNATURE_TAU_PRIME: float = 0.35
SIGNATURE_V_QUALIFIERS: tuple[str, ...] = ("compound", "single", "low")

# Reproducibility (C9): the pipeline never runs with random_state=None
DEFAULT_RANDOM_STATE: int = 42

# Preprocessing (C2)
SCALING_OPTIONS: tuple[str, ...] = ("standard", "robust", "none")
DEFAULT_SCALING: str = "standard"

# Per-dimension subsampling protocol (C6)
DEFAULT_N_MAX: int = 50_000
DEFAULT_N_SUBSAMPLES: int = 5

# k-NN defaults (§3.1)
DEFAULT_K: int = 5
DEFAULT_QUERY_CHUNK_SIZE: int = 65_536

# Exact Euclidean MST for N1 (C4): neighbours precomputed per instance
DEFAULT_N1_NEIGHBORS: int = 16

# D5 (2.0.0rc2): fraction of PCA variance that defines the effective dimensionality r_95
D5_VARIANCE_THRESHOLD: float = 0.95

# DBSCAN defaults (§3.1, D4)
DEFAULT_DBSCAN_MIN_SAMPLES: int = 3

# SVC defaults (§3.1, D7)
DEFAULT_SVC_MAX_ITER: int = 10_000
# Exposed in 2.0.0rc3 alongside the iteration cap: on the study's hardest
# subsample the fit stops at the cap without converging, so the tolerance has
# to be visible rather than buried in scikit-learn's default.
DEFAULT_SVC_TOL: float = 1e-4

# D2 sub-weights: alpha (F3), beta (N1), gamma (kDN)
DEFAULT_D2_WEIGHTS: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

# Neighbour search algorithm (2.0.0rc3). "auto" applies the rule in
# ``cipa._knn.select_knn_algorithm``: kd_tree up to KD_TREE_MAX_DIM features,
# brute force above it. The threshold keeps the rule n1 already used for the
# minimum spanning tree; brute replaces ball_tree above it because it measured
# 20x faster on the study's hardest subsample with identical neighbours.
KNN_ALGORITHM_OPTIONS: tuple[str, ...] = ("auto", "ball_tree", "kd_tree", "brute")
DEFAULT_KNN_ALGORITHM: str = "auto"
KD_TREE_MAX_DIM: int = 15
