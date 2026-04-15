"""Seven complexity dimensions of the CIPA framework (Stage C).

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

from cipa.dimensions.d1_imbalance import compute_d1
from cipa.dimensions.d2_overlap import compute_d2
from cipa.dimensions.d3_hardness import compute_d3
from cipa.dimensions.d4_fragmentation import compute_d4
from cipa.dimensions.d5_dimensionality import compute_d5
from cipa.dimensions.d6_informativeness import compute_d6
from cipa.dimensions.d7_boundary import compute_d7

__all__ = [
    "compute_d1",
    "compute_d2",
    "compute_d3",
    "compute_d4",
    "compute_d5",
    "compute_d6",
    "compute_d7",
]
