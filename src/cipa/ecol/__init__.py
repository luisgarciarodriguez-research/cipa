"""Python implementations of ECoL complexity measures used by CIPA.

Re-exports the four measure functions for convenient import.

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

from cipa.ecol.f3 import compute_f3
from cipa.ecol.l1 import compute_l1
from cipa.ecol.n1 import compute_n1
from cipa.ecol.n2 import compute_n2

__all__ = ["compute_f3", "compute_l1", "compute_n1", "compute_n2"]
