"""Logging configuration for the cipa package.

All modules obtain their logger via:
    import logging
    logger = logging.getLogger(__name__)

Users configure verbosity via:
    import logging
    logging.getLogger("cipa").setLevel(logging.DEBUG)

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

import logging

# Null handler prevents "No handlers could be found" warnings
# when the library is used without any logging configuration.
logging.getLogger("cipa").addHandler(logging.NullHandler())
