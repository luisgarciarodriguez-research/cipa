"""ECoL L1: Non-linearity of Linear Classifier. See §3.1 of García Rodríguez et al. (2026).

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
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def compute_l1(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 2_000,
    random_state: int | None = None,
) -> tuple[float, bool]:
    """L1 ∈ [0, 1]. Higher = more non-linear boundary.

    Returns (l1, converged). Falls back to 0.5 if SVC does not converge.
    """
    svc = LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        dual="auto",
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svc.fit(X, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

    if not converged:
        logger.warning(
            "L1: LinearSVC did not converge after %d iterations. Using fallback L1=0.5.",
            max_iter,
        )
        return 0.5, False

    error_rate = 1.0 - float(accuracy_score(y, svc.predict(X)))
    return float(np.clip(error_rate, 0.0, 1.0)), True
