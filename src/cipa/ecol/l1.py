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

from cipa._constants import DEFAULT_SVC_MAX_ITER, DEFAULT_SVC_TOL

logger = logging.getLogger(__name__)


def compute_l1(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = DEFAULT_SVC_MAX_ITER,
    random_state: int | None = None,
    tol: float = DEFAULT_SVC_TOL,
) -> tuple[float, bool, int]:
    """Compute L1: Non-linearity of the linear classifier (ECoL measure).

    Trains a LinearSVC with balanced class weights on the given data and
    measures its training error. A high error rate indicates that no linear
    hyperplane can separate the classes well, implying a complex or non-linear
    decision boundary.

    The error rate of the fitted model is always returned (C5). If LinearSVC
    stops at ``max_iter`` without converging, the error of that last iterate
    is still used and ``converged`` is False; there is no fixed fallback value.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Binary label vector.
    max_iter : int
        Maximum number of iterations for LinearSVC.
    random_state : int or None
        Seed for LinearSVC reproducibility.
    tol : float
        Stopping tolerance for LinearSVC (2.0.0rc3). Exposed because on the
        study's hardest subsample the fit reaches ``max_iter`` without
        converging, which makes the tolerance part of what the reported value
        means rather than an implementation detail.

    Returns
    -------
    l1 : float
        L1 ∈ [0, 1]. Higher = more non-linear boundary.
    converged : bool
        True if LinearSVC converged within max_iter.
    n_iter : int
        Iterations actually taken (2.0.0rc3). Equal to ``max_iter`` when the
        fit stopped at the cap, which is how a consumer can tell an incomplete
        fit from a converged one and say so in a manuscript.
    """
    svc = LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        dual="auto",
        tol=tol,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svc.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    if not converged:
        logger.warning(
            "L1: LinearSVC did not converge after %d iterations; "
            "using the training error of the last iterate.",
            max_iter,
        )

    n_iter_array = np.asarray(svc.n_iter_).ravel()
    n_iter = int(n_iter_array[0]) if n_iter_array.size else 0

    error_rate = 1.0 - float(accuracy_score(y, svc.predict(X)))
    return float(np.clip(error_rate, 0.0, 1.0)), converged, n_iter
