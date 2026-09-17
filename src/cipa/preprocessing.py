"""Feature preprocessing applied once per dataset before any dimension (C2).

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
from typing import Any

import numpy as np

from cipa._constants import SCALING_OPTIONS
from cipa.dataset import CIPADataset

logger = logging.getLogger(__name__)


def preprocess_features(
    X: np.ndarray,
    scaling: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop constant columns and scale the remaining ones.

    Steps
    -----
    1. Columns with σ = 0 are removed. A column is constant when its maximum
       equals its minimum, which is the exact floating-point test for σ = 0
       (a column of identical values can yield σ ≈ 1e-17 through rounding).
    2. Scaling, per column, with statistics computed on all N rows:

       - ``"standard"``: (x − mean) / σ, with σ the population standard
         deviation (``ddof=0``, as ``sklearn.preprocessing.StandardScaler``).
       - ``"robust"``: (x − median) / IQR, with IQR = Q75 − Q25 (linear
         interpolation). Intended only for sensitivity analysis. When a
         column has IQR = 0 (binary or sparse columns) it is divided by its σ
         instead, still centred on the median; σ > 0 because constant columns
         were removed in step 1. The fallback columns are reported.
       - ``"none"``: no scaling (used by the v1.2.1 regression test).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Finite feature matrix.
    scaling : {"standard", "robust", "none"}
        Scaling method.

    Returns
    -------
    X_out : np.ndarray, shape (N, d_used)
        Preprocessed feature matrix (a new array unless nothing changed).
    info : dict
        ``scaling``, ``n_features_in``, ``n_features_used``,
        ``dropped_constant_columns`` (indices into the input columns),
        ``n_dropped_constant_columns`` and, for ``"robust"``,
        ``robust_std_fallback_columns`` (indices into the input columns).

    Raises
    ------
    ValueError
        If ``scaling`` is not a valid option or every column is constant.
    """
    if scaling not in SCALING_OPTIONS:
        raise ValueError(f"scaling must be one of {SCALING_OPTIONS}, got {scaling!r}")

    n_features_in = X.shape[1]
    constant = np.ptp(X, axis=0) == 0
    dropped = np.flatnonzero(constant)
    kept = np.flatnonzero(~constant)
    if kept.size == 0:
        raise ValueError("All feature columns are constant; nothing left to characterise.")
    if dropped.size:
        logger.info("preprocess: dropping %d constant column(s): %s", dropped.size, dropped.tolist())
        X = X[:, kept]

    info: dict[str, Any] = {
        "scaling": scaling,
        "n_features_in": n_features_in,
        "n_features_used": int(kept.size),
        "dropped_constant_columns": dropped.tolist(),
        "n_dropped_constant_columns": int(dropped.size),
    }

    if scaling == "standard":
        X_out = X - X.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            X_out /= X.std(axis=0)
    elif scaling == "robust":
        q25, median, q75 = np.percentile(X, [25, 50, 75], axis=0)
        scale = q75 - q25
        fallback = scale == 0
        if fallback.any():
            scale = np.where(fallback, X.std(axis=0), scale)
            logger.info(
                "preprocess: IQR = 0 in %d column(s); scaled by σ instead", int(fallback.sum())
            )
        X_out = X - median
        X_out /= scale
        info["robust_std_fallback_columns"] = kept[fallback].tolist()
    else:
        X_out = X

    if not np.all(np.isfinite(X_out)):
        raise ValueError("Scaling produced non-finite values; check for near-constant columns.")
    return X_out, info


def preprocess_dataset(
    dataset: CIPADataset,
    scaling: str,
) -> tuple[CIPADataset, dict[str, Any]]:
    """Apply ``preprocess_features`` to a dataset, keeping its labels and name.

    Parameters
    ----------
    dataset : CIPADataset
        Dataset to preprocess (all N rows).
    scaling : {"standard", "robust", "none"}
        Scaling method.

    Returns
    -------
    CIPADataset
        New dataset with the preprocessed feature matrix.
    info : dict
        See ``preprocess_features``.
    """
    X_out, info = preprocess_features(dataset.X, scaling)
    return dataset._with_features(X_out), info
