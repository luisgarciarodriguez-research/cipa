"""CIPADataset: validated binary classification dataset. See §3 of García Rodríguez et al. (2026).

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

import numpy as np

logger = logging.getLogger(__name__)


class CIPADataset:
    """A validated binary classification dataset for CIPA analysis.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix. Converted to float64. Must have no NaN or Inf.
    y : np.ndarray, shape (N,)
        Binary label vector. Exactly two unique values.
    minority_label : int or bool
        Label identifying C+ (the minority class).
    majority_label : int or bool
        Label identifying C- (the majority class).
    name : str or None
        Optional human-readable name.

    Raises
    ------
    ValueError
        If the input data fails validation (X shape, NaN/Inf, label count, or class sizes).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        minority_label: int | bool,
        majority_label: int | bool,
        name: str | None = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if X.shape[1] < 1:
            raise ValueError("X must have at least 1 feature (d >= 1)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length: {len(X)} != {len(y)}"
            )

        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(
                f"y must have exactly 2 unique values, got {len(unique_labels)}: {unique_labels}"
            )
        if minority_label not in unique_labels:
            raise ValueError(
                f"minority_label {minority_label!r} not found in y (values: {unique_labels})"
            )
        if majority_label not in unique_labels:
            raise ValueError(
                f"majority_label {majority_label!r} not found in y (values: {unique_labels})"
            )
        if minority_label == majority_label:
            raise ValueError("minority_label and majority_label must be different")

        minority_mask = y == minority_label
        n_minority = int(minority_mask.sum())
        n_majority = int((~minority_mask).sum())

        if n_minority < 2:
            raise ValueError(f"n_minority must be >= 2, got {n_minority}")
        if len(X) < 10:
            raise ValueError(f"Dataset must have N >= 10 instances, got {len(X)}")

        self._X = X
        self._y = y
        self._minority_label = minority_label
        self._majority_label = majority_label
        self.name = name
        self._minority_mask = minority_mask
        self._majority_mask = ~minority_mask
        self._n_minority = n_minority
        self._n_majority = n_majority

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        name: str | None = None,
    ) -> CIPADataset:
        """Convenience constructor: auto-detects minority label by frequency.

        The minority class is the less frequent class.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Feature matrix.
        y : np.ndarray, shape (N,)
            Binary label vector. Exactly two unique values.
        name : str or None
            Optional human-readable name for the dataset.

        Returns
        -------
        CIPADataset

        Raises
        ------
        ValueError
            If y does not have exactly two unique values, or both classes have
            equal frequency (ambiguous minority detection).
        """
        y_arr = np.asarray(y)
        unique, counts = np.unique(y_arr, return_counts=True)
        if len(unique) != 2:
            raise ValueError(
                f"y must have exactly 2 unique values, got {len(unique)}"
            )
        if counts[0] == counts[1]:
            raise ValueError(
                "Cannot auto-detect minority class: both classes have equal "
                f"frequency ({counts[0]} each). Pass minority_label explicitly."
            )
        minority_label = unique[np.argmin(counts)]
        majority_label = unique[np.argmax(counts)]
        return cls(X, y, minority_label=minority_label, majority_label=majority_label, name=name)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """Feature matrix, shape (N, d), float64."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Label vector, shape (N,)."""
        return self._y

    @property
    def minority_label(self) -> int | bool:
        return self._minority_label

    @property
    def majority_label(self) -> int | bool:
        return self._majority_label

    @property
    def N(self) -> int:
        return len(self._y)

    @property
    def d(self) -> int:
        return self._X.shape[1]

    @property
    def n_minority(self) -> int:
        return self._n_minority

    @property
    def n_majority(self) -> int:
        return self._n_majority

    @property
    def IR(self) -> float:
        return self._n_majority / self._n_minority

    @property
    def minority_mask(self) -> np.ndarray:
        return self._minority_mask

    @property
    def majority_mask(self) -> np.ndarray:
        return self._majority_mask

    @property
    def X_minority(self) -> np.ndarray:
        return self._X[self._minority_mask]

    @property
    def X_majority(self) -> np.ndarray:
        return self._X[self._majority_mask]

    def __repr__(self) -> str:
        return (
            f"CIPADataset(name={self.name!r}, N={self.N}, d={self.d}, "
            f"IR={self.IR:.1f}, n_minority={self.n_minority})"
        )
