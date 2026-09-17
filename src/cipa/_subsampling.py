"""Per-dimension subsampling protocol (C6). See §3.1 of García Rodríguez et al. (2026).

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

logger = logging.getLogger(__name__)

# Independent seed streams derived from the pipeline's random_state
STREAM_D2_D7 = 0
STREAM_D4 = 1


def validate_random_state(random_state: Any) -> int:
    """Return random_state as a non-negative int; None and other types are rejected (C9).

    Raises
    ------
    ValueError
        If random_state is None, not an integer, or negative.
    """
    if random_state is None or isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError(
            f"random_state must be a non-negative integer (never None), got {random_state!r}"
        )
    if random_state < 0:
        raise ValueError(f"random_state must be >= 0, got {random_state}")
    return int(random_state)


def derive_seed(random_state: int, stream: int, index: int) -> int:
    """Seed for subsample ``index`` of ``stream``, derived from ``random_state``.

    Uses ``numpy.random.SeedSequence`` so every subsample has its own
    reproducible stream, independent of the global ``np.random`` state.
    ``np.random.default_rng(seed)`` regenerates the subsample.
    """
    sequence = np.random.SeedSequence(entropy=random_state, spawn_key=(stream, index))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def stratified_indices(
    minority_mask: np.ndarray,
    n_max: int,
    seed: int,
) -> np.ndarray:
    """Draw n_max row indices without replacement, preserving the class ratio.

    The minority count is round(n_max · n_minority / N), so the subsample IR
    matches the dataset IR up to one instance. It is floored at 2 (the
    minimum a CIPADataset accepts), which only binds when the expected count
    is below 1.5.

    Parameters
    ----------
    minority_mask : np.ndarray of bool, shape (N,)
        True for minority rows.
    n_max : int
        Subsample size; must be < N.
    seed : int
        Seed for ``np.random.default_rng``.

    Returns
    -------
    np.ndarray of int, shape (n_max,)
        Sorted row indices.
    """
    minority_idx = np.flatnonzero(minority_mask)
    majority_idx = np.flatnonzero(~minority_mask)
    n = len(minority_mask)
    n_min = round(n_max * len(minority_idx) / n)
    if n_min < 2:
        logger.warning(
            "stratified subsample: expected %.2f minority rows in %d; using 2",
            n_max * len(minority_idx) / n, n_max,
        )
        n_min = 2
    n_min = min(n_min, len(minority_idx))
    n_maj = min(n_max - n_min, len(majority_idx))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(minority_idx, size=n_min, replace=False),
        rng.choice(majority_idx, size=n_maj, replace=False),
    ])
    return np.sort(chosen)


def minority_indices(n_minority: int, n_max: int, seed: int) -> np.ndarray:
    """Draw n_max positions out of the n_minority minority rows, sorted.

    Parameters
    ----------
    n_minority : int
        Number of minority rows; must be > n_max.
    n_max : int
        Subsample size.
    seed : int
        Seed for ``np.random.default_rng``.

    Returns
    -------
    np.ndarray of int, shape (n_max,)
        Sorted positions into the minority rows.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_minority, size=n_max, replace=False))


def median_iqr(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Median and interquartile range (Q75 − Q25, linear interpolation)."""
    arr = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.percentile(arr, [25, 50, 75])
    return float(median), float(q75 - q25)
