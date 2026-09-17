"""Synthetic datasets for the v1.2.1 regression test (cipa 2.0.0 handoff §1, §5).

Every case is continuous, duplicate-free and has no constant column, so the
2.0.0 corrections for duplicates (C3, C4) and constant columns (C2) are inert
and D1–D6 must reproduce v1.2.1 exactly. The builders use only numpy, so the
same module is imported by the test and by ``generate_v1_2_1_reference.py``,
which runs against a checkout of the ``v1.2.1`` tag.
"""

from __future__ import annotations

import numpy as np

SEED = 42


def gaussian_overlap() -> tuple[np.ndarray, np.ndarray]:
    """Two partially overlapping Gaussians, N=600, d=6, IR=5."""
    rng = np.random.default_rng(201)
    X_min = rng.normal(loc=1.2, scale=1.0, size=(100, 6))
    X_maj = rng.normal(loc=0.0, scale=1.0, size=(500, 6))
    return np.vstack([X_min, X_maj]), np.array([1] * 100 + [0] * 500)


def fragmented_minority() -> tuple[np.ndarray, np.ndarray]:
    """Minority split into four sub-concepts inside a wide majority, N=800, IR=9."""
    rng = np.random.default_rng(202)
    centers = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [-4, 0, 0, 0], [0, -4, 0, 0]])
    X_min = np.vstack([rng.normal(loc=c, scale=0.4, size=(20, 4)) for c in centers])
    X_maj = rng.normal(loc=0.0, scale=2.5, size=(720, 4))
    return np.vstack([X_min, X_maj]), np.array([1] * 80 + [0] * 720)


def correlated_high_dim() -> tuple[np.ndarray, np.ndarray]:
    """Correlated features from a random linear mix, N=400, d=30, IR=3."""
    rng = np.random.default_rng(203)
    latent = rng.normal(size=(400, 8))
    X = latent @ rng.normal(size=(8, 30)) + 0.3 * rng.normal(size=(400, 30))
    y = np.array([1] * 100 + [0] * 300)
    X[:100, :3] += 1.0
    return X, y


def heterogeneous_scales() -> tuple[np.ndarray, np.ndarray]:
    """Columns on very different scales, N=1500, d=8, IR=29."""
    rng = np.random.default_rng(204)
    scales = np.array([1e-2, 1.0, 5.0, 1e2, 1e3, 0.5, 20.0, 1e4])
    X_min = rng.normal(loc=0.8, scale=1.0, size=(50, 8)) * scales
    X_maj = rng.normal(loc=0.0, scale=1.0, size=(1450, 8)) * scales
    return np.vstack([X_min, X_maj]), np.array([1] * 50 + [0] * 1450)


CASES = {
    "gaussian_overlap": gaussian_overlap,
    "fragmented_minority": fragmented_minority,
    "correlated_high_dim": correlated_high_dim,
    "heterogeneous_scales": heterogeneous_scales,
}
