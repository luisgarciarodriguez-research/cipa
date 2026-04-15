"""Shared pytest fixtures for CIPA unit and integration tests.

All fixtures return CIPADataset objects (not raw arrays) unless
they specifically test raw-array construction.

See DESIGN-02 §9 for the fixture design rationale.

Each fixture uses its own seeded RNG so fixtures are fully independent of
test-collection order and of each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset


# ---------------------------------------------------------------------------
# Structural fixtures — each captures a specific complexity scenario
# ---------------------------------------------------------------------------

@pytest.fixture
def perfectly_separable() -> CIPADataset:
    """Two Gaussians with no overlap. IR=1:1. Expected: D2≈0, D3≈0."""
    rng = np.random.default_rng(101)
    X_min = rng.normal(loc=5.0, scale=0.5, size=(50, 4))
    X_maj = rng.normal(loc=0.0, scale=0.5, size=(50, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 50 + [0] * 50)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="perfectly_separable")


@pytest.fixture
def perfectly_overlapping() -> CIPADataset:
    """Single Gaussian, labels split randomly. IR=1:1. Expected: D2≈high."""
    rng = np.random.default_rng(102)
    X = rng.normal(size=(100, 4))
    y = np.array([1] * 50 + [0] * 50)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="perfectly_overlapping")


@pytest.fixture
def high_imbalance() -> CIPADataset:
    """IR=100:1, well-separated classes. Expected: D1 high, D2 low."""
    rng = np.random.default_rng(103)
    X_min = rng.normal(loc=5.0, scale=0.3, size=(10, 4))
    X_maj = rng.normal(loc=0.0, scale=0.3, size=(1000, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 10 + [0] * 1000)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="high_imbalance")


@pytest.fixture
def all_outliers() -> CIPADataset:
    """All minority instances surrounded only by majority. Expected: D3≈1.0.

    Minority points live inside the majority cloud (same distribution) but are
    so outnumbered (50:1) that each minority point's k=5 nearest neighbours are
    virtually guaranteed to be majority → n_same=0 → all outliers → D3≈1.0.
    """
    rng = np.random.default_rng(104)
    X_maj = rng.normal(loc=0.0, scale=2.0, size=(500, 4))
    X_min = rng.normal(loc=0.0, scale=2.0, size=(10, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 10 + [0] * 500)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="all_outliers")


@pytest.fixture
def all_safe() -> CIPADataset:
    """All minority instances deeply in same-class region. Expected: D3=0.0."""
    rng = np.random.default_rng(105)
    X_min = rng.normal(loc=5.0, scale=0.1, size=(50, 4))
    X_maj = rng.normal(loc=0.0, scale=0.1, size=(50, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 50 + [0] * 50)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="all_safe")


@pytest.fixture
def fragmented_minority() -> CIPADataset:
    """Minority in 5 isolated clusters. Expected: D4 high."""
    rng = np.random.default_rng(106)
    centers = [(10, 0), (0, 10), (-10, 0), (0, -10), (5, 5)]
    X_min = np.vstack([
        rng.normal(loc=list(c) + [0, 0], scale=0.3, size=(6, 4))
        for c in centers
    ])
    X_maj = rng.normal(loc=[0, 0, 0, 0], scale=2.0, size=(200, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 30 + [0] * 200)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="fragmented_minority")


@pytest.fixture
def single_cluster_minority() -> CIPADataset:
    """Minority in one tight cluster. Expected: D4≈0."""
    rng = np.random.default_rng(107)
    X_min = rng.normal(loc=[5, 5, 5, 5], scale=0.2, size=(30, 4))
    X_maj = rng.normal(loc=[0, 0, 0, 0], scale=2.0, size=(200, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 30 + [0] * 200)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="single_cluster_minority")


@pytest.fixture
def high_dimensional() -> CIPADataset:
    """d=200, small minority. Expected: D5 high."""
    rng = np.random.default_rng(108)
    X_min = rng.normal(size=(20, 200))
    X_maj = rng.normal(size=(200, 200))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 20 + [0] * 200)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="high_dimensional")


@pytest.fixture
def uninformative_features() -> CIPADataset:
    """X is random noise, y is random binary. Expected: D6≈1."""
    rng = np.random.default_rng(109)
    X = rng.normal(size=(200, 10))
    y = rng.choice([0, 1], size=200, p=[0.8, 0.2])
    while (y == 1).sum() < 5:
        y = rng.choice([0, 1], size=200, p=[0.8, 0.2])
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="uninformative_features")


@pytest.fixture
def linearly_separable() -> CIPADataset:
    """Two clearly linearly separable clusters. Expected: L1=0, D7 low."""
    rng = np.random.default_rng(110)
    X_min = rng.normal(loc=[3, 3], scale=0.5, size=(30, 2))
    X_maj = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 30 + [0] * 100)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="linearly_separable")


# ---------------------------------------------------------------------------
# Minimal valid dataset (for fast smoke tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_valid() -> CIPADataset:
    """Smallest valid dataset: N=12, d=2, n_minority=2, IR=5."""
    rng = np.random.default_rng(111)
    X_min = rng.normal(loc=5.0, scale=0.5, size=(2, 2))
    X_maj = rng.normal(loc=0.0, scale=0.5, size=(10, 2))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 2 + [0] * 10)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="minimal_valid")
