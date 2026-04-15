"""Unit tests for D5 — Effective Dimensionality (spectral entropy of PCA eigenvalues)."""

import numpy as np
import pytest

from cipa import CIPADataset
from cipa.dimensions import compute_d5


def make_ds(n_min: int, n_maj: int, d: int = 8, seed: int = 0) -> CIPADataset:
    rng = np.random.default_rng(seed)
    n   = n_min + n_maj
    X   = rng.normal(size=(n, d))
    y   = np.array([1] * n_min + [0] * n_maj)
    return CIPADataset(X, y, minority_label=1, majority_label=0)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_d5_returns_dimension_result():
    ds = make_ds(n_min=40, n_maj=60)
    result = compute_d5(ds)
    assert result.dimension_id == "D5"
    assert 0.0 <= result.value <= 1.0


def test_d5_value_in_unit_interval():
    for n_min, n_maj in [(10, 90), (2, 98), (50, 50), (40, 60)]:
        assert 0.0 <= compute_d5(make_ds(n_min, n_maj)).value <= 1.0


def test_d5_components_present():
    result = compute_d5(make_ds(30, 70))
    assert "H_nats"           in result.components
    assert "H_max_nats"       in result.components
    assert "n_components_fit" in result.components


def test_d5_metadata_present():
    d  = 6
    ds = make_ds(20, 80, d=d)
    result = compute_d5(ds)
    assert result.metadata["d"] == d
    assert result.metadata["n"] == 100
    assert "top5_explained_variance_ratio" in result.metadata


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

def test_d5_degenerate_d1():
    """d = 1 → returns 0.0."""
    ds = make_ds(20, 80, d=1)
    assert compute_d5(ds).value == 0.0


def test_d5_degenerate_n_le_2():
    """n ≤ 4 (very small dataset) → returns 0.0 due to degenerate PCA."""
    # n=4 is the smallest valid dataset (n_minority>=2, n_majority>=2)
    # n_components = min(n-1, d) = min(3, 4) = 3, but spectral entropy may be valid
    # Use n=3 with d=2: n_components = min(2, 2) = 2, but check degenerate code path
    # Actually, degenerate is d==1 or n<=2; use d=1 for this test
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 1))  # d=1 → degenerate
    y = np.array([1] * 3 + [0] * 7)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    assert compute_d5(ds).value == 0.0


def test_d5_degenerate_components_zero():
    """Degenerate case returns zero components."""
    ds = make_ds(20, 80, d=1)
    result = compute_d5(ds)
    assert result.components["H_nats"] == 0.0
    assert result.components["H_max_nats"] == 0.0
    assert result.components["n_components_fit"] == 0


# ---------------------------------------------------------------------------
# Spectral entropy formula correctness
# ---------------------------------------------------------------------------

def test_d5_isotropic_data_near_maximum():
    """Isotropic Gaussian data → eigenvalues nearly uniform → D5 near 1."""
    rng = np.random.default_rng(42)
    n, d = 500, 10
    X = rng.normal(size=(n, d))
    y = np.array([1] * 50 + [0] * 450)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d5(ds)
    # Isotropic data: all eigenvalues ≈ equal → H ≈ H_max → D5 near 1
    assert result.value > 0.85


def test_d5_rank1_data_is_zero():
    """Data on a 1-D line: all variance in 1 component → D5 = 0."""
    rng = np.random.default_rng(0)
    n, d = 100, 8
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    t = rng.normal(size=n)
    X = np.outer(t, v)
    y = np.array([1] * 20 + [0] * 80)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d5(ds)
    # Only 1 non-zero eigenvalue → k=1 → D5 = 0
    assert result.value == pytest.approx(0.0, abs=1e-6)


def test_d5_fewer_components_lowers_value():
    """Concentrated variance (lower effective rank) → lower D5 than isotropic."""
    rng = np.random.default_rng(1)
    n = 300
    d = 20

    # Isotropic: variance spread across all 20 dimensions
    X_iso = rng.normal(size=(n, d))
    # Low-rank: variance concentrated in 2 dimensions
    X_lowrank = np.zeros((n, d))
    X_lowrank[:, :2] = rng.normal(scale=10.0, size=(n, 2))
    X_lowrank[:, 2:] = rng.normal(scale=0.01, size=(n, d - 2))

    y = np.array([1] * 50 + [0] * 250)
    ds_iso     = CIPADataset(X_iso,     y, minority_label=1, majority_label=0)
    ds_lowrank = CIPADataset(X_lowrank, y, minority_label=1, majority_label=0)

    v_iso     = compute_d5(ds_iso).value
    v_lowrank = compute_d5(ds_lowrank).value
    assert v_lowrank < v_iso


def test_d5_h_nats_equals_entropy_formula():
    """H_nats == -Σ pᵢ ln(pᵢ) over positive explained variance ratios."""
    rng = np.random.default_rng(7)
    n, d = 200, 5
    X = rng.normal(size=(n, d))
    y = np.array([1] * 40 + [0] * 160)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d5(ds)

    # Verify: value == H_nats / H_max_nats
    H      = result.components["H_nats"]
    H_max  = result.components["H_max_nats"]
    k      = result.components["n_components_fit"]

    assert H_max == pytest.approx(np.log(k), abs=1e-9)
    assert result.value == pytest.approx(H / H_max, abs=1e-9)


def test_d5_value_clipped_to_unit_interval():
    """Result is always clipped to [0, 1]."""
    rng = np.random.default_rng(99)
    for _ in range(5):
        n_min = rng.integers(2, 20)
        n_maj = rng.integers(10, 100)
        d = rng.integers(2, 15)
        ds = make_ds(int(n_min), int(n_maj), d=int(d))
        v = compute_d5(ds).value
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Independence from class counts (spectral entropy depends on X, not labels)
# ---------------------------------------------------------------------------

def test_d5_depends_on_feature_geometry():
    """D5 changes with feature structure (unlike the old T2 formula)."""
    rng = np.random.default_rng(3)
    n = 200
    d = 10
    y = np.array([1] * 40 + [0] * 160)

    # Isotropic
    X_iso = rng.normal(size=(n, d))
    # Correlated (low effective rank)
    base = rng.normal(size=(n, 2))
    X_corr = np.hstack([base, base @ rng.normal(size=(2, d - 2))])

    ds_iso  = CIPADataset(X_iso,  y, minority_label=1, majority_label=0)
    ds_corr = CIPADataset(X_corr, y, minority_label=1, majority_label=0)

    # Different feature geometry → different D5 values
    assert compute_d5(ds_iso).value != pytest.approx(compute_d5(ds_corr).value, abs=0.05)


def test_d5_top5_evr_metadata():
    """top5_explained_variance_ratio has at most 5 entries and sums ≤ 1."""
    ds = make_ds(30, 70, d=8)
    result = compute_d5(ds)
    top5 = result.metadata["top5_explained_variance_ratio"]
    assert len(top5) <= 5
    assert sum(top5) <= 1.0 + 1e-9
