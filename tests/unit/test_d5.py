"""Unit tests for D5 — Effective Dimensionality relative to the minority (cipa 2.0.0rc2).

D5 = ρ/(1+ρ) with ρ = r_95/|C₊|; the 1.x spectral entropy is kept as the
informative component ``spectral_entropy_norm``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cipa import CIPADataset
from cipa._constants import D5_VARIANCE_THRESHOLD
from cipa.dimensions import compute_d5, d5_dimensionality


def make_ds(n_min: int, n_maj: int, d: int = 8, seed: int = 0) -> CIPADataset:
    rng = np.random.default_rng(seed)
    n   = n_min + n_maj
    X   = rng.normal(size=(n, d))
    y   = np.array([1] * n_min + [0] * n_maj)
    return CIPADataset(X, y, minority_label=1, majority_label=0)


def subspace_ds(k: int, n_min: int, n: int = 400, d: int = 12, seed: int = 0) -> CIPADataset:
    """Rows in a k-dimensional subspace of R^d with negligible noise: r_95 = k by construction."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(d, k)))
    X = rng.normal(size=(n, k)) @ basis.T + 1e-9 * rng.normal(size=(n, d))
    y = np.array([1] * n_min + [0] * (n - n_min))
    return CIPADataset(X, y, minority_label=1, majority_label=0)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_d5_returns_dimension_result():
    ds = make_ds(n_min=40, n_maj=60)
    result = compute_d5(ds)
    assert result.dimension_id == "D5"
    assert 0.0 <= result.value < 1.0


def test_d5_components_present():
    result = compute_d5(make_ds(30, 70))
    assert set(result.components) == {
        "r_95", "n_minority", "rho", "H_nats", "H_max_nats",
        "spectral_entropy_norm", "n_components_fit",
    }
    assert result.components["n_minority"] == 30


def test_d5_metadata_present():
    d  = 6
    ds = make_ds(20, 80, d=d)
    result = compute_d5(ds)
    assert result.metadata["d"] == d
    assert result.metadata["n"] == 100
    assert result.metadata["variance_threshold"] == D5_VARIANCE_THRESHOLD == 0.95
    assert "top5_explained_variance_ratio" in result.metadata


def test_d5_top5_evr_metadata():
    """top5_explained_variance_ratio has at most 5 entries and sums ≤ 1."""
    result = compute_d5(make_ds(30, 70, d=8))
    top5 = result.metadata["top5_explained_variance_ratio"]
    assert len(top5) <= 5
    assert sum(top5) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Formula: r_95, rho, value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 5, 9])
def test_d5_exact_value_with_known_r95(k):
    """Data in a k-dimensional subspace → r_95 = k and D5 = k/(|C₊| + k)."""
    n_min = 50
    result = compute_d5(subspace_ds(k, n_min))
    assert result.components["r_95"] == k
    assert result.components["rho"] == pytest.approx(k / n_min, abs=1e-15)
    assert result.value == pytest.approx(k / (n_min + k), abs=1e-15)


def test_d5_rho_over_one_plus_rho_matches_components():
    result = compute_d5(make_ds(25, 175, d=10, seed=3))
    rho = result.components["rho"]
    assert rho == result.components["r_95"] / result.components["n_minority"]
    assert result.value == rho / (1.0 + rho)


def test_d5_is_half_when_r95_equals_minority_size():
    result = compute_d5(subspace_ds(k=4, n_min=4))
    assert result.components["r_95"] == 4
    assert result.value == 0.5


def test_d5_more_minority_lowers_value_with_same_spectrum():
    """Same X (same spectrum), more minority instances → lower D5."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 10)) * np.linspace(1.0, 3.0, 10)
    values, spectra = [], []
    for n_min in (10, 40, 120):
        y = np.array([1] * n_min + [0] * (300 - n_min))
        result = compute_d5(CIPADataset(X, y, minority_label=1, majority_label=0))
        values.append(result.value)
        spectra.append((result.components["r_95"], result.components["H_nats"]))
    assert spectra[0] == spectra[1] == spectra[2]
    assert values[0] > values[1] > values[2]


def test_d5_bounded_below_one_when_dimensions_exceed_minority():
    """d > N (as in gene expression): r_95 capped at N−1, D5 high but < 1."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 500))
    y = np.array([1] * 5 + [0] * 55)
    result = compute_d5(CIPADataset(X, y, minority_label=1, majority_label=0))
    assert result.components["r_95"] <= 59
    assert result.components["n_components_fit"] <= 59
    assert 0.5 < result.value < 1.0


def test_d5_in_unit_interval_random_shapes():
    rng = np.random.default_rng(99)
    for _ in range(8):
        n_min = int(rng.integers(2, 20))
        n_maj = int(rng.integers(10, 100))
        d = int(rng.integers(2, 40))
        assert 0.0 <= compute_d5(make_ds(n_min, n_maj, d=d)).value < 1.0


def test_d5_r95_capped_when_cumulative_ratio_never_reaches_threshold(monkeypatch):
    """Rounding can leave Σ ratios < 0.95; r_95 is then the number of fitted components."""

    class ShortPCA:
        def __init__(self, n_components, random_state=None):
            self.explained_variance_ratio_ = np.full(n_components, 0.9 / n_components)

        def fit(self, X):
            return self

    monkeypatch.setattr(d5_dimensionality, "PCA", ShortPCA)
    result = compute_d5(make_ds(20, 80, d=6))
    assert result.components["r_95"] == 6
    assert result.value == pytest.approx((6 / 20) / (1 + 6 / 20))


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

def test_d5_degenerate_d1():
    """d = 1 → returns 0.0 with zeroed components."""
    result = compute_d5(make_ds(20, 80, d=1))
    assert result.value == 0.0
    assert result.components["r_95"] == 0
    assert result.components["rho"] == 0.0
    assert result.components["n_minority"] == 20
    assert result.components["H_nats"] == 0.0
    assert result.components["H_max_nats"] == 0.0
    assert result.components["spectral_entropy_norm"] == 0.0
    assert result.components["n_components_fit"] == 0
    assert result.metadata["top5_explained_variance_ratio"] == []


def test_d5_degenerate_n_le_2():
    """N ≤ 2 → 0.0 (CIPADataset requires N ≥ 10, so a stand-in object is used)."""
    stub = SimpleNamespace(X=np.array([[0.0, 1.0], [1.0, 0.0]]), n_minority=1)
    result = compute_d5(stub)
    assert result.value == 0.0
    assert result.metadata["n"] == 2


def test_d5_single_component_is_not_special_cased():
    """Rank-1 data: r_95 = 1 and D5 = 1/(1 + |C₊|); the spectral entropy is 0."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=8)
    X = np.outer(rng.normal(size=100), v / np.linalg.norm(v))
    y = np.array([1] * 20 + [0] * 80)
    result = compute_d5(CIPADataset(X, y, minority_label=1, majority_label=0))
    assert result.components["r_95"] == 1
    assert result.value == pytest.approx(1 / 21, abs=1e-15)
    assert result.components["n_components_fit"] == 1
    assert result.components["spectral_entropy_norm"] == 0.0


# ---------------------------------------------------------------------------
# Informative component: the 1.x spectral entropy
# ---------------------------------------------------------------------------

def test_d5_spectral_entropy_formula():
    """spectral_entropy_norm == H_nats / ln(n_components_fit)."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 5))
    y = np.array([1] * 40 + [0] * 160)
    result = compute_d5(CIPADataset(X, y, minority_label=1, majority_label=0))
    H     = result.components["H_nats"]
    H_max = result.components["H_max_nats"]
    k     = result.components["n_components_fit"]
    assert H_max == pytest.approx(np.log(k), abs=1e-9)
    assert result.components["spectral_entropy_norm"] == pytest.approx(H / H_max, abs=1e-9)


def test_d5_spectral_entropy_isotropic_near_maximum():
    """Isotropic noise: entropy ≈ 1 (the 1.x limitation), while D5 stays low."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 10))
    y = np.array([1] * 50 + [0] * 450)
    result = compute_d5(CIPADataset(X, y, minority_label=1, majority_label=0))
    assert result.components["spectral_entropy_norm"] > 0.85
    assert result.value < 0.2


def test_d5_spectral_entropy_lower_for_concentrated_variance():
    rng = np.random.default_rng(1)
    n, d = 300, 20
    X_iso = rng.normal(size=(n, d))
    X_lowrank = np.zeros((n, d))
    X_lowrank[:, :2] = rng.normal(scale=10.0, size=(n, 2))
    X_lowrank[:, 2:] = rng.normal(scale=0.01, size=(n, d - 2))
    y = np.array([1] * 50 + [0] * 250)
    iso = compute_d5(CIPADataset(X_iso, y, minority_label=1, majority_label=0))
    low = compute_d5(CIPADataset(X_lowrank, y, minority_label=1, majority_label=0))
    assert low.components["spectral_entropy_norm"] < iso.components["spectral_entropy_norm"]
    assert low.components["r_95"] < iso.components["r_95"]
    assert low.value < iso.value
