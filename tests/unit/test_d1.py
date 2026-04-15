"""Unit tests for D1 — Imbalance Distribution (normalised binary entropy formula)."""

import numpy as np
import pytest

from cipa import CIPADataset
from cipa.dimensions import compute_d1


def make_ds(n_minority, n_majority, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_minority + n_majority, d))
    y = np.array([1] * n_minority + [0] * n_majority)
    return CIPADataset(X, y, minority_label=1, majority_label=0)


def _expected_d1(n_min: int, n_maj: int) -> float:
    """D1 = 1 - H(Y) where H(Y) is binary entropy in bits."""
    N = n_min + n_maj
    p_plus = n_min / N
    p_minus = n_maj / N
    H_Y = -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))
    return float(np.clip(1.0 - H_Y, 0.0, 1.0))


def test_d1_returns_dimension_result():
    ds = make_ds(10, 100)
    result = compute_d1(ds)
    assert result.dimension_id == "D1"
    assert 0.0 <= result.value <= 1.0


def test_d1_balanced_is_zero():
    """IR = 1 → H(Y) = 1 bit → D1 = 0."""
    ds = make_ds(50, 50)
    result = compute_d1(ds)
    assert pytest.approx(result.value, abs=1e-9) == 0.0


def test_d1_increases_with_imbalance():
    low = compute_d1(make_ds(50, 100)).value   # IR=2
    mid = compute_d1(make_ds(20, 200)).value   # IR=10
    high = compute_d1(make_ds(5, 500)).value   # IR=100
    assert low < mid < high


def test_d1_high_imbalance_close_to_1():
    """IR ≈ 577 → D1 ≈ 0.982 (H(Y) formula)."""
    ds = make_ds(492, 284315)  # CreditCard-like
    result = compute_d1(ds)
    expected = _expected_d1(492, 284315)
    assert pytest.approx(result.value, abs=1e-6) == expected
    assert result.value > 0.97


def test_d1_components_present():
    ds = make_ds(10, 100)
    result = compute_d1(ds)
    assert "IR" in result.components
    assert "H_Y_bits" in result.components
    assert result.components["IR"] == pytest.approx(ds.IR)


def test_d1_metadata_present():
    ds = make_ds(10, 100)
    result = compute_d1(ds)
    assert result.metadata["n_minority"] == 10
    assert result.metadata["N"] == 110


def test_d1_exact_formula():
    """Verify exact H(Y) formula for concrete values."""
    # IR=3 (100 min, 300 maj): p+=0.25, p-=0.75
    # H(Y) = -(0.25*log2(0.25) + 0.75*log2(0.75)) ≈ 0.8113
    # D1 = 1 - 0.8113 ≈ 0.1887
    n_min, n_maj = 100, 300
    result = compute_d1(make_ds(n_min, n_maj))
    expected = _expected_d1(n_min, n_maj)
    assert pytest.approx(result.value, abs=1e-9) == expected


def test_d1_formula_ir_7():
    """IR=7 (100 min, 700 maj): p+=0.125, p-=0.875 → D1 ≈ 0.456."""
    n_min, n_maj = 100, 700
    result = compute_d1(make_ds(n_min, n_maj))
    expected = _expected_d1(n_min, n_maj)
    assert pytest.approx(result.value, abs=1e-9) == expected


def test_d1_h_y_bits_component_correct():
    """H_Y_bits component equals the binary entropy of the class distribution."""
    n_min, n_maj = 30, 70
    ds = make_ds(n_min, n_maj)
    result = compute_d1(ds)
    N = n_min + n_maj
    p_plus = n_min / N
    p_minus = n_maj / N
    expected_H = -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))
    assert pytest.approx(result.components["H_Y_bits"], abs=1e-9) == expected_H


def test_d1_value_equals_one_minus_h_y():
    """D1 = 1 - H_Y_bits always."""
    for n_min, n_maj in [(10, 90), (5, 95), (2, 48), (50, 50)]:
        result = compute_d1(make_ds(n_min, n_maj))
        assert pytest.approx(result.value, abs=1e-9) == 1.0 - result.components["H_Y_bits"]
