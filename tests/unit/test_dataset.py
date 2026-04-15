"""Unit tests for CIPADataset. Covers all SPEC-01 §3.3 validation rules."""

import numpy as np
import pytest

from cipa import CIPADataset


@pytest.fixture
def valid_X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 4)).astype(np.float64)


@pytest.fixture
def valid_y():
    return np.array([1] * 4 + [0] * 16)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_construction_succeeds(valid_X, valid_y):
    ds = CIPADataset(valid_X, valid_y, minority_label=1, majority_label=0)
    assert ds.N == 20
    assert ds.d == 4
    assert ds.n_minority == 4
    assert ds.n_majority == 16
    assert pytest.approx(ds.IR) == 4.0


def test_from_arrays_auto_detects_minority(valid_X, valid_y):
    ds = CIPADataset.from_arrays(valid_X, valid_y)
    assert ds.minority_label == 1
    assert ds.majority_label == 0


def test_name_stored(valid_X, valid_y):
    ds = CIPADataset(valid_X, valid_y, minority_label=1, majority_label=0, name="test")
    assert ds.name == "test"


def test_X_converted_to_float64(valid_y):
    X_int = np.ones((20, 4), dtype=np.int32)
    X_int[:4] = 5
    ds = CIPADataset(X_int, valid_y, minority_label=1, majority_label=0)
    assert ds.X.dtype == np.float64


def test_x_minority_x_majority_shapes(valid_X, valid_y):
    ds = CIPADataset(valid_X, valid_y, minority_label=1, majority_label=0)
    assert ds.X_minority.shape == (4, 4)
    assert ds.X_majority.shape == (16, 4)


def test_masks_are_boolean(valid_X, valid_y):
    ds = CIPADataset(valid_X, valid_y, minority_label=1, majority_label=0)
    assert ds.minority_mask.dtype == bool
    assert ds.majority_mask.dtype == bool
    assert ds.minority_mask.sum() == 4
    assert ds.majority_mask.sum() == 16


def test_repr_contains_key_info(valid_X, valid_y):
    ds = CIPADataset(valid_X, valid_y, minority_label=1, majority_label=0, name="demo")
    r = repr(ds)
    assert "demo" in r
    assert "N=20" in r
    assert "d=4" in r


# ---------------------------------------------------------------------------
# Validation errors (SPEC-01 §3.3)
# ---------------------------------------------------------------------------

def test_X_not_2d_raises(valid_y):
    with pytest.raises(ValueError, match="2-dimensional"):
        CIPADataset(np.ones(20), valid_y, minority_label=1, majority_label=0)


def test_X_contains_nan_raises(valid_y):
    X = np.ones((20, 4))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        CIPADataset(X, valid_y, minority_label=1, majority_label=0)


def test_X_contains_inf_raises(valid_y):
    X = np.ones((20, 4))
    X[1, 2] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        CIPADataset(X, valid_y, minority_label=1, majority_label=0)


def test_length_mismatch_raises(valid_X):
    y_wrong = np.array([1] * 5 + [0] * 10)
    with pytest.raises(ValueError, match="same length"):
        CIPADataset(valid_X, y_wrong, minority_label=1, majority_label=0)


def test_not_binary_raises(valid_X):
    y_multi = np.array([0, 1, 2] * 6 + [0, 1])
    with pytest.raises(ValueError, match="2 unique"):
        CIPADataset(valid_X, y_multi, minority_label=0, majority_label=1)


def test_minority_label_not_in_y_raises(valid_X, valid_y):
    with pytest.raises(ValueError, match="minority_label"):
        CIPADataset(valid_X, valid_y, minority_label=99, majority_label=0)


def test_n_minority_less_than_2_raises():
    X = np.ones((10, 2))
    y = np.array([1] + [0] * 9)
    with pytest.raises(ValueError, match="n_minority must be >= 2"):
        CIPADataset(X, y, minority_label=1, majority_label=0)


def test_n_less_than_10_raises():
    X = np.ones((9, 2))
    y = np.array([1, 1] + [0] * 7)
    with pytest.raises(ValueError, match="N >= 10"):
        CIPADataset(X, y, minority_label=1, majority_label=0)


def test_same_minority_majority_label_raises(valid_X, valid_y):
    with pytest.raises(ValueError, match="different"):
        CIPADataset(valid_X, valid_y, minority_label=1, majority_label=1)


# ---------------------------------------------------------------------------
# from_arrays edge cases
# ---------------------------------------------------------------------------

def test_from_arrays_equal_classes_raises():
    X = np.ones((20, 2))
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(ValueError, match="equal frequency"):
        CIPADataset.from_arrays(X, y)
