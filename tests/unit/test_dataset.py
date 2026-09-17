"""Unit tests for CIPADataset. Covers all SPEC-01 §3.3 validation rules."""

import warnings

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


def test_from_arrays_auto_detects_minority_and_is_deprecated(valid_X, valid_y):
    with pytest.warns(DeprecationWarning, match="minority_label"):
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
    with pytest.raises(ValueError, match="equal frequency"), pytest.warns(DeprecationWarning):
        CIPADataset.from_arrays(X, y)


def test_from_arrays_non_binary_raises():
    with pytest.raises(ValueError, match="2 unique"), pytest.warns(DeprecationWarning):
        CIPADataset.from_arrays(np.ones((20, 2)), np.array([0, 1, 2, 3] * 5))


# ---------------------------------------------------------------------------
# C1 — declared minority is taken as declared
# ---------------------------------------------------------------------------

def test_declared_minority_that_is_majority_warns_and_is_not_inverted(valid_X, valid_y, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="cipa"), \
            pytest.warns(UserWarning, match="more than majority_label"):
        ds = CIPADataset(valid_X, valid_y, minority_label=0, majority_label=1, name="flip")
    assert ds.minority_label == 0
    assert ds.n_minority == 16
    assert ds.minority_is_majority is True
    assert "more than majority_label" in caplog.text


def test_balanced_classes_do_not_warn(recwarn):
    X = np.random.default_rng(0).normal(size=(20, 2))
    ds = CIPADataset(X, np.array([0, 1] * 10), minority_label=1, majority_label=0)
    assert ds.minority_is_majority is False
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_pipeline_records_minority_is_majority(valid_X, valid_y):
    from cipa import CIPAPipeline

    with pytest.warns(UserWarning):
        ds = CIPADataset(valid_X, valid_y, minority_label=0, majority_label=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # the pipeline must not warn again
        result = CIPAPipeline().run_scoring_only(ds)
    meta = result.metadata
    assert meta["minority_is_majority"] is True
    assert (meta["minority_label"], meta["n_minority"], meta["n_majority"]) == (0, 16, 4)
    assert result.difficulty_score.dimensions[2].metadata["n_queries"] == 16


def test_subset_and_with_features_validate():
    X = np.random.default_rng(1).normal(size=(30, 3))
    y = np.array([1] * 6 + [0] * 24)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    sub = ds._subset(np.arange(4, 30))
    assert (sub.N, sub.n_minority, sub.n_majority) == (26, 2, 24)
    with pytest.raises(ValueError, match="n_minority"):
        ds._subset(np.arange(5, 30))
    with pytest.raises(ValueError, match="N >= 10"):
        ds._subset(np.arange(0, 8))
    with pytest.raises(ValueError, match="invalid shape"):
        ds._with_features(np.ones((29, 3)))
