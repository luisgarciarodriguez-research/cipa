"""Unit tests for C2: constant columns out and scaling."""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline
from cipa.dimensions import compute_d6
from cipa.preprocessing import preprocess_dataset, preprocess_features


def _matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=3.0, scale=[1.0, 50.0, 0.01, 7.0], size=(200, 4))
    X[:, 2] = np.round(X[:, 2] * 100) / 100  # coarse but not constant
    return X


def _with_constants(X: np.ndarray) -> np.ndarray:
    n = len(X)
    return np.column_stack([np.full(n, 0.1), X[:, :2], np.full(n, 7.0), X[:, 2:]])


def test_constant_columns_are_dropped_and_reported():
    X = _with_constants(_matrix())
    X_out, info = preprocess_features(X, "none")
    assert X_out.shape == (200, 4)
    assert info["dropped_constant_columns"] == [0, 3]
    assert info["n_dropped_constant_columns"] == 2
    assert info["n_features_in"] == 6
    assert info["n_features_used"] == 4
    np.testing.assert_array_equal(X_out, _matrix())


def test_no_constant_columns_reports_empty_list():
    _, info = preprocess_features(_matrix(), "standard")
    assert info["dropped_constant_columns"] == []
    assert info["n_dropped_constant_columns"] == 0


def test_standard_scaling_gives_zero_mean_unit_std():
    X_out, info = preprocess_features(_with_constants(_matrix()), "standard")
    assert info["scaling"] == "standard"
    np.testing.assert_allclose(X_out.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(X_out.std(axis=0), 1.0, atol=1e-12)


def test_none_scaling_leaves_values_untouched():
    X = _matrix()
    X_out, _ = preprocess_features(X, "none")
    np.testing.assert_array_equal(X_out, X)


def test_robust_scaling_uses_median_and_iqr():
    X = _matrix()
    X_out, info = preprocess_features(X, "robust")
    np.testing.assert_allclose(np.median(X_out, axis=0), 0.0, atol=1e-12)
    q25, q75 = np.percentile(X_out, [25, 75], axis=0)
    np.testing.assert_allclose(q75 - q25, 1.0, atol=1e-12)
    assert info["robust_std_fallback_columns"] == []


def test_robust_with_zero_iqr_falls_back_to_std_without_inf_or_nan():
    rng = np.random.default_rng(1)
    n = 200
    sparse = np.zeros(n)
    sparse[:10] = rng.normal(5.0, 1.0, 10)  # IQR = 0, sigma > 0
    binary = (np.arange(n) < 20).astype(float)  # IQR = 0, sigma > 0
    X = np.column_stack([np.full(n, 2.0), sparse, rng.normal(size=n), binary])
    X_out, info = preprocess_features(X, "robust")
    assert np.all(np.isfinite(X_out))
    assert info["dropped_constant_columns"] == [0]
    assert info["robust_std_fallback_columns"] == [1, 3]
    # fallback columns: centred on the median (0) and divided by sigma
    np.testing.assert_allclose(X_out[:, 0], sparse / sparse.std())
    np.testing.assert_allclose(X_out[:, 2], binary / binary.std())


def test_invalid_scaling_raises():
    with pytest.raises(ValueError, match="scaling"):
        preprocess_features(_matrix(), "minmax")


def test_all_constant_raises():
    with pytest.raises(ValueError, match="constant"):
        preprocess_features(np.ones((20, 3)), "standard")


def test_non_finite_scaling_result_raises():
    X = np.column_stack([np.linspace(0, 1, 20), np.r_[np.zeros(19), 1e-320]])
    with pytest.raises(ValueError, match="non-finite"):
        preprocess_features(X, "standard")


def test_preprocess_dataset_keeps_labels_and_name():
    X = _with_constants(_matrix())
    y = np.array([1] * 40 + [0] * 160)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0, name="demo")
    out, info = preprocess_dataset(ds, "standard")
    assert out.d == 4 and out.N == 200
    assert out.name == "demo"
    assert out.n_minority == 40
    np.testing.assert_array_equal(out.y, y)
    assert info["n_dropped_constant_columns"] == 2
    assert ds.d == 6  # the original dataset is not modified


def test_d6_is_not_contaminated_by_constant_columns():
    """Before C2, each constant column added an MI of 0 to the average."""
    X = _matrix()
    y = np.array([1] * 40 + [0] * 160)
    X[:40, 0] += 2.0
    clean = CIPADataset(X, y, minority_label=1, majority_label=0)
    padded = CIPADataset(_with_constants(X), y, minority_label=1, majority_label=0)

    # Without preprocessing the constant columns pull the mean MI down
    assert compute_d6(padded, random_state=0).value > compute_d6(clean, random_state=0).value

    pipe = CIPAPipeline(random_state=0, scaling="none")
    d6_clean = pipe.run_dimensions_only(clean)[5].value
    d6_padded = pipe.run_dimensions_only(padded)[5].value
    assert d6_padded == d6_clean


def test_pipeline_records_dropped_columns():
    X = _with_constants(_matrix())
    y = np.array([1] * 40 + [0] * 160)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = CIPAPipeline(random_state=0).run_scoring_only(ds)
    prep = result.metadata["preprocessing"]
    assert prep["dropped_constant_columns"] == [0, 3]
    assert prep["scaling"] == "standard"
    assert result.metadata["d"] == 4
