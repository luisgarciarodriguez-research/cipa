"""Unit tests for D6 — Feature Informativeness."""

import numpy as np
import pytest

from cipa import CIPADataset
from cipa.dimensions import compute_d6


def make_ds(X, y):
    return CIPADataset(X, y, minority_label=1, majority_label=0)


def test_d6_returns_dimension_result(uninformative_features):
    result = compute_d6(uninformative_features, random_state=0)
    assert result.dimension_id == "D6"
    assert 0.0 <= result.value <= 1.0


def test_d6_uninformative_features_high(uninformative_features):
    """Random X uncorrelated with y → D6 should be high."""
    result = compute_d6(uninformative_features, random_state=0)
    assert result.value > 0.6


def test_d6_informative_features_low():
    """X perfectly predicts y → D6 should be low."""
    rng = np.random.default_rng(42)
    X_min = rng.normal(loc=5.0, scale=0.1, size=(30, 4))
    X_maj = rng.normal(loc=0.0, scale=0.1, size=(120, 4))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * 30 + [0] * 120)
    ds = make_ds(X, y)
    result = compute_d6(ds, random_state=0)
    assert result.value < 0.3


def test_d6_components_present(uninformative_features):
    result = compute_d6(uninformative_features, random_state=0)
    assert "H_Y_nats" in result.components
    assert "I_mean_nats" in result.components
    assert "mi_scores" in result.components
    assert len(result.components["mi_scores"]) == uninformative_features.d


def test_d6_top_3_features_are_valid_indices(uninformative_features):
    result = compute_d6(uninformative_features, random_state=0)
    top3 = result.components["top_3_features"]
    assert len(top3) == 3
    assert all(0 <= idx < uninformative_features.d for idx in top3)
