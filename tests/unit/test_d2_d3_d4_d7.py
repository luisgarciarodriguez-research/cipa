"""Unit tests for D2, D3, D4, D7."""

import numpy as np
import pytest

from cipa.dimensions import compute_d2, compute_d3, compute_d4, compute_d7

# ---------------------------------------------------------------------------
# D2 — Class Overlap
# ---------------------------------------------------------------------------

def test_d2_separable_is_low(perfectly_separable):
    result = compute_d2(perfectly_separable)
    assert result.dimension_id == "D2"
    assert result.value < 0.2


def test_d2_overlapping_is_high(perfectly_overlapping):
    result = compute_d2(perfectly_overlapping)
    assert result.value > 0.4


def test_d2_in_range(high_imbalance):
    result = compute_d2(high_imbalance)
    assert 0.0 <= result.value <= 1.0


def test_d2_invalid_weights_raise(perfectly_separable):
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        compute_d2(perfectly_separable, weights=(0.5, 0.5, 0.5))


def test_d2_components_present(perfectly_separable):
    result = compute_d2(perfectly_separable)
    for key in ("F3", "N1", "kDN", "alpha", "beta", "gamma"):
        assert key in result.components


# ---------------------------------------------------------------------------
# D3 — Instance Hardness
# ---------------------------------------------------------------------------

def test_d3_all_safe_is_zero(all_safe):
    result = compute_d3(all_safe)
    assert result.dimension_id == "D3"
    assert result.value < 0.1  # mostly safe


def test_d3_all_outliers_is_one(all_outliers):
    result = compute_d3(all_outliers)
    assert result.value > 0.8


def test_d3_in_range(high_imbalance):
    result = compute_d3(high_imbalance)
    assert 0.0 <= result.value <= 1.0


def test_d3_typology_counts_sum_to_n_minority(perfectly_overlapping):
    result = compute_d3(perfectly_overlapping)
    c = result.components
    total = c["n_safe"] + c["n_borderline"] + c["n_rare"] + c["n_outlier"]
    assert total == perfectly_overlapping.n_minority


def test_d3_k_auto_reduced_when_k_ge_n_minority():
    """When k >= n_minority, k should be silently reduced with a warning."""
    rng = np.random.default_rng(0)
    from cipa import CIPADataset
    X = np.vstack([rng.normal(5, 0.5, (3, 2)), rng.normal(0, 0.5, (20, 2))])
    y = np.array([1] * 3 + [0] * 20)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d3(ds, k=10)  # k=10 > n_minority=3
    assert result.metadata["k"] < 10
    assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# D4 — Sub-concept Fragmentation
# ---------------------------------------------------------------------------

def test_d4_single_cluster_is_low(single_cluster_minority):
    result = compute_d4(single_cluster_minority)
    assert result.dimension_id == "D4"
    assert result.value < 0.3


def test_d4_fragmented_is_higher(fragmented_minority):
    result = compute_d4(fragmented_minority)
    assert result.value > 0.1  # should detect multiple sub-clusters


def test_d4_in_range(high_imbalance):
    result = compute_d4(high_imbalance)
    assert 0.0 <= result.value <= 1.0


def test_d4_ecindex_in_range(fragmented_minority):
    result = compute_d4(fragmented_minority)
    assert 0.0 <= result.components["ECindex"] <= 1.0


def test_d4_degenerate_few_minority():
    rng = np.random.default_rng(0)
    from cipa import CIPADataset
    X = np.vstack([rng.normal(0, 1, (2, 4)), rng.normal(5, 1, (20, 4))])
    y = np.array([1, 1] + [0] * 20)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d4(ds, dbscan_min_samples=3)
    assert result.value == 0.0
    assert result.metadata.get("degenerate")


# ---------------------------------------------------------------------------
# D7 — Boundary Complexity
# ---------------------------------------------------------------------------

def test_d7_separable_is_low(perfectly_separable):
    result = compute_d7(perfectly_separable, random_state=0)
    assert result.dimension_id == "D7"
    assert result.value < 0.3


def test_d7_overlapping_is_higher(perfectly_overlapping):
    result = compute_d7(perfectly_overlapping, random_state=0)
    assert result.value > 0.3


def test_d7_in_range(high_imbalance):
    result = compute_d7(high_imbalance, random_state=0)
    assert 0.0 <= result.value <= 1.0


def test_d7_components_present(perfectly_separable):
    result = compute_d7(perfectly_separable, random_state=0)
    assert "L1" in result.components
    assert "N2norm" in result.components
    assert result.components["converged"] is True
    assert result.metadata["svc_max_iter"] == 10_000
