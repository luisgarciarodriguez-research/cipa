"""Unit tests for C3: neighbours that tolerate exact duplicates."""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset
from cipa._knn import _KNNCache, kneighbors_excluding_self
from cipa.dimensions import compute_d2, compute_d3
from cipa.ecol import compute_n2


class _StubNeighbors:
    """Returns crafted (distance, index) arrays to force tie orders sklearn may produce."""

    def __init__(self, dist, idx):
        self.dist, self.idx = np.asarray(dist, float), np.asarray(idx)

    def kneighbors(self, X, n_neighbors):
        return self.dist[: len(X)], self.idx[: len(X)]


def test_self_in_first_column_is_removed():
    stub = _StubNeighbors([[0.0, 1.0, 2.0]], [[4, 7, 9]])
    dist, idx = kneighbors_excluding_self(stub, np.zeros((1, 2)), np.array([4]), k=2)
    np.testing.assert_array_equal(idx, [[7, 9]])
    np.testing.assert_array_equal(dist, [[1.0, 2.0]])


def test_twin_before_self_is_kept():
    """Row 4 has a twin (row 8); the tree listed the twin first."""
    stub = _StubNeighbors([[0.0, 0.0, 3.0]], [[8, 4, 9]])
    dist, idx = kneighbors_excluding_self(stub, np.zeros((1, 2)), np.array([4]), k=2)
    np.testing.assert_array_equal(idx, [[8, 9]])
    np.testing.assert_array_equal(dist, [[0.0, 3.0]])


def test_self_absent_among_many_twins_drops_last_column():
    stub = _StubNeighbors([[0.0, 0.0, 0.0]], [[1, 2, 3]])
    _, idx = kneighbors_excluding_self(stub, np.zeros((1, 2)), np.array([0]), k=2)
    np.testing.assert_array_equal(idx, [[1, 2]])


def test_chunked_queries_match_single_block():
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(300, 2)), 1)  # many exact duplicates
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(algorithm="ball_tree").fit(X)
    whole = kneighbors_excluding_self(nn, X, np.arange(300), k=4, chunk_size=1000)
    chunked = kneighbors_excluding_self(nn, X, np.arange(300), k=4, chunk_size=7)
    np.testing.assert_array_equal(whole[1], chunked[1])
    assert not np.any(whole[1] == np.arange(300)[:, None])


def _twin_pairs(n_pairs: int = 12) -> CIPADataset:
    """Each location holds one minority and one majority row, far from the rest."""
    locs = np.arange(n_pairs, dtype=float)[:, None] * np.array([[100.0, 0.0]])
    X = np.vstack([locs, locs])
    y = np.array([1] * n_pairs + [0] * n_pairs)
    return CIPADataset(X, y, minority_label=1, majority_label=0)


def test_twin_with_other_label_counts_in_kdn():
    ds = _twin_pairs()
    result = compute_d2(ds, k=1)
    assert result.components["kDN"] == 1.0


def test_twin_with_other_label_counts_in_d3():
    ds = _twin_pairs()
    result = compute_d3(ds, k=1)
    assert result.components["n_outlier"] == ds.n_minority
    assert result.value == 1.0


def test_d3_counts_duplicate_groups_without_self():
    """Groups of 6 identical rows: 3 minority + 3 majority, k = 5.

    Each minority row sees the other 5 rows of its group (2 minority, 3
    majority), so it is borderline and D3 = 1/3. Keeping the row itself as a
    neighbour would count 3 minority neighbours and call it safe.
    """
    groups = np.arange(8, dtype=float)[:, None] * np.array([[50.0, 50.0]])
    X = np.vstack([np.repeat(groups, 3, axis=0), np.repeat(groups, 3, axis=0)])
    y = np.array([1] * 24 + [0] * 24)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    result = compute_d3(ds, k=5)
    assert result.components["n_borderline"] == 24
    assert result.value == pytest.approx(1 / 3)


def test_cache_never_returns_self():
    rng = np.random.default_rng(3)
    X = np.round(rng.normal(size=(200, 2)), 1)
    y = np.array([1] * 30 + [0] * 170)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    cache = _KNNCache(ds, k=5, chunk_size=16)
    _, idx_all = cache.query_all()
    assert not np.any(idx_all == np.arange(200)[:, None])
    _, idx_min = cache.query_minority()
    assert not np.any(idx_min == np.flatnonzero(ds.minority_mask)[:, None])
    assert cache.query_all() is cache.query_all()  # cached


def test_cache_clamps_k_and_rejects_non_positive():
    ds = _twin_pairs(5)
    assert _KNNCache(ds, k=50).k == ds.N - 1
    with pytest.raises(ValueError, match="k must be > 0"):
        _KNNCache(ds, k=0)


def test_n2_same_class_twin_is_intra_neighbour_at_zero():
    X = np.array([[0.0], [0.0], [10.0], [10.0], [5.0], [5.2]])
    y = np.array([0, 0, 1, 1, 0, 1])
    # intra: 0, 0, 0, 0, 5, 4.8 ; inter: 5.2, 5.2, 5, 5, 0.2, 0.2
    expected_raw = (5 + 4.8) / (5.2 + 5.2 + 5 + 5 + 0.2 + 0.2)
    assert compute_n2(X, y) == pytest.approx(expected_raw / (1 + expected_raw))
