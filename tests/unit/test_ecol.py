"""Unit tests for ECoL measures: F3, N1, N2, L1."""

import numpy as np
import pytest

from cipa.ecol import (
    compute_f3,
    compute_l1,
    compute_n1,
    compute_n2,
    euclidean_minimum_spanning_tree,
)


def two_gaussians(n0=50, n1=50, sep=5.0, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(n0, d))
    X1 = rng.normal(loc=sep, scale=0.5, size=(n1, d))
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def overlapping(n=100, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------

def test_f3_separable_is_zero():
    X, y = two_gaussians(sep=10.0)
    assert compute_f3(X, y) == 0.0


def test_f3_overlapping_is_high():
    X, y = overlapping()
    assert compute_f3(X, y) > 0.5


def test_f3_in_range():
    X, y = two_gaussians(sep=1.0)
    assert 0.0 <= compute_f3(X, y) <= 1.0


def test_f3_single_feature():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array([0, 0, 1, 1])
    assert compute_f3(X, y) == 0.0


# ---------------------------------------------------------------------------
# N1 — exact Euclidean MST without a dense matrix (C4)
# ---------------------------------------------------------------------------

def _dense_n1(X, y):
    """v1.2.1 reference: scipy MST on the dense matrix (valid without duplicates)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    mst = minimum_spanning_tree(squareform(pdist(X))).tocoo()
    border = set()
    for i, j in zip(mst.row, mst.col, strict=True):
        if y[i] != y[j]:
            border.update((int(i), int(j)))
    return len(border) / len(X), float(mst.data.sum())


def _kruskal_reference(X):
    """Dense Kruskal with the documented key (w, min(i, j), max(i, j)); zero edges valid."""
    from scipy.spatial.distance import pdist, squareform

    n = len(X)
    D = squareform(pdist(X))
    rows, cols = np.triu_indices(n, 1)
    w = D[rows, cols]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = set()
    for e in np.lexsort((cols, rows, w)):
        a, b = find(rows[e]), find(cols[e])
        if a != b:
            parent[max(a, b)] = min(a, b)
            edges.add((int(rows[e]), int(cols[e])))
            if len(edges) == n - 1:
                break
    return edges, float(sum(D[i, j] for i, j in edges))


def test_n1_separable_is_near_zero():
    # MST always has ≥1 inter-class edge; for well-separated data it's exactly 1 → n1 = 2/N
    X, y = two_gaussians(sep=10.0)
    assert compute_n1(X, y) == pytest.approx(2 / len(X))


def test_n1_overlapping_is_high():
    X, y = overlapping()
    assert compute_n1(X, y) > 0.3


def test_n1_in_range():
    X, y = two_gaussians(sep=1.0)
    assert 0.0 <= compute_n1(X, y) <= 1.0


@pytest.mark.parametrize(
    "n,d,sep,n_neighbors,seed",
    [(60, 2, 0.5, 1, 0), (500, 4, 1.0, 16, 1), (1200, 12, 2.0, 5, 2), (2000, 30, 0.0, 16, 3),
     (900, 1, 3.0, 3, 4)],
)
def test_n1_equals_dense_method_without_duplicates(n, d, sep, n_neighbors, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, n)
    X[y == 1] += sep
    expected_n1, expected_weight = _dense_n1(X, y)
    rows, cols, weights = euclidean_minimum_spanning_tree(X, n_neighbors=n_neighbors)
    assert len(rows) == n - 1
    assert np.all(rows < cols)
    assert weights.sum() == pytest.approx(expected_weight, rel=1e-12)
    assert compute_n1(X, y, n_neighbors=n_neighbors) == expected_n1


@pytest.mark.parametrize("seed", range(6))
def test_mst_with_duplicates_and_ties_matches_keyed_reference(seed):
    rng = np.random.default_rng(100 + seed)
    n = 250
    X = rng.normal(size=(n, 3))
    X[rng.integers(0, n, n // 2)] = X[rng.integers(0, n, n // 2)]  # exact duplicates
    if seed % 2:
        X = np.round(X, 1)  # many equal non-zero distances
    rows, cols, weights = euclidean_minimum_spanning_tree(X, n_neighbors=int(rng.integers(1, 12)))
    edges, total = _kruskal_reference(X)
    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == edges
    assert weights.sum() == pytest.approx(total, rel=1e-12)


def test_mst_small_and_degenerate_inputs():
    rows, cols, weights = euclidean_minimum_spanning_tree(np.zeros((1, 2)))
    assert len(rows) == len(cols) == len(weights) == 0
    rows, cols, weights = euclidean_minimum_spanning_tree(np.zeros((5, 2)))
    assert len(rows) == 4 and np.all(weights == 0.0)


def test_mst_exact_search_uses_tree_for_large_components():
    """Two dense far-apart blobs force the exact search of many unresolved points."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(size=(400, 2)), rng.normal(size=(400, 2)) + 50.0])
    X[5] = X[6]  # a tie inside a blob
    rows, cols, weights = euclidean_minimum_spanning_tree(X, n_neighbors=2)
    edges, total = _kruskal_reference(X)
    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == edges
    assert weights.sum() == pytest.approx(total, rel=1e-12)


def test_mst_exact_search_resolves_ties_in_tree_path():
    """Equidistant candidates outside a large component: the smallest index wins."""
    blob = np.array([[x, y] for x in range(10) for y in range(10)], dtype=float)
    far = np.array([[4.5, 30.0], [4.5, -21.0], [30.0, 4.5], [-21.0, 4.5]])
    X = np.vstack([blob, far])
    rows, cols, weights = euclidean_minimum_spanning_tree(X, n_neighbors=2)
    edges, total = _kruskal_reference(X)
    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == edges
    assert weights.sum() == pytest.approx(total)


def test_n1_twins_with_different_labels_are_both_borderline():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 2)) * 10
    y = np.zeros(40, dtype=int)
    X = np.vstack([X, X[:1]])  # row 40 duplicates row 0
    y = np.append(y, 1)
    rows, cols, _ = euclidean_minimum_spanning_tree(X)
    assert (0, 40) in set(zip(rows.tolist(), cols.tolist(), strict=True))
    assert compute_n1(X, y) == pytest.approx(2 / 41)


def test_n1_memory_is_bounded_at_50k():
    """N = 50,000: a dense N×N float matrix would need 20 GB."""
    import tracemalloc

    rng = np.random.default_rng(11)
    X = rng.normal(size=(50_000, 3))
    y = rng.integers(0, 2, 50_000)
    tracemalloc.start()
    try:
        n1 = compute_n1(X, y)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert 0.0 < n1 < 1.0
    assert peak < 500 * 2**20, f"peak traced memory {peak / 2**20:.0f} MiB"


# ---------------------------------------------------------------------------
# N2
# ---------------------------------------------------------------------------

def test_n2_separable_is_low():
    X, y = two_gaussians(sep=10.0)
    # Separated: intra << inter → N2_raw small → N2norm = N2_raw/(1+N2_raw) small
    assert compute_n2(X, y) < 0.15


def test_n2_overlapping_is_high():
    X, y = overlapping()
    assert compute_n2(X, y) > 0.3


def test_n2_in_range():
    X, y = two_gaussians(sep=2.0)
    assert 0.0 <= compute_n2(X, y) <= 1.0


def test_n2_all_inter_distances_zero_returns_zero():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    assert compute_n2(X, y) == 0.0


# ---------------------------------------------------------------------------
# L1 (C5)
# ---------------------------------------------------------------------------

def test_l1_linearly_separable_is_low():
    X, y = two_gaussians(sep=10.0)
    l1, converged, n_iter = compute_l1(X, y, random_state=0)
    assert converged
    assert l1 < 0.1
    assert 0 < n_iter < 10_000


def test_l1_overlapping_is_higher():
    X, y = overlapping()
    l1, _, _ = compute_l1(X, y, random_state=0)
    assert l1 > 0.2


def test_l1_in_range():
    X, y = two_gaussians(sep=1.0)
    l1, _, _ = compute_l1(X, y, random_state=0)
    assert 0.0 <= l1 <= 1.0


def test_l1_default_max_iter_is_10000():
    import inspect

    assert inspect.signature(compute_l1).parameters["max_iter"].default == 10_000


def test_l1_default_tol_is_exposed():
    """svc_tol is part of what the reported L1 means when the fit stops at the cap."""
    import inspect

    assert inspect.signature(compute_l1).parameters["tol"].default == 1e-4


def test_l1_a_looser_tolerance_stops_sooner():
    X, y = two_gaussians(sep=1.0, n0=200, n1=200)
    _, _, tight = compute_l1(X, y, random_state=0, tol=1e-6)
    _, _, loose = compute_l1(X, y, random_state=0, tol=1e-1)
    assert loose <= tight


def test_l1_non_convergence_reports_error_rate_not_fixed_half():
    """Without convergence L1 is the training error of the last iterate (no 0.5 fallback)."""
    import warnings

    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40)) * rng.uniform(1, 1e3, 40)
    y = (X[:, 0] + rng.normal(scale=200, size=300) > 0).astype(int)
    y[:10] = 1

    l1, converged, n_iter = compute_l1(X, y, max_iter=1, random_state=0)
    assert converged is False
    # n_iter pins the cap: this is what tells a consumer the fit is incomplete.
    assert n_iter == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc = LinearSVC(class_weight="balanced", max_iter=1, random_state=0, dual="auto").fit(X, y)
    assert l1 == pytest.approx(float(np.mean(svc.predict(X) != y)))
    assert l1 != 0.5
