"""Unit tests for the shared neighbour-algorithm selection (2.0.0rc3, P1)."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from cipa import CIPADataset, CIPAPipeline
from cipa._constants import KD_TREE_MAX_DIM, KNN_ALGORITHM_OPTIONS
from cipa._knn import _KNNCache, kneighbors_excluding_self, select_knn_algorithm
from cipa.ecol import compute_n1, compute_n2
from cipa.ecol.n2 import (
    _classes_coincide_exactly,
    _duplicate_rounding_bound,
)


def labelled(n=200, d=4, minority=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=minority, replace=False)] = 1
    return X, y


def with_duplicates_and_ties(seed=0):
    """Rows that repeat exactly, plus points equidistant from a query.

    Duplicates make neighbour identity ambiguous and ties make the order
    ambiguous, which is where two exact algorithms are allowed to disagree.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(40, 6))
    X = np.vstack([base, base[:15]])            # 15 exact duplicates
    X = np.vstack([X, np.eye(6), -np.eye(6)])   # 12 points equidistant from the origin
    X = np.vstack([X, np.zeros((1, 6))])        # the query at the origin
    y = np.zeros(len(X), dtype=int)
    y[::3] = 1
    return X, y


# --- the rule -------------------------------------------------------------

@pytest.mark.parametrize(
    ("n_features", "expected"),
    [(1, "kd_tree"), (4, "kd_tree"), (15, "kd_tree"), (16, "brute"), (70, "brute"),
     (432, "brute"), (5000, "brute")],
)
def test_auto_rule_splits_at_the_threshold(n_features, expected):
    assert select_knn_algorithm(n_features) == expected


def test_threshold_constant_is_the_boundary_used():
    assert select_knn_algorithm(KD_TREE_MAX_DIM) == "kd_tree"
    assert select_knn_algorithm(KD_TREE_MAX_DIM + 1) == "brute"


@pytest.mark.parametrize("algorithm", ["ball_tree", "kd_tree", "brute"])
def test_explicit_algorithm_is_returned_unchanged(algorithm):
    assert select_knn_algorithm(4, algorithm) == algorithm
    assert select_knn_algorithm(432, algorithm) == algorithm


def test_every_option_is_accepted():
    for algorithm in KNN_ALGORITHM_OPTIONS:
        assert select_knn_algorithm(8, algorithm) in ("ball_tree", "kd_tree", "brute")


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="algorithm must be one of"):
        select_knn_algorithm(8, "kdtree")


# --- the three call sites resolve through it ------------------------------

@pytest.mark.parametrize(("d", "expected"), [(4, "kd_tree"), (40, "brute")])
def test_cache_resolves_auto_from_the_feature_count(d, expected):
    X, y = labelled(d=d)
    cache = _KNNCache(CIPADataset(X, y, minority_label=1, majority_label=0), k=5)
    assert cache.algorithm == expected


def test_cache_honours_a_forced_algorithm():
    X, y = labelled(d=40)
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    assert _KNNCache(ds, k=5, algorithm="ball_tree").algorithm == "ball_tree"


def test_cache_rejects_an_unknown_algorithm():
    X, y = labelled()
    ds = CIPADataset(X, y, minority_label=1, majority_label=0)
    with pytest.raises(ValueError, match="algorithm must be one of"):
        _KNNCache(ds, k=5, algorithm="tree")


@pytest.mark.parametrize("measure", [compute_n1, compute_n2])
@pytest.mark.parametrize("d", [4, 40])
def test_n1_and_n2_accept_and_honour_the_parameter(measure, d):
    X, y = labelled(d=d)
    auto = measure(X, y)
    for algorithm in ("ball_tree", "kd_tree", "brute"):
        assert measure(X, y, algorithm=algorithm) == pytest.approx(auto, abs=1e-12)


# --- equivalence where it is allowed to be hard ---------------------------

@pytest.mark.parametrize("k", [1, 3, 5])
def test_brute_and_ball_tree_return_the_same_neighbour_sets(k):
    """Duplicates and ties may reorder neighbours; the sets and distances match."""
    X, _ = with_duplicates_and_ties()
    idx = np.arange(len(X))
    out = {}
    for algorithm in ("ball_tree", "brute"):
        nn = NearestNeighbors(algorithm=algorithm).fit(X)
        out[algorithm] = kneighbors_excluding_self(nn, X, idx, k)

    d_ball, d_brute = out["ball_tree"][0], out["brute"][0]
    # Distances agree to the rounding of the dot-product formulation brute
    # force uses. The discrepancy is concentrated on exact duplicates, where
    # a tree returns 0 and brute returns ~6e-08; see
    # test_brute_force_reports_a_tiny_positive_distance_for_duplicates.
    assert np.allclose(d_ball, d_brute, atol=1e-7)
    # Neighbour identity is deliberately not compared. With exact duplicates
    # either twin is equally correct, and where more than k rows share the
    # k-th distance both algorithms are exact while selecting different
    # subsets. What must agree is the distances, above, and each algorithm
    # must be self-consistent: the index it returns really is at the distance
    # it reports. That is what the measures consume.
    for algorithm, (dist, idx_) in out.items():
        recomputed = np.linalg.norm(X[:, None, :] - X[idx_], axis=2)
        recomputed = recomputed[np.arange(len(X))[:, None], np.arange(k)[None, :]]
        assert np.allclose(recomputed, dist, atol=1e-7), algorithm


def test_kd_tree_matches_ball_tree_exactly():
    """Both walk the tree with true distances, so they agree bit for bit."""
    X, _ = with_duplicates_and_ties()
    idx = np.arange(len(X))
    out = {}
    for algorithm in ("ball_tree", "kd_tree"):
        nn = NearestNeighbors(algorithm=algorithm).fit(X)
        out[algorithm] = kneighbors_excluding_self(nn, X, idx, 5)
    assert np.array_equal(out["ball_tree"][0], out["kd_tree"][0])


def test_brute_force_can_report_a_nonzero_distance_for_an_exact_duplicate():
    """Pins the one measured difference between brute force and the trees.

    Brute force evaluates ``x.x - 2x.y + y.y`` rather than the norm of the
    difference, so for an exact duplicate the cancellation can leave a tiny
    positive value where a tree returns exactly 0. It depends on the row's
    magnitude: with unit basis vectors the arithmetic is exact, with general
    rows it is not. Any guard that tests a distance against exact zero is
    therefore algorithm-dependent, which is why this test exists.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 6))
    X = np.vstack([base, base[:10]])  # rows 30..39 duplicate rows 0..9
    idx = np.arange(len(X))
    d_tree, _ = kneighbors_excluding_self(
        NearestNeighbors(algorithm="ball_tree").fit(X), X, idx, 1
    )
    d_brute, _ = kneighbors_excluding_self(
        NearestNeighbors(algorithm="brute").fit(X), X, idx, 1
    )
    duplicated = np.r_[np.arange(10), np.arange(30, 40)]
    assert np.all(d_tree[duplicated, 0] == 0.0)
    assert np.any(d_brute[duplicated, 0] > 0.0), "no rounding observed; fixture too benign"
    assert np.all(d_brute[duplicated, 0] < 1e-7)

    # Exact basis vectors are the benign case, kept so the bound is not
    # mistaken for a guarantee in either direction.
    E = np.vstack([np.eye(6), np.eye(6)[:3]])
    d_exact, _ = kneighbors_excluding_self(
        NearestNeighbors(algorithm="brute").fit(E), E, np.arange(len(E)), 1
    )
    assert np.all(d_exact[[0, 1, 2, 6, 7, 8], 0] == 0.0)


def test_n1_and_n2_agree_across_algorithms_with_duplicates_and_ties():
    X, y = with_duplicates_and_ties()
    for measure in (compute_n1, compute_n2):
        values = {a: measure(X, y, algorithm=a) for a in ("ball_tree", "kd_tree", "brute")}
        assert max(values.values()) - min(values.values()) < 1e-9, values


# --- chunk_size still bounds the query blocks ----------------------------

@pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
def test_chunking_does_not_change_brute_force_results(chunk_size):
    """chunk_size bounds brute force memory, so it must not alter the answer."""
    X, _ = labelled(n=120, d=40)
    nn = NearestNeighbors(algorithm="brute").fit(X)
    idx = np.arange(len(X))
    d_ref, i_ref = kneighbors_excluding_self(nn, X, idx, 5, chunk_size=len(X))
    d, i = kneighbors_excluding_self(nn, X, idx, 5, chunk_size=chunk_size)
    assert np.allclose(d, d_ref)
    assert np.array_equal(i, i_ref)


# --- N2 degeneracy: structural, not distance-based (2.0.0rc3) -------------


def coinciding_classes(n=40, d=8, seed=0):
    """Every row has an exact duplicate in the opposite class: truly degenerate."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    X = np.vstack([base, base])
    y = np.array([0] * n + [1] * n)
    return X, y


def almost_coinciding_classes(offset, n=40, d=8, seed=0):
    """Classes separated by a real but minuscule margin: the hardest case."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    X = np.vstack([base, base + offset])
    y = np.array([0] * n + [1] * n)
    return X, y


@pytest.mark.parametrize("algorithm", ["ball_tree", "kd_tree", "brute"])
def test_coinciding_classes_are_degenerate_under_every_algorithm(algorithm):
    X, y = coinciding_classes()
    assert compute_n2(X, y, algorithm=algorithm) == 0.0


@pytest.mark.parametrize("algorithm", ["ball_tree", "kd_tree", "brute"])
def test_a_minuscule_but_real_boundary_is_not_degenerate(algorithm):
    """Below the rounding floor, so the old exact guard could not tell them apart.

    The classes do not coincide, so N2 must report a hard boundary rather than
    the easiest possible one.
    """
    X, y = almost_coinciding_classes(offset=1e-9)
    value = compute_n2(X, y, algorithm=algorithm)
    assert value > 0.99, value


def test_the_rounding_floor_is_what_separates_the_two_cases():
    """The bound is crossed by the near-degenerate case, which is the point."""
    X_deg, _ = coinciding_classes()
    bound = _duplicate_rounding_bound(X_deg)
    assert bound > 0.0
    # A per-row offset of 1e-9 over 80 rows sums to ~2.3e-7, under the bound:
    # distances alone cannot distinguish the two cases.
    assert 80 * 1e-9 * np.sqrt(8) < bound


def test_structural_check_is_exact_on_both_sides():
    X_deg, y_deg = coinciding_classes()
    assert _classes_coincide_exactly(X_deg, y_deg)
    X_near, y_near = almost_coinciding_classes(offset=1e-9)
    assert not _classes_coincide_exactly(X_near, y_near)


def test_structural_check_needs_a_twin_for_every_row():
    """One unmatched row is enough to make the dataset non-degenerate."""
    X, y = coinciding_classes(n=10)
    X = X.copy()
    X[0] += 1.0  # row 0 of class 0 no longer has a twin in class 1
    assert not _classes_coincide_exactly(X, y)


def test_partial_overlap_is_not_degenerate():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20, 5))
    X = np.vstack([base, base[:10], rng.normal(size=(10, 5)) + 50.0])
    y = np.array([0] * 20 + [1] * 20)
    assert not _classes_coincide_exactly(X, y)
    assert compute_n2(X, y) > 0.0


def test_bound_grows_with_row_magnitude():
    small = np.ones((10, 4))
    large = np.ones((10, 4)) * 100.0
    assert _duplicate_rounding_bound(large) > _duplicate_rounding_bound(small)


def test_bound_of_an_empty_matrix_is_zero():
    assert _duplicate_rounding_bound(np.empty((0, 3))) == 0.0


# --- the pipeline surface (2.0.0rc3, P1.3) --------------------------------


def pipeline_dataset(n=300, d=40, minority=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.zeros(n, dtype=int)
    y[:minority] = 1
    return CIPADataset(X, y, minority_label=1, majority_label=0)


def test_pipeline_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="algorithm must be one of"):
        CIPAPipeline(algorithm="kdtree")


@pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
def test_pipeline_records_what_was_asked_and_what_was_resolved(algorithm):
    result = CIPAPipeline(algorithm=algorithm, n_subsamples=1).run(pipeline_dataset())
    assert result.metadata["algorithm"] == algorithm
    resolved = result.metadata["algorithm_resolved"]
    assert resolved == select_knn_algorithm(result.metadata["d"], algorithm)
    assert resolved in ("ball_tree", "kd_tree", "brute")
    assert result.metadata["chunk_size"] == CIPAPipeline()._chunk_size


def test_pipeline_default_is_auto_and_resolves_to_brute_in_high_dimension():
    result = CIPAPipeline(n_subsamples=1).run(pipeline_dataset(d=40))
    assert result.metadata["algorithm"] == "auto"
    assert result.metadata["algorithm_resolved"] == "brute"


def test_pipeline_gives_the_same_score_under_every_algorithm():
    """The choice is a performance decision; it must not move the index."""
    dataset = pipeline_dataset()
    scores = {
        algorithm: float(
            CIPAPipeline(algorithm=algorithm, n_subsamples=1).run(dataset).difficulty_score.value
        )
        for algorithm in ("auto", "ball_tree", "kd_tree", "brute")
    }
    assert max(scores.values()) - min(scores.values()) == pytest.approx(0.0, abs=1e-12), scores


@pytest.mark.parametrize("chunk_size", [64, 100_000])
def test_pipeline_chunk_size_reaches_the_measures_without_changing_them(chunk_size):
    """chunk_size now threads through D2 and D7 into N1 and N2."""
    dataset = pipeline_dataset(n=200, d=20)
    result = CIPAPipeline(chunk_size=chunk_size, n_subsamples=1).run(dataset)
    reference = CIPAPipeline(n_subsamples=1).run(dataset)
    assert result.metadata["chunk_size"] == chunk_size
    assert float(result.difficulty_score.value) == pytest.approx(
        float(reference.difficulty_score.value), abs=1e-12
    )


# --- what actually bounds brute-force memory (2.0.0rc3, P1.4) ------------


def test_brute_force_euclidean_uses_the_argkmin_reduction():
    """Canary for the memory claim in the 2.0.0rc3 CHANGELOG.

    Since scikit-learn 1.1, ``kneighbors(algorithm="brute")`` on float64
    Euclidean data dispatches to the Cython ``ArgKmin`` reduction, which
    streams with a per-thread heap instead of materialising a
    ``chunk_size x n_fit`` distance block. That is why switching the default
    to brute force above 15 features costs no memory, and why ``chunk_size``
    bounds only the output arrays.

    If a future scikit-learn stops dispatching here, the fallback is
    ``pairwise_distances_chunked``, whose footprint is governed by
    ``sklearn.get_config()["working_memory"]`` and would need to be measured
    again. This test fails first, so that is not discovered in a run.
    """
    from sklearn.metrics._pairwise_distances_reduction import ArgKmin

    X, _ = labelled(n=200, d=40)
    assert ArgKmin.is_usable_for(X, X, "euclidean")
