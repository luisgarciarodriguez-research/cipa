"""Unit and pipeline tests for C6 (per-dimension subsampling) and C9 (determinism)."""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline
from cipa._subsampling import (
    STREAM_D2_D7,
    STREAM_D4,
    derive_seed,
    median_iqr,
    minority_indices,
    stratified_indices,
    validate_random_state,
)


def make_dataset(n_min: int, n_maj: int, d: int = 3, seed: int = 0) -> CIPADataset:
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal(loc=1.5, size=(n_min, d)),
        rng.normal(loc=0.0, size=(n_maj, d)),
    ])
    y = np.array([1] * n_min + [0] * n_maj)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="sub")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_derive_seed_is_deterministic_and_stream_specific():
    assert derive_seed(42, STREAM_D2_D7, 0) == derive_seed(42, STREAM_D2_D7, 0)
    seeds = {derive_seed(42, s, i) for s in (STREAM_D2_D7, STREAM_D4) for i in range(5)}
    assert len(seeds) == 10
    assert derive_seed(42, STREAM_D2_D7, 0) != derive_seed(43, STREAM_D2_D7, 0)


def test_derive_seed_ignores_global_numpy_state():
    np.random.seed(0)
    a = derive_seed(7, STREAM_D4, 3)
    np.random.seed(999)
    np.random.random(100)
    assert derive_seed(7, STREAM_D4, 3) == a


@pytest.mark.parametrize("n_min,n_maj,n_max", [(37, 963, 250), (5, 4995, 1000), (300, 700, 999)])
def test_stratified_indices_preserve_ir_within_one_instance(n_min, n_maj, n_max):
    mask = np.array([True] * n_min + [False] * n_maj)
    idx = stratified_indices(mask, n_max, seed=1)
    assert len(idx) == n_max == len(np.unique(idx))
    assert np.all(np.diff(idx) > 0)
    expected = n_max * n_min / (n_min + n_maj)
    assert abs(mask[idx].sum() - expected) <= 1


def test_stratified_indices_floor_two_minority():
    mask = np.array([True] * 3 + [False] * 9997)
    idx = stratified_indices(mask, 100, seed=0)
    assert mask[idx].sum() == 2


def test_minority_indices_sorted_unique():
    idx = minority_indices(500, 120, seed=5)
    assert len(np.unique(idx)) == 120 and np.all(np.diff(idx) > 0) and idx.max() < 500


def test_median_iqr():
    assert median_iqr([1.0, 2.0, 3.0, 4.0, 5.0]) == (3.0, 2.0)


@pytest.mark.parametrize("bad", [None, -1, 1.5, "42", True])
def test_validate_random_state_rejects_invalid(bad):
    with pytest.raises(ValueError, match="random_state"):
        validate_random_state(bad)


def test_validate_random_state_accepts_numpy_int():
    assert validate_random_state(np.int64(3)) == 3


# ---------------------------------------------------------------------------
# Pipeline protocol
# ---------------------------------------------------------------------------

def test_pipeline_rejects_none_random_state():
    with pytest.raises(ValueError, match="never None"):
        CIPAPipeline(random_state=None)


def test_pipeline_default_random_state_is_42():
    result = CIPAPipeline().run_scoring_only(make_dataset(20, 80))
    assert result.metadata["random_state"] == 42


@pytest.mark.parametrize("kwargs,match", [
    ({"scaling": "minmax"}, "scaling"),
    ({"n_max": 5}, "n_max"),
    ({"n_subsamples": 0}, "n_subsamples"),
])
def test_pipeline_rejects_invalid_protocol(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CIPAPipeline(**kwargs)


def test_no_subsampling_at_or_below_n_max():
    ds = make_dataset(40, 160)
    dims = CIPAPipeline(n_max=200).run_dimensions_only(ds)
    for dim in dims:
        assert dim.metadata["subsampled"] is False
        assert dim.metadata["n_subsamples"] == 1
        assert dim.metadata["seeds"] == []
        assert dim.iqr == 0.0


def test_d2_d7_subsample_protocol_above_n_max():
    ds = make_dataset(60, 1140)
    dims = CIPAPipeline(n_max=300, n_subsamples=4, random_state=3).run_dimensions_only(ds)
    d2, d7 = dims[1], dims[6]
    for dim in (d2, d7):
        assert dim.metadata["subsampled"] is True
        assert dim.metadata["n_subsamples"] == 4
        assert dim.metadata["n_used"] == 300
        assert len(dim.metadata["seeds"]) == 4
        assert dim.metadata["n_minority_used"] == [15, 15, 15, 15]  # 300 · 60/1200
        values = dim.metadata["values"]
        assert dim.value == pytest.approx(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        assert dim.iqr == pytest.approx(q75 - q25)
    # D2 and D7 share the same draws
    assert d2.metadata["seeds"] == d7.metadata["seeds"]
    for key in ("F3", "N1", "kDN"):
        assert key in d2.metadata["components_iqr"]
        assert d2.components[key] == pytest.approx(
            np.median(d2.metadata["components_per_subsample"][key])
        )
    assert isinstance(d7.components["converged"], bool)


def test_subsample_draws_reproduce_from_recorded_seeds():
    ds = make_dataset(60, 1140)
    pipe = CIPAPipeline(n_max=300, n_subsamples=3, random_state=11, scaling="none")
    d2 = pipe.run_dimensions_only(ds)[1]
    from cipa.dimensions import compute_d2
    for seed, value in zip(d2.metadata["seeds"], d2.metadata["values"], strict=True):
        idx = stratified_indices(ds.minority_mask, 300, seed)
        assert compute_d2(ds._subset(idx)).value == value


def test_subsampling_is_deterministic_with_seed():
    ds = make_dataset(60, 1140)
    a = CIPAPipeline(n_max=300, n_subsamples=3, random_state=5).run_scoring_only(ds)
    b = CIPAPipeline(n_max=300, n_subsamples=3, random_state=5).run_scoring_only(ds)
    c = CIPAPipeline(n_max=300, n_subsamples=3, random_state=6).run_scoring_only(ds)
    va = [d.value for d in a.difficulty_score.dimensions]
    assert va == [d.value for d in b.difficulty_score.dimensions]
    assert a.difficulty_score.dimensions[1].metadata["seeds"] != \
        c.difficulty_score.dimensions[1].metadata["seeds"]


def test_d1_d5_d6_and_d3_use_all_rows_above_n_max():
    ds = make_dataset(60, 1140)
    dims = CIPAPipeline(n_max=300, random_state=0, scaling="none").run_dimensions_only(ds)
    from cipa.dimensions import compute_d1, compute_d3, compute_d5, compute_d6
    assert dims[0].value == compute_d1(ds).value
    assert dims[4].value == compute_d5(ds).value
    assert dims[5].value == compute_d6(ds, random_state=0).value
    d3 = dims[2]
    assert d3.metadata["subsampled"] is False
    assert d3.metadata["n_used"] == 1200
    assert d3.metadata["n_queries"] == 60
    counts = d3.components
    assert counts["n_safe"] + counts["n_borderline"] + counts["n_rare"] + counts["n_outlier"] == 60
    assert d3.value == compute_d3(ds).value


def test_d4_subsamples_only_above_n_max_minority():
    ds = make_dataset(60, 1140)
    below = CIPAPipeline(n_max=60, random_state=0).run_dimensions_only(ds)[3]
    assert below.metadata["subsampled"] is False
    assert below.metadata["n_used"] == 60

    above = CIPAPipeline(n_max=50, n_subsamples=3, random_state=0).run_dimensions_only(ds)[3]
    assert above.metadata["subsampled"] is True
    assert above.metadata["n_used"] == 50
    assert above.metadata["n_subsamples"] == 3
    assert len(above.metadata["run_metadata"]) == 3
    assert above.value == pytest.approx(np.median(above.metadata["values"]))
    # list-valued components stay per draw only
    assert "cluster_sizes" not in above.components
    assert len(above.metadata["components_per_subsample"]["cluster_sizes"]) == 3


def test_ds_uses_the_medians():
    ds = make_dataset(60, 1140)
    result = CIPAPipeline(n_max=300, n_subsamples=5, random_state=2).run_scoring_only(ds)
    score = result.difficulty_score
    expected = sum(w * d.value for w, d in zip(score.weights, score.dimensions, strict=True))
    assert score.value == pytest.approx(expected)


def test_n_max_can_change():
    ds = make_dataset(60, 1140)
    small = CIPAPipeline(n_max=200, n_subsamples=2).run_dimensions_only(ds)[1]
    large = CIPAPipeline(n_max=600, n_subsamples=2).run_dimensions_only(ds)[1]
    assert small.metadata["n_used"] == 200
    assert large.metadata["n_used"] == 600
