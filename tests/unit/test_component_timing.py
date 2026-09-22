"""Per-component timing of D2 and D7 (2.0.0rc3, P3).

The timing pilot had to instrument D2 and D7 from outside to find out that D3
and L1 were the bottlenecks. These tests pin the instrumentation that makes
that unnecessary.
"""

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline
from cipa.dimensions import compute_d2, compute_d7


def dataset(n=400, d=25, minority=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.zeros(n, dtype=int)
    y[:minority] = 1
    return CIPADataset(X, y, minority_label=1, majority_label=0)


@pytest.mark.parametrize(
    ("compute", "expected"),
    [(compute_d2, {"F3", "N1", "kDN"}), (compute_d7, {"L1", "N2"})],
)
def test_each_component_is_timed_separately(compute, expected):
    result = compute(dataset())
    timings = result.metadata["component_seconds"]
    assert set(timings) == expected
    assert all(v >= 0.0 for v in timings.values())


@pytest.mark.parametrize("n_subsamples", [1, 3])
def test_component_seconds_sum_to_the_dimension_total(n_subsamples):
    """Summed, not averaged, so the breakdown reconciles with time_seconds."""
    pipeline = CIPAPipeline(n_subsamples=n_subsamples, n_max=300)
    dims = pipeline.run_dimensions_only(dataset(n=600))
    for dim in dims:
        if dim.dimension_id not in ("D2", "D7"):
            continue
        timings = dim.metadata["component_seconds"]
        total = dim.metadata["time_seconds"]
        assert sum(timings.values()) == pytest.approx(total, rel=0.10, abs=0.05)


def test_a_warm_cache_moves_cost_out_of_kdn():
    """The index build is charged to kDN only when D2 is what fits it.

    This is the figure the docstring warns about: under the pipeline D3 warms
    the cache first, so kDN looks far cheaper than in a standalone call.
    """
    from cipa._knn import _KNNCache

    ds = dataset(n=1500, d=30, minority=200)
    cold = compute_d2(ds).metadata["component_seconds"]["kDN"]

    cache = _KNNCache(ds, k=5)
    cache.query_all()  # pay for the index up front, as D3 would
    warm = compute_d2(ds, knn_cache=cache).metadata["component_seconds"]["kDN"]

    assert warm < cold


def test_timings_do_not_alter_the_values():
    ds = dataset()
    assert compute_d2(ds).value == pytest.approx(compute_d2(ds).value)
    assert compute_d7(ds, random_state=0).value == pytest.approx(
        compute_d7(ds, random_state=0).value
    )
