"""Integration tests for CIPAPipeline (SPEC-12).

These tests exercise the full C → I → P → A pipeline end-to-end using
synthetic datasets whose expected outputs are well-understood.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline, CIPAResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dataset(
    n_minority: int,
    n_majority: int,
    d: int = 4,
    sep: float = 5.0,
    seed: int = 0,
    name: str = "test",
) -> CIPADataset:
    """Two well-separated Gaussians with controlled imbalance ratio."""
    rng = np.random.default_rng(seed)
    X_min = rng.normal(loc=sep, scale=0.5, size=(n_minority, d))
    X_maj = rng.normal(loc=0.0, scale=0.5, size=(n_majority, d))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * n_minority + [0] * n_majority)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name=name)


def make_overlapping(
    n_minority: int = 50,
    n_majority: int = 200,
    seed: int = 1,
) -> CIPADataset:
    """Both classes from the same distribution — maximum overlap."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_minority + n_majority, 4))
    y = np.array([1] * n_minority + [0] * n_majority)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name="overlapping")


# ---------------------------------------------------------------------------
# Basic pipeline smoke tests
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    def test_run_returns_cipa_result(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        assert isinstance(result, CIPAResult)

    def test_result_has_all_fields(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        assert result.difficulty_score is not None
        assert result.profile is not None
        assert result.action is not None

    def test_dataset_name_propagated(self):
        ds = make_dataset(20, 100, name="my_dataset")
        result = CIPAPipeline().run(ds)
        assert result.dataset_name == "my_dataset"

    def test_result_serializable(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_ds_in_unit_interval(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        assert 0.0 <= result.difficulty_score.value <= 1.0

    def test_seven_dimensions_returned(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        assert len(result.difficulty_score.dimensions) == 7
        ids = [d.dimension_id for d in result.difficulty_score.dimensions]
        assert ids == [f"D{i}" for i in range(1, 8)]

    def test_all_dimension_values_in_range(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        for dim in result.difficulty_score.dimensions:
            assert 0.0 <= dim.value <= 1.0, f"{dim.dimension_id}={dim.value} out of range"

    def test_signature_is_valid(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        assert result.profile.signature in {"I", "II", "III", "IV", "V"}

    def test_band_matches_ds_value(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run(ds)
        v = result.difficulty_score.value
        band = result.difficulty_score.band
        if v < 0.25:
            assert band == "Low"
        elif v < 0.50:
            assert band == "Moderate"
        elif v < 0.75:
            assert band == "High"
        else:
            assert band == "Extreme"

    def test_action_has_recommendations(self):
        ds = make_dataset(20, 200)
        result = CIPAPipeline().run(ds)
        assert len(result.action.evaluation_metrics) >= 2
        assert len(result.action.model_families) >= 1
        assert len(result.action.validation_protocol) >= 1


# ---------------------------------------------------------------------------
# run_dimensions_only
# ---------------------------------------------------------------------------

class TestRunDimensionsOnly:
    def test_returns_seven_results(self):
        ds = make_dataset(20, 100)
        dims = CIPAPipeline().run_dimensions_only(ds)
        assert len(dims) == 7

    def test_ids_in_order(self):
        ds = make_dataset(20, 100)
        dims = CIPAPipeline().run_dimensions_only(ds)
        assert [d.dimension_id for d in dims] == [f"D{i}" for i in range(1, 8)]

    def test_values_match_full_run(self):
        ds = make_dataset(20, 100, seed=7)
        pipe = CIPAPipeline(random_state=0)
        dims_only = pipe.run_dimensions_only(ds)
        full = pipe.run(ds)
        for a, b in zip(dims_only, full.difficulty_score.dimensions):
            assert a.value == pytest.approx(b.value, abs=1e-9)


# ---------------------------------------------------------------------------
# run_scoring_only
# ---------------------------------------------------------------------------

class TestRunScoringOnly:
    def test_returns_difficulty_score(self):
        from cipa import DifficultyScore
        ds = make_dataset(20, 100)
        score = CIPAPipeline().run_scoring_only(ds)
        assert isinstance(score, DifficultyScore)

    def test_value_matches_full_run(self):
        ds = make_dataset(20, 100, seed=9)
        pipe = CIPAPipeline(random_state=0)
        score = pipe.run_scoring_only(ds)
        full = pipe.run(ds)
        assert score.value == pytest.approx(full.difficulty_score.value, abs=1e-9)


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_custom_weights_change_ds(self):
        ds = make_dataset(20, 100)
        default = CIPAPipeline().run_scoring_only(ds).value
        # All weight on D1
        custom = CIPAPipeline(weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).run_scoring_only(ds)
        assert default != pytest.approx(custom.value, abs=0.01)

    def test_uniform_weights_accepted(self):
        ds = make_dataset(20, 100)
        w = tuple([1 / 7] * 7)
        result = CIPAPipeline(weights=w).run(ds)
        assert 0.0 <= result.difficulty_score.value <= 1.0


# ---------------------------------------------------------------------------
# Semantic sanity: well-separated vs overlapping
# ---------------------------------------------------------------------------

class TestSemanticSanity:
    def test_separated_has_lower_d2_than_overlapping(self):
        pipe = CIPAPipeline(random_state=0)
        sep = make_dataset(50, 200, sep=8.0, seed=42, name="sep")
        over = make_overlapping(seed=42)
        d2_sep = pipe.run_dimensions_only(sep)[1].value
        d2_over = pipe.run_dimensions_only(over)[1].value
        assert d2_sep < d2_over

    def test_high_imbalance_has_high_d1(self):
        pipe = CIPAPipeline()
        ds = make_dataset(5, 500, sep=5.0, seed=0)
        dims = pipe.run_dimensions_only(ds)
        assert dims[0].value > 0.7  # D1 should be high

    def test_balanced_has_low_d1(self):
        pipe = CIPAPipeline()
        ds = make_dataset(100, 100, sep=5.0, seed=0)
        dims = pipe.run_dimensions_only(ds)
        assert dims[0].value < 0.1  # IR=1 → D1≈0

    def test_low_complexity_gives_low_ds(self):
        """Well-separated, mild imbalance → Low or Moderate difficulty."""
        pipe = CIPAPipeline()
        ds = make_dataset(50, 100, sep=8.0, seed=0)
        result = pipe.run(ds)
        assert result.difficulty_score.band in {"Low", "Moderate"}

    def test_high_complexity_gives_high_ds(self):
        """Fully overlapping with severe imbalance → High or Extreme."""
        pipe = CIPAPipeline()
        ds = make_overlapping(n_minority=20, n_majority=500, seed=5)
        result = pipe.run(ds)
        assert result.difficulty_score.band in {"High", "Extreme"}


# ---------------------------------------------------------------------------
# random_state reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        ds = make_dataset(30, 150, seed=0)
        r1 = CIPAPipeline(random_state=42).run(ds)
        r2 = CIPAPipeline(random_state=42).run(ds)
        assert r1.difficulty_score.value == pytest.approx(r2.difficulty_score.value, abs=1e-9)
        assert r1.profile.signature == r2.profile.signature


# ---------------------------------------------------------------------------
# knn_subsample: selective subsampling for expensive dimensions
# ---------------------------------------------------------------------------

class TestKnnSubsample:
    def test_knn_subsample_returns_valid_result(self):
        """Pipeline with knn_subsample returns a valid CIPAResult."""
        ds = make_dataset(30, 270, seed=0)
        result = CIPAPipeline(knn_subsample=100, random_state=0).run(ds)
        assert isinstance(result, CIPAResult)
        assert 0.0 <= result.difficulty_score.value <= 1.0

    def test_knn_subsample_none_equals_full(self):
        """knn_subsample=None (default) must produce same result as running full."""
        ds = make_dataset(20, 80, seed=0)
        r_full = CIPAPipeline(random_state=0).run(ds)
        r_no_sub = CIPAPipeline(knn_subsample=None, random_state=0).run(ds)
        assert r_full.difficulty_score.value == pytest.approx(r_no_sub.difficulty_score.value, abs=1e-9)

    def test_knn_subsample_larger_than_n_uses_full(self):
        """knn_subsample ≥ N → no subsampling → identical to knn_subsample=None."""
        ds = make_dataset(20, 80, seed=0)  # N=100
        r_none = CIPAPipeline(random_state=0).run(ds)
        r_big  = CIPAPipeline(knn_subsample=10_000, random_state=0).run(ds)
        assert r_none.difficulty_score.value == pytest.approx(r_big.difficulty_score.value, abs=1e-9)

    def test_knn_subsample_d1_reflects_full_dataset(self):
        """D1 is computed on the full dataset even when knn_subsample is active.

        The full dataset has extreme IR; with subsampling D1 would be lower.
        We verify D1 equals what compute_d1 gives on the full dataset.
        """
        from cipa.dimensions import compute_d1
        ds = make_dataset(10, 490, seed=0)  # N=500, IR=49
        full_d1 = compute_d1(ds).value

        pipe = CIPAPipeline(knn_subsample=50, random_state=0)
        dims = pipe.run_dimensions_only(ds)
        pipeline_d1 = dims[0].value  # D1 is first dimension

        assert pipeline_d1 == pytest.approx(full_d1, abs=1e-9)

    def test_knn_subsample_all_dimensions_in_range(self):
        """All 7 dimensions are in [0, 1] with knn_subsample active."""
        ds = make_dataset(20, 180, seed=1)
        dims = CIPAPipeline(knn_subsample=50, random_state=0).run_dimensions_only(ds)
        for dim in dims:
            assert 0.0 <= dim.value <= 1.0, f"{dim.dimension_id}={dim.value} out of range"

    def test_knn_subsample_reproducible(self):
        """Same seed → same result with knn_subsample."""
        ds = make_dataset(20, 180, seed=0)
        r1 = CIPAPipeline(knn_subsample=50, random_state=7).run(ds)
        r2 = CIPAPipeline(knn_subsample=50, random_state=7).run(ds)
        assert r1.difficulty_score.value == pytest.approx(r2.difficulty_score.value, abs=1e-9)
