"""Integration tests for CIPAPipeline (SPEC-12).

These tests exercise the full C → I → P → A pipeline end-to-end using
synthetic datasets whose expected outputs are well-understood.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import cipa
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
        for a, b in zip(dims_only, full.difficulty_score.dimensions, strict=True):
            assert a.value == pytest.approx(b.value, abs=1e-9)


# ---------------------------------------------------------------------------
# run_scoring_only
# ---------------------------------------------------------------------------

class TestRunScoringOnly:
    def test_returns_result_without_action(self):
        ds = make_dataset(20, 100)
        result = CIPAPipeline().run_scoring_only(ds)
        assert isinstance(result, CIPAResult)
        assert result.action is None
        assert result.profile.signature in {"I", "II", "III", "IV", "V"}

    def test_value_matches_full_run(self):
        ds = make_dataset(20, 100, seed=9)
        pipe = CIPAPipeline(random_state=0)
        score = pipe.run_scoring_only(ds)
        full = pipe.run(ds)
        assert score.difficulty_score.value == full.difficulty_score.value
        assert score.profile.to_dict() == full.profile.to_dict()

    def test_handoff_call_is_serializable_with_required_fields(self):
        """The call cipa-extended makes per dataset (handoff §3)."""
        rng = np.random.default_rng(3)
        X = np.column_stack([rng.normal(size=(400, 5)), np.ones(400)])
        y = np.array([1] * 40 + [0] * 360)
        X[:40, :2] += 1.0
        w_expert = (0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13)
        result = CIPAPipeline(
            weights=w_expert, random_state=42, scaling="standard",
            n_max=300, n_subsamples=3, n_jobs=-1,
        ).run_scoring_only(CIPADataset(X, y, minority_label=1, majority_label=0, name="key"))
        d = json.loads(json.dumps(result.to_dict()))

        dims = d["difficulty_score"]["dimensions"]
        assert [x["dimension_id"] for x in dims] == [f"D{i}" for i in range(1, 8)]
        for x in dims:
            assert {"value", "iqr", "components", "metadata"} <= set(x)
            assert {"n_used", "n_subsamples", "seeds", "time_seconds"} <= set(x["metadata"])
        assert {"F3", "N1", "kDN"} <= set(dims[1]["components"])
        assert {"n_safe", "n_borderline", "n_rare", "n_outlier"} <= set(dims[2]["components"])
        assert {"ECindex", "n_clusters"} <= set(dims[3]["components"])
        assert {"L1", "N2norm", "converged"} <= set(dims[6]["components"])
        assert dims[1]["metadata"]["n_subsamples"] == 3

        meta = d["metadata"]
        assert meta["preprocessing"]["dropped_constant_columns"] == [5]
        assert meta["l1_fits"] == 3 and meta["l1_not_converged"] >= 0
        assert set(meta["time_seconds"]) >= {"preprocessing", "D1", "D7", "total"}
        assert meta["cipa_version"] == cipa.__version__
        assert d["difficulty_score"]["band"] in {"Low", "Moderate", "High", "Extreme"}
        assert d["profile"]["signature"] in {"I", "II", "III", "IV", "V"}
        assert "qualifier" in d["profile"]
        assert d["action"] is None


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_custom_weights_change_ds(self):
        ds = make_dataset(20, 100)
        default = CIPAPipeline().run_scoring_only(ds).difficulty_score.value
        # All weight on D1
        custom = CIPAPipeline(weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).run_scoring_only(ds)
        assert default != pytest.approx(custom.difficulty_score.value, abs=0.01)

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
# Removed 1.x parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", ["knn_subsample", "n1_max_exact", "large_n_subsample"])
def test_removed_subsampling_parameters_are_rejected(param):
    with pytest.raises(TypeError):
        CIPAPipeline(**{param: 100})


def test_l1_non_convergence_is_counted(caplog):
    import logging

    ds = make_overlapping(n_minority=60, n_majority=240, seed=4)
    with caplog.at_level(logging.WARNING, logger="cipa"):
        result = CIPAPipeline(svc_max_iter=1, scaling="none").run_scoring_only(ds)
    assert result.metadata["l1_not_converged"] == 1
    assert result.difficulty_score.dimensions[6].components["converged"] is False
    assert "did not converge" in caplog.text
