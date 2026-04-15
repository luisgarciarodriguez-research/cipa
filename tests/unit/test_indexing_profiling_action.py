"""Unit tests for indexing (SPEC-09), profiling (SPEC-10), and action (SPEC-11)."""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset
from cipa._constants import DEFAULT_WEIGHTS
from cipa.action import compute_action
from cipa.indexing import classify_band, compute_difficulty_score
from cipa.profiling import compute_profile
from cipa.types import DimensionResult, DifficultyScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dim(i: int, value: float) -> DimensionResult:
    """Shorthand: DimensionResult for D{i} with the given value."""
    return DimensionResult(value=float(value), dimension_id=f"D{i}")


def make_dims(values: tuple[float, ...]) -> tuple[DimensionResult, ...]:
    """Build an ordered (D1..D7) tuple from a 7-tuple of floats."""
    assert len(values) == 7
    return tuple(make_dim(i + 1, v) for i, v in enumerate(values))


def make_difficulty(values: tuple[float, ...], weights=DEFAULT_WEIGHTS) -> DifficultyScore:
    return compute_difficulty_score(make_dims(values), weights)


def make_dataset(ir: float = 10.0, n: int = 110) -> CIPADataset:
    n_min = max(2, round(n / (ir + 1)))
    n_maj = n - n_min
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = np.array([1] * n_min + [0] * n_maj)
    return CIPADataset(X, y, minority_label=1, majority_label=0)


# ===========================================================================
# INDEXING — compute_difficulty_score & classify_band
# ===========================================================================

class TestClassifyBand:
    def test_low(self):
        assert classify_band(0.0) == "Low"
        assert classify_band(0.24) == "Low"

    def test_moderate(self):
        assert classify_band(0.25) == "Moderate"
        assert classify_band(0.49) == "Moderate"

    def test_high(self):
        assert classify_band(0.50) == "High"
        assert classify_band(0.74) == "High"

    def test_extreme(self):
        assert classify_band(0.75) == "Extreme"
        assert classify_band(1.00) == "Extreme"


class TestComputeDifficultyScore:
    def test_returns_difficulty_score(self):
        ds = make_difficulty((0.5,) * 7)
        assert 0.0 <= ds.value <= 1.0
        assert ds.band in {"Low", "Moderate", "High", "Extreme"}

    def test_zero_dims_gives_zero(self):
        ds = make_difficulty((0.0,) * 7)
        assert ds.value == pytest.approx(0.0)
        assert ds.band == "Low"

    def test_one_dims_gives_one(self):
        ds = make_difficulty((1.0,) * 7)
        assert ds.value == pytest.approx(1.0)
        assert ds.band == "Extreme"

    def test_weighted_sum_correct(self):
        # Only D1 = 1.0, rest = 0 → DS = w1 = 0.10
        ds = make_difficulty((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert ds.value == pytest.approx(0.10, abs=1e-9)

    def test_creditcard_dims_give_high_ds(self):
        """Table 2 CreditCard dims should yield a High or Extreme DS."""
        ds = make_difficulty((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68))
        assert ds.value > 0.50  # at least High difficulty
        assert ds.band in {"High", "Extreme"}

    def test_breast_cancer_target(self):
        """Table 2: Breast Cancer W. DS ≈ 0.08 ±0.05."""
        ds = make_difficulty((0.09, 0.08, 0.06, 0.03, 0.03, 0.05, 0.04))
        assert ds.value < 0.15

    def test_dimensions_preserved(self):
        dims = make_dims((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        ds = compute_difficulty_score(dims)
        assert ds.dimensions == dims

    def test_weights_preserved(self):
        custom = (1/7,) * 7
        ds = make_difficulty((0.5,) * 7, weights=custom)
        assert ds.weights == custom

    def test_contributions_sum_to_value(self):
        ds = make_difficulty((0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8))
        assert sum(ds.contributions.values()) == pytest.approx(ds.value, abs=1e-9)

    # --- Validation errors ---

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="7 elements"):
            compute_difficulty_score(make_dims((0.5,) * 7)[:5])

    def test_wrong_order_raises(self):
        dims = list(make_dims((0.5,) * 7))
        dims[0], dims[1] = dims[1], dims[0]  # swap D1↔D2
        with pytest.raises(ValueError, match="D1"):
            compute_difficulty_score(tuple(dims))

    def test_weights_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length 7"):
            compute_difficulty_score(make_dims((0.5,) * 7), weights=(0.5,) * 5)

    def test_weights_dont_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            compute_difficulty_score(make_dims((0.5,) * 7), weights=(0.2,) * 7)

    def test_negative_weight_raises(self):
        w = list(DEFAULT_WEIGHTS)
        w[0] = -0.1
        w[1] += 0.2  # compensate: w[0] dropped by 0.20, so add 0.20 to keep sum = 1
        with pytest.raises(ValueError, match=">= 0"):
            compute_difficulty_score(make_dims((0.5,) * 7), weights=tuple(w))


# ===========================================================================
# PROFILING — compute_profile
# ===========================================================================

class TestComputeProfile:
    def _profile(self, values: tuple[float, ...]):
        ds = make_difficulty(values)
        return compute_profile(ds)

    def test_returns_complexity_profile(self):
        p = self._profile((0.5,) * 7)
        assert p.signature in {"I", "II", "III", "IV", "V"}
        assert len(p.vector) == 7

    def test_signature_i_all_low(self):
        """All D2-D7 < 0.25 → Signature I."""
        p = self._profile((0.09, 0.08, 0.06, 0.03, 0.03, 0.05, 0.04))
        assert p.signature == "I"
        assert p.signature_name == "Imbalance-dominated"

    def test_signature_iv_d5_dominates(self):
        """D5 is max(D1-D5) and > 0.55 and > D2 + 0.10 → Sig. IV."""
        p = self._profile((0.32, 0.58, 0.49, 0.31, 0.81, 0.40, 0.35))
        assert p.signature == "IV"
        assert p.signature_name == "Dimensionality-dominated"

    def test_signature_iii_d4_dominates(self):
        """D4 is max(D1-D5) and > 0.50 → Sig. III."""
        p = self._profile((0.30, 0.41, 0.36, 0.62, 0.22, 0.30, 0.25))
        assert p.signature == "III"
        assert p.signature_name == "Fragmented"

    def test_signature_ii_d2_elevated(self):
        """D2 > 0.55, D2 >= D1 → Sig. II (TCGA-style with D2 high, no D5 spike)."""
        p = self._profile((0.51, 0.71, 0.63, 0.52, 0.40, 0.50, 0.45))
        assert p.signature == "II"
        assert p.signature_name == "Overlap-dominated"

    def test_signature_v_compound(self):
        """Multiple dimensions elevated, no clear single dominant → Sig. V."""
        p = self._profile((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68))
        assert p.signature == "V"
        assert p.signature_name == "Compound"

    def test_dominant_dimensions_correct(self):
        """Dims ≥ 0.55 should appear in dominant_dimensions, sorted descending."""
        p = self._profile((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68))
        assert "D1" in p.dominant_dimensions
        assert "D2" in p.dominant_dimensions
        assert "D5" not in p.dominant_dimensions  # 0.12 < 0.55
        # sorted descending by value
        vals = [p.vector[int(d[1]) - 1] for d in p.dominant_dimensions]
        assert vals == sorted(vals, reverse=True)

    def test_no_dominant_dims_when_all_low(self):
        p = self._profile((0.09, 0.08, 0.06, 0.03, 0.03, 0.05, 0.04))
        assert p.dominant_dimensions == []

    def test_vector_matches_input(self):
        values = (0.3, 0.5, 0.4, 0.2, 0.6, 0.3, 0.4)
        p = self._profile(values)
        assert p.vector == pytest.approx(values, abs=1e-9)


# ===========================================================================
# ACTION — compute_action
# ===========================================================================

class TestComputeAction:
    def _action(self, values: tuple[float, ...], ir: float = 10.0, n: int = 200):
        ds = make_difficulty(values)
        profile = compute_profile(ds)
        dataset = make_dataset(ir=ir, n=n)
        return compute_action(profile, ds, dataset)

    def test_returns_action_recommendation(self):
        rec = self._action((0.5,) * 7)
        assert isinstance(rec.evaluation_metrics, list)
        assert isinstance(rec.preprocessing_strategy, list)
        assert isinstance(rec.model_families, list)
        assert isinstance(rec.validation_protocol, list)
        assert isinstance(rec.rationale, dict)
        assert isinstance(rec.warnings, list)

    def test_always_includes_auc_pr_and_f1(self):
        rec = self._action((0.1,) * 7, ir=1.0)
        assert "AUC-PR" in rec.evaluation_metrics
        assert "F1-score (minority class)" in rec.evaluation_metrics

    def test_always_includes_stratified_cv(self):
        rec = self._action((0.1,) * 7)
        protocols = " ".join(rec.validation_protocol)
        assert "Stratified" in protocols

    def test_gmean_added_for_high_ds(self):
        rec = self._action((0.7,) * 7)
        assert "G-mean" in rec.evaluation_metrics

    def test_mcc_added_for_extreme_ds(self):
        rec = self._action((1.0,) * 7)
        assert "MCC" in rec.evaluation_metrics

    def test_auc_roc_added_when_d2_high(self):
        # D2 = 0.70
        rec = self._action((0.30, 0.70, 0.30, 0.20, 0.20, 0.20, 0.20))
        assert "AUC-ROC" in rec.evaluation_metrics

    def test_no_resampling_for_low_ds(self):
        rec = self._action((0.0,) * 7, ir=1.0)
        assert any("No resampling" in s for s in rec.preprocessing_strategy)

    def test_smote_for_signature_i(self):
        # D1 high enough + D2-D7 < 0.25 → Sig I; DS > 0.25 → resampling applies
        rec = self._action((1.0, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24), ir=2.0)
        assert rec.preprocessing_strategy  # non-empty
        strategies = " ".join(rec.preprocessing_strategy)
        assert "SMOTE" in strategies

    def test_borderline_smote_for_signature_ii(self):
        rec = self._action((0.51, 0.71, 0.63, 0.52, 0.40, 0.50, 0.45))
        strategies = " ".join(rec.preprocessing_strategy)
        assert "Borderline-SMOTE" in strategies or "ADASYN" in strategies

    def test_model_families_nonempty(self):
        rec = self._action((0.5,) * 7)
        assert len(rec.model_families) >= 1

    def test_cost_sensitive_added_for_extreme_ds(self):
        rec = self._action((1.0,) * 7)
        models = " ".join(rec.model_families)
        assert "Cost-sensitive" in models

    def test_warning_for_high_ir(self):
        rec = self._action((0.3,) * 7, ir=10.0)
        joined = " ".join(rec.warnings)
        assert "Accuracy" in joined

    def test_warning_for_d5_high(self):
        # D5 = 0.80 (severe dimensionality)
        rec = self._action((0.32, 0.40, 0.35, 0.28, 0.80, 0.30, 0.25))
        joined = " ".join(rec.warnings)
        assert "dimensionality" in joined.lower()

    def test_warning_for_d3_high(self):
        rec = self._action((0.50, 0.60, 0.75, 0.40, 0.30, 0.40, 0.45))
        joined = " ".join(rec.warnings)
        assert "rare or outliers" in joined.lower()

    def test_rationale_covers_all_recommendations(self):
        rec = self._action((0.6,) * 7)
        all_recs = (
            rec.evaluation_metrics
            + rec.preprocessing_strategy
            + rec.model_families
            + rec.validation_protocol
        )
        for item in all_recs:
            assert item in rec.rationale, f"Missing rationale for: {item!r}"

    def test_no_duplicate_metrics(self):
        rec = self._action((0.8,) * 7)
        assert len(rec.evaluation_metrics) == len(set(rec.evaluation_metrics))

    def test_no_duplicate_preprocessing(self):
        rec = self._action((0.8,) * 7)
        assert len(rec.preprocessing_strategy) == len(set(rec.preprocessing_strategy))

    def test_large_dataset_scalability_model(self):
        rec = self._action((0.5,) * 7, n=150_000, ir=5.0)
        models = " ".join(rec.model_families)
        assert "linear or tree-based" in models.lower()

    def test_small_dataset_leave_one_out(self):
        rec = self._action((0.6,) * 7, n=100, ir=5.0)
        protocols = " ".join(rec.validation_protocol)
        assert "Leave-one-out" in protocols or "bootstrap" in protocols.lower()

    def test_high_ir_k10_validation(self):
        rec = self._action((0.5,) * 7, ir=25.0)
        protocols = " ".join(rec.validation_protocol)
        assert "k=10" in protocols
