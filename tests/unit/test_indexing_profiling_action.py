"""Unit tests for indexing (SPEC-09), profiling (SPEC-10), and action (SPEC-11)."""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset
from cipa._constants import DEFAULT_WEIGHTS, SIGNATURE_NAMES
from cipa.action import compute_action
from cipa.indexing import classify_band, compute_difficulty_score
from cipa.profiling import compute_profile
from cipa.types import DifficultyScore, DimensionResult

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

# (D1, D2, D3, D4, D5, D6, D7) → (signature, dominant_dimension, qualifier)
_SIGNATURE_CASES = {
    # Dominance of each candidate
    "D1 dominates":                 ((0.80, 0.30, 0.20, 0.30, 0.10, 0.20, 0.20), ("I", "D1", None)),
    "D2 dominates":                 ((0.20, 0.70, 0.40, 0.30, 0.10, 0.60, 0.30), ("II", "D2", None)),
    "D4 dominates":                 ((0.20, 0.30, 0.10, 0.65, 0.40, 0.20, 0.20), ("III", "D4", None)),
    "D5 dominates":                 ((0.30, 0.30, 0.10, 0.20, 0.90, 0.95, 0.20), ("IV", "D5", None)),
    "D1 max beats elevated D5":     ((0.70, 0.30, 0.20, 0.30, 0.60, 0.20, 0.20), ("I", "D1", None)),
    "max only over D1 D2 D4 D5":    ((0.60, 0.30, 0.90, 0.30, 0.10, 0.20, 0.95), ("I", "D1", None)),
    "just above tau":               ((0.10, 0.20, 0.10, 0.5000001, 0.10, 0.10, 0.10), ("III", "D4", None)),
    # Ties at the maximum: D1 > D2 > D4 > D5
    "tie D1 D2":                    ((0.60, 0.60, 0.10, 0.20, 0.20, 0.20, 0.20), ("I", "D1", None)),
    "tie D2 D4":                    ((0.20, 0.70, 0.10, 0.70, 0.20, 0.20, 0.20), ("II", "D2", None)),
    "tie D4 D5":                    ((0.20, 0.30, 0.10, 0.55, 0.55, 0.20, 0.20), ("III", "D4", None)),
    "tie all four":                 ((0.60, 0.60, 0.10, 0.60, 0.60, 0.20, 0.20), ("I", "D1", None)),
    # Exactly tau does not dominate
    "tau exact single":             ((0.20, 0.50, 0.10, 0.30, 0.20, 0.20, 0.20), ("V", None, "single")),
    "tau exact compound":           ((0.50, 0.20, 0.10, 0.50, 0.20, 0.20, 0.20), ("V", None, "compound")),
    # D3, D6, D7 above tau never dominate
    "D3 D6 D7 high":                ((0.30, 0.35, 0.90, 0.20, 0.10, 0.80, 0.70), ("V", None, "compound")),
    "only D6 high":                 ((0.10, 0.20, 0.10, 0.30, 0.20, 0.95, 0.10), ("V", None, "single")),
    # Qualifiers of V
    "compound below tau":           ((0.40, 0.45, 0.10, 0.20, 0.20, 0.30, 0.20), ("V", None, "compound")),
    "low with tau prime exact":     ((0.35, 0.35, 0.10, 0.20, 0.35, 0.30, 0.20), ("V", None, "low")),
    "all zero":                     ((0.0,) * 7, ("V", None, "low")),
}


class TestComputeProfile:
    def _profile(self, values: tuple[float, ...], **kwargs):
        ds = make_difficulty(values)
        return compute_profile(ds, **kwargs)

    def test_returns_complexity_profile(self):
        p = self._profile((0.5,) * 7)
        assert p.signature in {"I", "II", "III", "IV", "V"}
        assert len(p.vector) == 7

    @pytest.mark.parametrize("name", list(_SIGNATURE_CASES))
    def test_signature_table(self, name):
        values, (sig, dominant, qualifier) = _SIGNATURE_CASES[name]
        p = self._profile(values)
        assert (p.signature, p.dominant_dimension, p.qualifier) == (sig, dominant, qualifier)
        assert p.signature_name == SIGNATURE_NAMES[sig]

    def test_active_and_elevated_dimensions_sorted_descending(self):
        p = self._profile((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68))
        assert p.active_dimensions == ["D1", "D2", "D3", "D4", "D7"]
        assert p.elevated_dimensions == ["D1", "D2", "D3", "D4", "D7", "D6"]

    def test_thresholds_are_configurable(self):
        values = (0.45, 0.30, 0.10, 0.20, 0.20, 0.20, 0.20)
        assert self._profile(values).signature == "V"
        p = self._profile(values, tau=0.40, tau_prime=0.25)
        assert (p.signature, p.tau, p.tau_prime) == ("I", 0.40, 0.25)
        low = self._profile((0.30, 0.30, 0.1, 0.1, 0.1, 0.1, 0.1), tau_prime=0.30)
        assert low.qualifier == "low"

    def test_vector_matches_input(self):
        values = (0.3, 0.5, 0.4, 0.2, 0.6, 0.3, 0.4)
        p = self._profile(values)
        assert p.vector == pytest.approx(values, abs=1e-9)

    def test_to_dict_contains_qualifier(self):
        d = self._profile((0.1,) * 7).to_dict()
        assert d["signature"] == "V" and d["qualifier"] == "low"
        assert d["tau"] == 0.50 and d["tau_prime"] == 0.35


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
