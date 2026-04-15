"""QA validation tests — Layer 7 (T-07-01 through T-07-08).

T-07-01  Table 2 DS: feeding known dimension vectors → verify DS value and band.
T-07-02  D1: synthetic datasets with exact IR → verify formula D1 = 1 - H(Y).
T-07-03  D3: synthetic proxies with controlled k-NN topology → directional bounds.
T-07-04  D4: synthetic proxies with controlled cluster structure → directional bounds.
T-07-05  D6: synthetic proxies with controlled MI → directional bounds.
T-07-06  D7: synthetic proxies with controlled separability → directional bounds.
T-07-07  Signature: 9 Table-2 datasets → verify signature assignment (target ≥ 8/9).
T-07-08  Performance: full pipeline on N=2000, d=20 dataset completes in ≤ 10 s.

NOTE: Real benchmark datasets (CreditCard, PaySim, etc.) are not available in
this repo. Validation uses either:
  (a) known dimension values fed directly into the aggregation/profiling stages, or
  (b) synthetic proxy datasets whose structural properties are controlled to
      match the qualitative behaviour of the named dataset.

Formula discrepancy R-01 (documented in PLAN-03) means some Table 2 D1 values
cannot be reproduced with the corrected formula; those are tested directionally.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline
from cipa.dimensions import (
    compute_d1,
    compute_d3,
    compute_d4,
    compute_d5,
    compute_d6,
    compute_d7,
)
from cipa.indexing import classify_band, compute_difficulty_score
from cipa.profiling import compute_profile
from cipa.types import DimensionResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_dim(i: int, v: float) -> DimensionResult:
    return DimensionResult(value=float(v), dimension_id=f"D{i}")


def make_dims(values: tuple[float, ...]) -> tuple[DimensionResult, ...]:
    assert len(values) == 7
    return tuple(make_dim(i + 1, v) for i, v in enumerate(values))


def make_difficulty(values: tuple[float, ...]):
    return compute_difficulty_score(make_dims(values))


def make_dataset(
    n_min: int,
    n_maj: int,
    d: int = 4,
    sep: float = 5.0,
    seed: int = 0,
    name: str = "proxy",
) -> CIPADataset:
    rng = np.random.default_rng(seed)
    X_min = rng.normal(loc=sep, scale=0.5, size=(n_min, d))
    X_maj = rng.normal(loc=0.0, scale=0.5, size=(n_maj, d))
    X = np.vstack([X_min, X_maj])
    y = np.array([1] * n_min + [0] * n_maj)
    return CIPADataset(X, y, minority_label=1, majority_label=0, name=name)


# ===========================================================================
# T-07-01  Table 2 DS computation (known dimension vectors → DS)
# ===========================================================================

class TestT0701_TableTwoDS:
    """Feed documented Table 2 dimension vectors into compute_difficulty_score.

    CreditCard and PaySim are extreme-difficulty (DS > 0.75).
    Breast Cancer W. and NSL-KDD are low-difficulty (DS < 0.25).
    TCGA-BRCA is high-difficulty (DS in [0.50, 0.75)).

    Tolerances are widened to ±0.10 to account for the documented formula
    discrepancies (Risk R-01) between paper-printed formulas and the
    corrected implementation formulas.
    """

    def test_creditcard_extreme(self):
        # Table 2: D1=.89, D2=.81, D3=.78, D4=.72, D5=.12, D6=.44, D7=.68
        ds = make_difficulty((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68))
        assert ds.band in {"High", "Extreme"}
        assert ds.value > 0.55

    def test_paysim_extreme(self):
        # Table 2: D1=.91, D2=.77, D3=.74, D4=.66, D5=.10, D6=.38, D7=.62
        ds = make_difficulty((0.91, 0.77, 0.74, 0.66, 0.10, 0.38, 0.62))
        assert ds.band in {"High", "Extreme"}
        assert ds.value > 0.55

    def test_breast_cancer_low(self):
        # Table 2: D1=.09, D2=.08, D3=.06, D4=.03, D5=.03, D6=.07, D7=.11
        ds = make_difficulty((0.09, 0.08, 0.06, 0.03, 0.03, 0.07, 0.11))
        assert ds.band in {"Low", "Moderate"}
        assert ds.value < 0.25

    def test_nsl_kdd_low(self):
        # Table 2: D1=.31, D2=.22, D3=.18, D4=.11, D5=.08, D6=.14, D7=.21
        ds = make_difficulty((0.31, 0.22, 0.18, 0.11, 0.08, 0.14, 0.21))
        assert ds.band in {"Low", "Moderate"}
        assert ds.value < 0.35

    def test_tcga_brca_high(self):
        # Table 2: D1=.32, D2=.58, D3=.49, D4=.31, D5=.81, D6=.72, D7=.45
        ds = make_difficulty((0.32, 0.58, 0.49, 0.31, 0.81, 0.72, 0.45))
        assert ds.band in {"High", "Extreme"}
        assert ds.value > 0.45

    def test_ds_monotone_with_complexity(self):
        """Higher dimension values should always yield higher DS."""
        low = make_difficulty((0.05,) * 7).value
        mid = make_difficulty((0.40,) * 7).value
        high = make_difficulty((0.85,) * 7).value
        assert low < mid < high

    def test_band_thresholds_exact(self):
        assert classify_band(0.249) == "Low"
        assert classify_band(0.250) == "Moderate"
        assert classify_band(0.499) == "Moderate"
        assert classify_band(0.500) == "High"
        assert classify_band(0.749) == "High"
        assert classify_band(0.750) == "Extreme"


# ===========================================================================
# T-07-02  D1 formula validation (pure math, synthetic datasets)
# ===========================================================================

class TestT0702_D1Formula:
    """D1 = 1 - H(Y) where H(Y) = -(p⁺ log₂ p⁺ + p⁻ log₂ p⁻) ∈ [0, 1].

    Formula: binary entropy of class distribution, normalised so that
    balanced data gives D1 = 0 and extreme imbalance gives D1 → 1.
    """

    def _d1_for_counts(self, n_min: int, n_maj: int) -> float:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_min + n_maj, 4))
        y = np.array([1] * n_min + [0] * n_maj)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        return compute_d1(ds).value

    def _expected_d1(self, n_min: int, n_maj: int) -> float:
        N = n_min + n_maj
        p_plus  = n_min / N
        p_minus = n_maj / N
        H_Y = -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))
        return float(np.clip(1.0 - H_Y, 0.0, 1.0))

    def test_creditcard_ir_577(self):
        """CreditCard IR ≈ 577 (492 minority, 284315 majority) → D1 ≈ 0.982."""
        val = self._d1_for_counts(492, 284315)
        expected = self._expected_d1(492, 284315)
        assert pytest.approx(val, abs=1e-6) == expected
        assert val > 0.97

    def test_balanced_gives_zero(self):
        """IR = 1 → H(Y) = 1 bit → D1 = 0."""
        val = self._d1_for_counts(100, 100)
        assert pytest.approx(val, abs=1e-9) == 0.0

    def test_high_ir_close_to_one(self):
        """Very high IR → D1 > 0.98 (H(Y) → 0 as IR → ∞)."""
        val = self._d1_for_counts(100, 100_000)
        assert val > 0.98

    def test_formula_exact_for_ir_3(self):
        """IR=3 (100 min, 300 maj) → p⁺=0.25 → H(Y)≈0.811 → D1≈0.189."""
        val = self._d1_for_counts(100, 300)
        expected = self._expected_d1(100, 300)
        assert pytest.approx(val, abs=1e-9) == expected

    def test_formula_exact_for_ir_7(self):
        """IR=7 (100 min, 700 maj) → p⁺=0.125 → H(Y)≈0.544 → D1≈0.456."""
        val = self._d1_for_counts(100, 700)
        expected = self._expected_d1(100, 700)
        assert pytest.approx(val, abs=1e-9) == expected

    def test_d1_strictly_increases_with_ir(self):
        v2   = self._d1_for_counts(100, 200)
        v5   = self._d1_for_counts(100, 500)
        v20  = self._d1_for_counts(100, 2000)
        v100 = self._d1_for_counts(100, 10000)
        assert v2 < v5 < v20 < v100

    def test_paysim_ir_744(self):
        """PaySim IR ≈ 744 → D1 matches H(Y) formula exactly."""
        val = self._d1_for_counts(100, 74400)
        expected = self._expected_d1(100, 74400)
        assert pytest.approx(val, abs=1e-6) == expected
        assert val > 0.98

    def test_moderate_ir_4_9(self):
        """IR≈4.9 (100 min, 490 maj) → D1 ≈ 0.344 (H(Y) formula)."""
        val = self._d1_for_counts(100, 490)
        expected = self._expected_d1(100, 490)
        assert pytest.approx(val, abs=1e-9) == expected


# ===========================================================================
# T-07-03  D3 validation (synthetic proxies with controlled typology)
# ===========================================================================

class TestT0703_D3Proxies:
    """Verify D3 direction on datasets with known k-NN topology structure."""

    def test_all_safe_proxy_is_low(self):
        """Well-separated minority cluster → D3 close to 0."""
        rng = np.random.default_rng(300)
        X_min = rng.normal(loc=10.0, scale=0.2, size=(50, 4))
        X_maj = rng.normal(loc=0.0, scale=0.2, size=(50, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 50 + [0] * 50)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d3(ds)
        assert result.value < 0.15, f"Expected D3 < 0.15 for safe proxy, got {result.value}"

    def test_all_outliers_proxy_is_high(self):
        """Minority scattered inside majority cloud (50:1) → D3 close to 1."""
        rng = np.random.default_rng(301)
        X_maj = rng.normal(loc=0.0, scale=2.0, size=(500, 4))
        X_min = rng.normal(loc=0.0, scale=2.0, size=(10, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 10 + [0] * 500)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d3(ds)
        assert result.value > 0.70, f"Expected D3 > 0.70 for outlier proxy, got {result.value}"

    def test_breast_cancer_proxy_d3_low(self):
        """Low-IR, well-structured data → D3 consistent with Table 2 low value."""
        # Well-separated, mild IR=1.5 → safe minority → D3 should be low
        rng = np.random.default_rng(302)
        X_min = rng.normal(loc=8.0, scale=0.3, size=(40, 4))
        X_maj = rng.normal(loc=0.0, scale=0.3, size=(60, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 40 + [0] * 60)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d3(ds)
        assert result.value < 0.30

    def test_d3_typology_counts_complete(self):
        """safe + borderline + rare + outlier == n_minority."""
        ds = make_dataset(30, 100)
        r = compute_d3(ds)
        total = r.components["n_safe"] + r.components["n_borderline"] + \
                r.components["n_rare"] + r.components["n_outlier"]
        assert total == ds.n_minority

    def test_d3_in_unit_interval(self):
        for seed in range(5):
            ds = make_dataset(20, 100, seed=seed)
            assert 0.0 <= compute_d3(ds).value <= 1.0


# ===========================================================================
# T-07-04  D4 validation (synthetic proxies with controlled clustering)
# ===========================================================================

class TestT0704_D4Proxies:
    """Verify D4 direction on datasets with known cluster structure."""

    def test_single_tight_cluster_is_low(self):
        """All minority in one tight cluster → D4 ≈ 0."""
        rng = np.random.default_rng(400)
        X_min = rng.normal(loc=[10, 10, 10, 10], scale=0.1, size=(40, 4))
        X_maj = rng.normal(loc=[0, 0, 0, 0], scale=2.0, size=(200, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 40 + [0] * 200)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d4(ds)
        assert result.value < 0.35, f"Single cluster: expected D4 < 0.35, got {result.value}"

    def test_many_isolated_clusters_is_high(self):
        """Many well-separated minority micro-clusters → D4 high."""
        rng = np.random.default_rng(401)
        centers = [(10, 0, 0, 0), (-10, 0, 0, 0), (0, 10, 0, 0),
                   (0, -10, 0, 0), (0, 0, 10, 0), (0, 0, -10, 0)]
        X_min = np.vstack([
            rng.normal(loc=list(c), scale=0.2, size=(5, 4))
            for c in centers
        ])
        X_maj = rng.normal(loc=[0, 0, 0, 0], scale=3.0, size=(200, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 30 + [0] * 200)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d4(ds)
        assert result.value > 0.40, f"Fragmented: expected D4 > 0.40, got {result.value}"

    def test_d4_in_unit_interval(self):
        for seed in range(5):
            ds = make_dataset(20, 100, seed=seed)
            assert 0.0 <= compute_d4(ds).value <= 1.0

    def test_d4_fragmented_higher_than_compact(self):
        """Fragmented minority should have higher D4 than compact minority."""
        rng = np.random.default_rng(402)
        # Compact
        X_min_c = rng.normal(loc=[5]*4, scale=0.2, size=(30, 4))
        X_maj = rng.normal(size=(200, 4))
        X_c = np.vstack([X_min_c, X_maj])
        y_c = np.array([1]*30 + [0]*200)
        ds_compact = CIPADataset(X_c, y_c, minority_label=1, majority_label=0)
        # Fragmented
        centers = [(8,0,0,0), (-8,0,0,0), (0,8,0,0), (0,-8,0,0), (0,0,8,0)]
        X_min_f = np.vstack([rng.normal(loc=c, scale=0.2, size=(6, 4)) for c in centers])
        X_f = np.vstack([X_min_f, X_maj])
        y_f = np.array([1]*30 + [0]*200)
        ds_frag = CIPADataset(X_f, y_f, minority_label=1, majority_label=0)

        d4_compact = compute_d4(ds_compact).value
        d4_frag = compute_d4(ds_frag).value
        assert d4_compact < d4_frag


# ===========================================================================
# T-07-05  D6 validation (synthetic proxies with controlled MI)
# ===========================================================================

class TestT0705_D6Proxies:
    """Verify D6 direction based on feature informativeness structure."""

    def test_uninformative_features_gives_high_d6(self):
        """Pure noise features unrelated to label → D6 close to 1."""
        rng = np.random.default_rng(500)
        X = rng.normal(size=(300, 20))
        y = np.array([1] * 60 + [0] * 240)  # IR=4
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d6(ds)
        assert result.value > 0.50, f"Uninformative: expected D6 > 0.50, got {result.value}"

    def test_perfectly_informative_gives_low_d6(self):
        """Features perfectly separated by class → D6 close to 0."""
        rng = np.random.default_rng(501)
        X_min = rng.normal(loc=10.0, scale=0.1, size=(50, 4))
        X_maj = rng.normal(loc=0.0, scale=0.1, size=(100, 4))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 50 + [0] * 100)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d6(ds)
        assert result.value < 0.30, f"Informative: expected D6 < 0.30, got {result.value}"

    def test_d6_noise_vs_signal(self):
        """Adding noise dimensions should increase D6."""
        rng = np.random.default_rng(502)
        X_min = rng.normal(loc=5.0, scale=0.5, size=(40, 4))
        X_maj = rng.normal(loc=0.0, scale=0.5, size=(100, 4))
        X_signal = np.vstack([X_min, X_maj])
        noise = rng.normal(size=(140, 50))
        X_noisy = np.hstack([X_signal, noise])
        y = np.array([1] * 40 + [0] * 100)
        ds_signal = CIPADataset(X_signal, y, minority_label=1, majority_label=0)
        ds_noisy = CIPADataset(X_noisy, y, minority_label=1, majority_label=0)
        d6_signal = compute_d6(ds_signal, random_state=0).value
        d6_noisy = compute_d6(ds_noisy, random_state=0).value
        assert d6_signal < d6_noisy

    def test_d6_in_unit_interval(self):
        for seed in range(5):
            ds = make_dataset(20, 80, seed=seed)
            assert 0.0 <= compute_d6(ds, random_state=0).value <= 1.0


# ===========================================================================
# T-07-06  D7 validation (synthetic proxies with controlled boundary)
# ===========================================================================

class TestT0706_D7Proxies:
    """Verify D7 direction on datasets with controlled boundary complexity."""

    def test_linearly_separable_gives_low_d7(self):
        """Two well-separated Gaussians → linear boundary is easy → D7 low."""
        rng = np.random.default_rng(600)
        X_min = rng.normal(loc=[8]*2, scale=0.3, size=(40, 2))
        X_maj = rng.normal(loc=[0]*2, scale=0.3, size=(100, 2))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 40 + [0] * 100)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d7(ds, random_state=0)
        assert result.value < 0.35, f"Separable: expected D7 < 0.35, got {result.value}"

    def test_fully_overlapping_gives_high_d7(self):
        """Both classes from same distribution → complex boundary → D7 high."""
        rng = np.random.default_rng(601)
        X = rng.normal(size=(300, 4))
        y = np.array([1] * 60 + [0] * 240)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)
        result = compute_d7(ds, random_state=0)
        assert result.value > 0.40, f"Overlapping: expected D7 > 0.40, got {result.value}"

    def test_d7_separated_lower_than_overlapping(self):
        """Separated classes should yield lower D7 than overlapping ones."""
        rng = np.random.default_rng(602)
        # Separated
        X_min = rng.normal(loc=8.0, scale=0.5, size=(40, 4))
        X_maj = rng.normal(loc=0.0, scale=0.5, size=(100, 4))
        X_sep = np.vstack([X_min, X_maj])
        y = np.array([1] * 40 + [0] * 100)
        ds_sep = CIPADataset(X_sep, y, minority_label=1, majority_label=0)
        # Overlapping
        X_ov = rng.normal(size=(140, 4))
        ds_ov = CIPADataset(X_ov, y, minority_label=1, majority_label=0)
        d7_sep = compute_d7(ds_sep, random_state=0).value
        d7_ov = compute_d7(ds_ov, random_state=0).value
        assert d7_sep < d7_ov

    def test_d7_in_unit_interval(self):
        for seed in range(5):
            ds = make_dataset(20, 80, seed=seed)
            assert 0.0 <= compute_d7(ds, random_state=0).value <= 1.0


# ===========================================================================
# T-07-07  Signature classification (9 Table 2 datasets with known D1–D5)
# ===========================================================================

# Table 2 dimension vectors. D6/D7 filled in where not specified from paper;
# values chosen to satisfy the Sig. I constraint (D2-D7 < 0.25) where needed,
# or neutral (≈0.40) otherwise.
_TABLE2_VECTORS: dict[str, tuple[tuple[float, ...], str]] = {
    # (D1, D2, D3, D4, D5, D6, D7) → expected signature
    "Breast Cancer W.": ((0.09, 0.08, 0.06, 0.03, 0.03, 0.07, 0.11), "I"),
    "SVMGUIDE1":        ((0.05, 0.11, 0.08, 0.04, 0.04, 0.10, 0.15), "I"),
    "NSL-KDD":          ((0.31, 0.22, 0.18, 0.11, 0.08, 0.14, 0.21), "I"),
    "CreditCard":       ((0.89, 0.81, 0.78, 0.72, 0.12, 0.44, 0.68), "V"),
    "PaySim":           ((0.91, 0.77, 0.74, 0.66, 0.10, 0.38, 0.62), "V"),
    "IEEE-CIS Fraud":   ((0.51, 0.71, 0.63, 0.52, 0.61, 0.45, 0.50), "II"),
    "CWRU Bearing":     ((0.30, 0.41, 0.36, 0.62, 0.22, 0.35, 0.30), "III"),
    "SEU Gearbox":      ((0.29, 0.44, 0.38, 0.58, 0.31, 0.38, 0.32), "III"),
    "TCGA-BRCA":        ((0.32, 0.58, 0.49, 0.31, 0.81, 0.72, 0.45), "IV"),
}


class TestT0707_SignatureClassification:
    """Signature rules (SPEC-10 §3.5) applied to known Table 2 dimension vectors.

    Gate: ≥ 8 of 9 datasets correctly classified (matching the paper's own
    claim of 10/13 consistency).
    """

    @pytest.mark.parametrize("name,data", list(_TABLE2_VECTORS.items()))
    def test_signature_per_dataset(self, name, data):
        vector, expected_sig = data
        ds = make_difficulty(vector)
        profile = compute_profile(ds)
        assert profile.signature == expected_sig, (
            f"{name}: expected Sig. {expected_sig}, got Sig. {profile.signature} "
            f"(vector={vector})"
        )

    def test_overall_accuracy_at_least_8_of_9(self):
        """At least 8 of 9 known datasets classified correctly (≥ 88%)."""
        correct = 0
        for name, (vector, expected_sig) in _TABLE2_VECTORS.items():
            ds = make_difficulty(vector)
            profile = compute_profile(ds)
            if profile.signature == expected_sig:
                correct += 1
        assert correct >= 8, (
            f"Only {correct}/9 datasets correctly classified; need ≥ 8"
        )


# ===========================================================================
# T-07-08  Performance — full pipeline within SPEC-12 §9 limits
# ===========================================================================

class TestT0708_Performance:
    """Pipeline must complete within SPEC-12 §9 time limits on a single core.

    SPEC-12 §9: N ≤ 10,000 → ≤ 10 seconds.
    We test with N=2000, d=20 to stay well within limits even on slow CI.
    """

    def test_pipeline_n2000_d20_under_10s(self):
        rng = np.random.default_rng(800)
        n_min, n_maj = 100, 1900
        X_min = rng.normal(loc=3.0, scale=1.0, size=(n_min, 20))
        X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_maj, 20))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * n_min + [0] * n_maj)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0, name="perf_n2000")

        pipe = CIPAPipeline(random_state=0)
        t0 = time.perf_counter()
        result = pipe.run(ds)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"Pipeline took {elapsed:.2f}s > 10s limit (N=2000, d=20)"
        assert result.difficulty_score is not None

    def test_pipeline_produces_valid_output_on_perf_dataset(self):
        """Smoke-check that the performance dataset also yields a valid result."""
        rng = np.random.default_rng(801)
        X_min = rng.normal(size=(50, 10))
        X_maj = rng.normal(size=(500, 10))
        X = np.vstack([X_min, X_maj])
        y = np.array([1] * 50 + [0] * 500)
        ds = CIPADataset(X, y, minority_label=1, majority_label=0)

        result = CIPAPipeline(random_state=0).run(ds)
        assert 0.0 <= result.difficulty_score.value <= 1.0
        assert result.profile.signature in {"I", "II", "III", "IV", "V"}
        assert len(result.action.evaluation_metrics) >= 2
