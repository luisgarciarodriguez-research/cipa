"""Unit tests for CIPA result types. See SPEC-01 §4."""

import json

import pytest

from cipa.types import (
    ActionRecommendation,
    CIPAResult,
    ComplexityProfile,
    DifficultyScore,
    DimensionResult,
)


def make_dim(value: float, id_: str = "D1") -> DimensionResult:
    return DimensionResult(value=value, dimension_id=id_)


def make_seven_dims(values=(0.1,) * 7) -> tuple:
    return tuple(DimensionResult(value=v, dimension_id=f"D{i+1}") for i, v in enumerate(values))


# ---------------------------------------------------------------------------
# DimensionResult
# ---------------------------------------------------------------------------

def test_dimension_result_valid():
    dr = DimensionResult(value=0.5, dimension_id="D3")
    assert dr.value == 0.5
    assert dr.dimension_id == "D3"


def test_dimension_result_coerces_value_to_float():
    dr = DimensionResult(value=1, dimension_id="D1")
    assert isinstance(dr.value, float)


def test_dimension_result_invalid_id_raises():
    with pytest.raises(ValueError, match="dimension_id"):
        DimensionResult(value=0.5, dimension_id="D8")


def test_dimension_result_value_out_of_range_raises():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        DimensionResult(value=1.1, dimension_id="D1")


def test_dimension_result_to_dict():
    dr = DimensionResult(value=0.42, dimension_id="D2", components={"x": 0.3})
    d = dr.to_dict()
    assert d["value"] == 0.42
    assert d["dimension_id"] == "D2"
    assert d["components"] == {"x": 0.3}
    assert json.dumps(d)  # must be JSON-serializable


# ---------------------------------------------------------------------------
# DifficultyScore
# ---------------------------------------------------------------------------

def test_difficulty_score_valid():
    dims = make_seven_dims()
    ds = DifficultyScore(value=0.3, band="Moderate", weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13), dimensions=dims)
    assert ds.value == 0.3
    assert ds.band == "Moderate"


def test_difficulty_score_contributions_sum_to_ds():
    weights = (0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13)
    values = (0.2, 0.3, 0.4, 0.1, 0.5, 0.6, 0.7)
    dims = make_seven_dims(values)
    expected_ds = sum(w * v for w, v in zip(weights, values, strict=True))
    ds = DifficultyScore(value=expected_ds, band="Moderate", weights=weights, dimensions=dims)
    assert abs(sum(ds.contributions.values()) - expected_ds) < 1e-9


def test_difficulty_score_invalid_band_raises():
    dims = make_seven_dims()
    with pytest.raises(ValueError, match="band"):
        DifficultyScore(value=0.3, band="Unknown", weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13), dimensions=dims)


def test_difficulty_score_to_dict_json_serializable():
    dims = make_seven_dims()
    ds = DifficultyScore(value=0.5, band="High", weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13), dimensions=dims)
    assert json.dumps(ds.to_dict())


# ---------------------------------------------------------------------------
# ComplexityProfile
# ---------------------------------------------------------------------------

def test_complexity_profile_valid():
    p = ComplexityProfile(
        vector=(0.1,) * 7, signature="II",
        signature_name="Overlap-dominated", dominant_dimension="D2",
    )
    assert p.signature == "II"
    assert p.qualifier is None


def test_complexity_profile_invalid_signature_raises():
    with pytest.raises(ValueError, match="signature"):
        ComplexityProfile(vector=(0.1,) * 7, signature="X", signature_name="bad")


def test_complexity_profile_wrong_vector_length_raises():
    with pytest.raises(ValueError, match="length 7"):
        ComplexityProfile(vector=(0.1,) * 6, signature="I", signature_name="n")


def test_complexity_profile_v_requires_qualifier():
    with pytest.raises(ValueError, match="qualifier"):
        ComplexityProfile(vector=(0.1,) * 7, signature="V", signature_name="Compound")
    with pytest.raises(ValueError, match="qualifier"):
        ComplexityProfile(vector=(0.1,) * 7, signature="V", signature_name="Compound",
                          qualifier="mixed")


def test_complexity_profile_qualifier_only_for_v():
    with pytest.raises(ValueError, match="only to Signature V"):
        ComplexityProfile(vector=(0.1,) * 7, signature="I", signature_name="n",
                          qualifier="low")


def test_dimension_result_iqr_serialized_and_validated():
    dr = DimensionResult(value=0.4, dimension_id="D2", iqr=0.05)
    assert dr.to_dict()["iqr"] == 0.05
    with pytest.raises(ValueError, match="iqr"):
        DimensionResult(value=0.4, dimension_id="D2", iqr=-0.1)


# ---------------------------------------------------------------------------
# CIPAResult serialization
# ---------------------------------------------------------------------------

def _score_and_profile():
    dims = make_seven_dims()
    ds = DifficultyScore(value=0.5, band="High", weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13), dimensions=dims)
    profile = ComplexityProfile(vector=(0.5,) * 7, signature="V",
                                signature_name="Compound", qualifier="compound")
    return ds, profile


def test_cipa_result_to_dict_json_serializable():
    import numpy as np

    ds, profile = _score_and_profile()
    action = ActionRecommendation(
        evaluation_metrics=["AUC-PR"], preprocessing_strategy=["SMOTE"],
        model_families=["RF"], validation_protocol=["k-fold"],
        rationale={}, warnings=[],
    )
    result = CIPAResult(dataset_name="test", difficulty_score=ds, profile=profile, action=action,
                        metadata={"flag": np.bool_(True), "n": np.int64(3), "x": np.float32(0.5),
                                  "arr": np.arange(2), "t": (1, 2)})
    d = result.to_dict()
    assert json.dumps(d)
    assert d["metadata"] == {"flag": True, "n": 3, "x": 0.5, "arr": [0, 1], "t": [1, 2]}


def test_cipa_result_without_action():
    ds, profile = _score_and_profile()
    result = CIPAResult(dataset_name=None, difficulty_score=ds, profile=profile)
    d = result.to_dict()
    assert d["action"] is None and d["metadata"] == {}
    assert json.dumps(d)
