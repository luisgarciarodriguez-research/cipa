"""Regression against v1.2.1 (cipa 2.0.0 handoff §1 and §5).

On duplicate-free data without constant columns, with ``scaling="none"``,
no subsampling and the same seed, D1–D4 and D6 must equal the values frozen
from the ``v1.2.1`` tag in ``v1_2_1_reference.json``. D7 must match too
wherever L1 converged in v1.2.1 (C5 changed only the non-convergent case).
Values are compared exactly: those formulas did not change.

D5 was redefined in 2.0.0rc2 (r_95 relative to the minority size), so its
value is not compared. The 1.x D5, the normalised spectral entropy, is still
reported as the component ``spectral_entropy_norm`` and must equal the
v1.2.1 D5 exactly, together with ``H_nats``, ``H_max_nats`` and
``n_components_fit``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cipa import CIPADataset, CIPAPipeline

from .cases import CASES, SEED

REFERENCE = json.loads((Path(__file__).parent / "v1_2_1_reference.json").read_text())["cases"]


@pytest.fixture(scope="module")
def computed() -> dict[str, dict]:
    """D1–D7 of every case under the 2.0.0 pipeline in regression mode."""
    out = {}
    for name, build in CASES.items():
        X, y = build()
        dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name=name)
        pipeline = CIPAPipeline(random_state=SEED, scaling="none", n_max=len(y))
        out[name] = {d.dimension_id: d for d in pipeline.run_dimensions_only(dataset)}
    return out


def test_reference_covers_at_least_three_cases():
    assert len(REFERENCE) >= 3
    assert set(REFERENCE) == set(CASES)


@pytest.mark.parametrize("name", list(CASES))
def test_cases_have_no_duplicates_or_constant_columns(name):
    X, _ = CASES[name]()
    assert len(np.unique(X, axis=0)) == len(X)
    assert np.all(np.ptp(X, axis=0) > 0)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("dim", ["D1", "D2", "D3", "D4", "D6"])
def test_d1_to_d4_and_d6_match_v1_2_1(computed, name, dim):
    assert computed[name][dim].value == REFERENCE[name][dim]["value"]


@pytest.mark.parametrize("name", list(CASES))
def test_d5_spectral_entropy_component_matches_v1_2_1_d5(computed, name):
    d5 = computed[name]["D5"]
    assert d5.components["spectral_entropy_norm"] == REFERENCE[name]["D5"]["value"]
    for key, value in REFERENCE[name]["D5"]["components"].items():
        assert d5.components[key] == value, (name, key)


@pytest.mark.parametrize("name", list(CASES))
def test_d7_matches_v1_2_1_where_l1_converged(computed, name):
    if not REFERENCE[name]["D7"]["svc_converged"]:
        pytest.skip("L1 did not converge in v1.2.1; C5 changes this case by design")
    assert computed[name]["D7"].value == REFERENCE[name]["D7"]["value"]


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("dim", ["D2", "D3", "D4"])
def test_components_match_v1_2_1(computed, name, dim):
    expected = REFERENCE[name][dim]["components"]
    for key, value in expected.items():
        assert computed[name][dim].components[key] == value, (name, dim, key)


def test_regression_mode_does_not_subsample(computed):
    for dims in computed.values():
        for dim in dims.values():
            assert dim.metadata["subsampled"] is False
            assert dim.iqr == 0.0
