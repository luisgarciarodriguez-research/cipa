"""D5 reference values from the CIPA Extended pilot (cipa 2.0.0rc2 handoff §3).

The pilot computed r_95 and D5 = ρ/(1+ρ) with an external script over
``v2.0.0rc1`` on imbdata 0.3.1, with ``scaling="standard"``, seed 42 and all
N rows. The pipeline computes D5 on the preprocessed matrix without
subsampling, so ``preprocess_dataset`` + ``compute_d5`` is the same path.

Runs only where imbdata is installed and the datasets are already cached;
it never downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipa import CIPADataset
from cipa.dimensions import compute_d5
from cipa.preprocessing import preprocess_dataset

imbdata = pytest.importorskip("imbdata")

# key: (N, d used, |C+|, r_95, D5)
PILOT_REFERENCE: dict[str, tuple[int, int, int, int, float]] = {
    "tcga_brca":         (826, 5_000, 147, 536, 0.785),
    "secom":             (1_567, 446, 104, 162, 0.609),
    "ozone_level":       (2_536, 72, 73, 20, 0.215),
    "pima_diabetes":     (768, 8, 268, 8, 0.029),
    "abalone_19":        (4_177, 8, 32, 3, 0.086),
    "credit_card_fraud": (284_807, 30, 492, 27, 0.052),
}


def _cached(key: str) -> bool:
    try:
        return bool(imbdata.info(key).get("is_cached"))
    except Exception:
        return False


@pytest.mark.parametrize("key", list(PILOT_REFERENCE))
def test_d5_reproduces_pilot_reference(key):
    if not _cached(key):
        pytest.skip(f"imbdata dataset {key!r} is not cached")
    N, d, n_minority, r_95, expected = PILOT_REFERENCE[key]
    X, y = imbdata.load(key)
    dataset = CIPADataset(
        X.to_numpy(dtype=np.float64), np.asarray(y).astype(int),
        minority_label=1, majority_label=0, name=key,
    )
    prepared, _ = preprocess_dataset(dataset, "standard")
    result = compute_d5(prepared, random_state=42)

    assert (prepared.N, prepared.d) == (N, d)
    assert result.components["n_minority"] == n_minority
    assert result.components["r_95"] == r_95
    assert result.value == pytest.approx(expected, abs=1e-3)
