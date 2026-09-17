"""Freeze the v1.2.1 dimension values used by ``test_regression_v1_2_1.py``.

Run it against a checkout of the ``v1.2.1`` tag, never against the working
tree::

    git archive v1.2.1 src | tar -x -C /tmp/cipa-v1.2.1
    PYTHONPATH=/tmp/cipa-v1.2.1/src python tests/regression/generate_v1_2_1_reference.py

The script refuses to run unless the imported ``CIPAPipeline`` still accepts
the 1.x ``knn_subsample`` parameter, so a 2.x import cannot overwrite the
reference by mistake.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import CASES, SEED  # noqa: E402

import cipa  # noqa: E402
from cipa import CIPADataset, CIPAPipeline  # noqa: E402


def main() -> None:
    """Compute D1–D7 with the v1.2.1 pipeline and write the JSON reference."""
    if "knn_subsample" not in inspect.signature(CIPAPipeline).parameters:
        raise SystemExit(f"Imported cipa from {cipa.__file__} is not v1.2.1; aborting.")

    reference: dict[str, dict] = {}
    for name, build in CASES.items():
        X, y = build()
        dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name=name)
        dims = CIPAPipeline(random_state=SEED).run_dimensions_only(dataset)
        reference[name] = {
            d.dimension_id: {"value": d.value, "components": d.to_dict()["components"]}
            for d in dims
        }
        reference[name]["D7"]["svc_converged"] = dims[6].metadata["svc_converged"]

    out = HERE / "v1_2_1_reference.json"
    payload = {"cipa_source": "tag v1.2.1 (6ea039f)", "seed": SEED, "cases": reference}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
