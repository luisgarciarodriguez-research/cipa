"""L1 solver configurations for D7 (2.0.0rc3, P2).

Under rc1/rc2, L1 on the hardest subsample of the study takes over an hour and
stops at the iteration cap without converging, so the reported value is the
training error of an incomplete fit. That is a validity problem before it is a
performance one. This probe measures the alternatives.

One configuration per process, pinned to a single thread, so several can run
concurrently without contending and each timing stays comparable to the rc1
baseline, which was measured on one core.

The reference matrix is the first IR-preserving 50,000-row subsample of
``ieee_cis_fraud`` after standardization -- the draw the pipeline itself uses
for D2 and D7 -- cached as .npy so every configuration fits the same data.

    python tests/benchmarks/l1_solver_probe.py --x-path X.npy --y-path y.npy \\
        --dual auto --tol 1e-4 --max-iter 10000 --label baseline
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def probe(
    X: np.ndarray,
    y: np.ndarray,
    dual: object,
    tol: float,
    max_iter: int,
    C: float,
    random_state: int,
) -> dict:
    """Fit one LinearSVC configuration and report what it cost and reached."""
    svc = LinearSVC(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        dual=dual,
        tol=tol,
        C=C,
    )
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svc.fit(X, y)
    elapsed = time.perf_counter() - start
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    error_rate = 1.0 - float(accuracy_score(y, svc.predict(X)))
    n_iter = np.asarray(svc.n_iter_).ravel()
    return {
        "dual": dual,
        "tol": tol,
        "max_iter": max_iter,
        "C": C,
        "converged": converged,
        "n_iter": int(n_iter[0]) if n_iter.size else None,
        "L1": float(np.clip(error_rate, 0.0, 1.0)),
        "seconds": round(elapsed, 1),
        "seconds_per_iter": round(elapsed / max(int(n_iter[0]), 1), 4) if n_iter.size else None,
    }


def main() -> None:
    """Run one configuration on the cached matrices and print it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--dual", default="auto", help="auto, true or false")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    dual: object = {"auto": "auto", "true": True, "false": False}[args.dual.lower()]
    X = np.load(args.x_path)
    y = np.load(args.y_path)
    result = probe(X, y, dual, args.tol, args.max_iter, args.C, args.random_state)
    result["label"] = args.label
    print(json.dumps(result))


if __name__ == "__main__":
    main()
