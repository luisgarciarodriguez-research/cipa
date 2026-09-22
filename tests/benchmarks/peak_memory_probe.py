"""Peak memory of a brute-force neighbour search (2.0.0rc3, P1.4).

Answers one question: does ``chunk_size`` bound the memory of the brute-force
search that ``select_knn_algorithm`` now picks above 15 features? A tree walks
nodes and holds little; brute force materialises a distance block, so the
block size is what has to be governed.

Run one configuration per process so the peak is attributable:

    python tests/benchmarks/peak_memory_probe.py --n-fit 50000 --d 432 \\
        --n-query 50000 --chunk-size 65536

Shapes are what determine memory, not values, so the matrices are synthetic.
Peak is the process high-water mark (``VmHWM``), which counts the allocations
scikit-learn makes internally and Python's own allocator cannot see.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def high_water_mark_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmHWM not reported by this kernel")


def probe(n_fit: int, d: int, n_query: int, chunk_size: int, k: int, seed: int) -> dict:
    """Fit a brute-force index and query it, reporting the memory it added."""
    from cipa._knn import kneighbors_excluding_self

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_fit, d))
    query_idx = rng.choice(n_fit, size=n_query, replace=False)
    X_query = X[query_idx]

    baseline = high_water_mark_mb()
    nn = NearestNeighbors(algorithm="brute", n_jobs=-1).fit(X)

    start = time.perf_counter()
    dist, _ = kneighbors_excluding_self(nn, X_query, query_idx, k, chunk_size)
    elapsed = time.perf_counter() - start
    peak = high_water_mark_mb()

    return {
        "n_fit": n_fit,
        "d": d,
        "n_query": n_query,
        "chunk_size": chunk_size,
        "k": k,
        "data_mb": round(X.nbytes / 2**20, 1),
        "baseline_peak_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "added_mb": round(peak - baseline, 1),
        "seconds": round(elapsed, 2),
        "checksum": float(dist[:, 0].sum()),
    }


def main() -> None:
    """Parse one configuration, probe it and print the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n-query", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(probe(
        args.n_fit, args.d, args.n_query, args.chunk_size, args.k, args.seed
    )))


if __name__ == "__main__":
    main()
