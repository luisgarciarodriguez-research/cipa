"""CIPAPipeline: orchestrates the full CIPA framework (C → I → P → A).

See §3 of García Rodríguez et al. (2026) for the data flow.

Computation protocol (cipa 2.0.0)
---------------------------------
1. Preprocessing (C2), once on all N rows: constant columns are dropped and
   the rest scaled (``scaling``). Every dimension sees the same matrix.
2. Each dimension runs on the rows that match what it measures (C6):

   ============================  ==============================================
   Dimension                     Rows
   ============================  ==============================================
   D1, D5, D6                    all N rows
   D3                            every minority row as query, neighbours
                                 searched among all N rows
   D4                            all minority rows; if |C+| > n_max,
                                 n_subsamples draws of n_max minority rows
   D2 (F3, N1, kDN), D7 (L1, N2)  all N rows if N <= n_max; otherwise
                                 n_subsamples stratified draws of n_max rows
                                 that keep the imbalance ratio (±1 row)
   ============================  ==============================================

   With several draws the dimension value and each numeric component are the
   median over draws, with their IQR recorded. D2 and D7 share the same draws.
   DS is computed from the medians.
3. Every source of randomness derives from ``random_state`` (C9): draw seeds
   come from ``numpy.random.SeedSequence``; the integer itself is passed to
   ``mutual_info_classif``, ``LinearSVC`` and PCA.

This module is part of the CIPA software package, companion implementation to:

    García Rodríguez, L., Neme Castillo, J. A., Gómez Adorno, H. M., & Fuentes Pineda, G. (2026).
    CIPA: A Multi-Domain Statistical Framework for Characterizing Imbalanced
    Datasets and Computing a Difficulty Score.
    COMIA 2026 — XVIII Congreso Mexicano de Inteligencia Artificial.
    DOI: TODO (pending publication)

Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Universidad Nacional Autónoma de México (UNAM)

Development supported by SECIHTI (researcher ID (CVU) 905206, Luis García Rodríguez).

License: MIT — see LICENSE file for full terms.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np

from cipa._constants import (
    DEFAULT_D2_WEIGHTS,
    DEFAULT_DBSCAN_MIN_SAMPLES,
    DEFAULT_K,
    DEFAULT_KNN_ALGORITHM,
    DEFAULT_N_MAX,
    DEFAULT_N_SUBSAMPLES,
    DEFAULT_QUERY_CHUNK_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALING,
    DEFAULT_SVC_MAX_ITER,
    DEFAULT_SVC_TOL,
    DEFAULT_WEIGHTS,
    KNN_ALGORITHM_OPTIONS,
    SCALING_OPTIONS,
    SIGNATURE_TAU,
    SIGNATURE_TAU_PRIME,
)
from cipa._subsampling import (
    STREAM_D2_D7,
    STREAM_D4,
    derive_seed,
    median_iqr,
    minority_indices,
    stratified_indices,
    validate_random_state,
)
from cipa.dataset import CIPADataset
from cipa.types import CIPAResult, DimensionResult

logger = logging.getLogger(__name__)


class CIPAPipeline:
    """Orchestrates the four CIPA stages: C → I → P → A.

    Parameters
    ----------
    weights : tuple of 7 floats
        Dimension weights for DS computation. Must sum to 1.
    random_state : int
        Seed for every random source (C9). Must be a non-negative integer;
        None is rejected. Default 42.
    scaling : {"standard", "robust", "none"}
        Feature scaling applied once to all N rows after dropping constant
        columns (C2). ``"robust"`` is meant for sensitivity analysis and
        ``"none"`` for comparisons with cipa 1.x.
    n_max : int
        Largest number of rows (D2, D7) or minority rows (D4) a single
        computation uses; above it the dimension is computed on
        ``n_subsamples`` draws of ``n_max`` rows (C6).
    n_subsamples : int
        Number of draws when a dimension exceeds ``n_max``.
    n_jobs : int or None
        Parallel jobs for neighbour searches, DBSCAN and, with
        scikit-learn >= 1.5, mutual information. It does not change results.
    k_neighbors : int
        k for kDN (D2) and NS-typology (D3).
    dbscan_min_samples : int
        DBSCAN min_samples for D4.
    dbscan_eps : float or None
        DBSCAN eps for D4. None = adaptive.
    d2_weights : tuple of 3 floats
        (alpha, beta, gamma) for (F3, N1, kDN) in D2. Must sum to 1.
    svc_max_iter : int
        Maximum LinearSVC iterations for L1 (C5).
    svc_tol : float
        Stopping tolerance for LinearSVC (2.0.0rc3). Exposed alongside the cap
        because on the hardest subsamples the fit stops at the cap, so the
        tolerance is part of what the reported L1 means.
    tau : float
        Dominance threshold of the signature rule (C7).
    tau_prime : float
        Threshold for the Signature V qualifier (C7).
    chunk_size : int
        Maximum number of neighbour queries sent to a tree at once. It also
        bounds the memory of a brute-force search, which materialises a
        distance block per chunk instead of walking a tree.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}
        Neighbour search algorithm for D2 (kDN, N1), D3 and D7 (N2). The
        default ``"auto"`` resolves per matrix through
        ``cipa._knn.select_knn_algorithm``: kd_tree up to 15 features, brute
        above. ``"ball_tree"`` reproduces the behaviour before 2.0.0rc3. The
        DBSCAN neighbourhood of D4 and the Boruvka fallback of N1 are not
        covered; see the CHANGELOG entry for 2.0.0rc3.

    Raises
    ------
    ValueError
        If random_state, scaling, n_max or n_subsamples is invalid.
    """

    def __init__(
        self,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        scaling: str = DEFAULT_SCALING,
        n_max: int = DEFAULT_N_MAX,
        n_subsamples: int = DEFAULT_N_SUBSAMPLES,
        n_jobs: int | None = None,
        k_neighbors: int = DEFAULT_K,
        dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
        dbscan_eps: float | None = None,
        d2_weights: tuple[float, float, float] = DEFAULT_D2_WEIGHTS,
        svc_max_iter: int = DEFAULT_SVC_MAX_ITER,
        svc_tol: float = DEFAULT_SVC_TOL,
        tau: float = SIGNATURE_TAU,
        tau_prime: float = SIGNATURE_TAU_PRIME,
        chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
        algorithm: str = DEFAULT_KNN_ALGORITHM,
    ) -> None:
        """Validate and store configuration; nothing is computed until a run method is called."""
        if scaling not in SCALING_OPTIONS:
            raise ValueError(f"scaling must be one of {SCALING_OPTIONS}, got {scaling!r}")
        if algorithm not in KNN_ALGORITHM_OPTIONS:
            raise ValueError(
                f"algorithm must be one of {KNN_ALGORITHM_OPTIONS}, got {algorithm!r}"
            )
        if n_max < 10:
            raise ValueError(f"n_max must be >= 10, got {n_max}")
        if n_subsamples < 1:
            raise ValueError(f"n_subsamples must be >= 1, got {n_subsamples}")
        self._weights = tuple(weights)
        self._random_state = validate_random_state(random_state)
        self._scaling = scaling
        self._n_max = int(n_max)
        self._n_subsamples = int(n_subsamples)
        self._n_jobs = n_jobs
        self._k = k_neighbors
        self._dbscan_min_samples = dbscan_min_samples
        self._dbscan_eps = dbscan_eps
        self._d2_weights = d2_weights
        self._svc_max_iter = svc_max_iter
        self._svc_tol = svc_tol
        self._tau = tau
        self._tau_prime = tau_prime
        self._chunk_size = chunk_size
        self._algorithm = algorithm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, dataset: CIPADataset) -> CIPAResult:
        """Execute the full CIPA pipeline: C → I → P → A.

        Parameters
        ----------
        dataset : CIPADataset

        Returns
        -------
        CIPAResult
            With ``action`` populated and the run record in ``metadata``.

        Raises
        ------
        ValueError
            If dataset fails validation (propagated from CIPADataset).
        """
        from cipa.action import compute_action

        result = self.run_scoring_only(dataset)
        result.action = compute_action(result.profile, result.difficulty_score, dataset)
        return result

    def run_dimensions_only(
        self, dataset: CIPADataset
    ) -> tuple[DimensionResult, ...]:
        """Run only Stage C (Characterization). Returns the seven dimension scores.

        Useful for inspection, custom weighting, or partial pipeline execution.

        Parameters
        ----------
        dataset : CIPADataset

        Returns
        -------
        tuple of 7 DimensionResult
            Ordered (D1, D2, D3, D4, D5, D6, D7).
        """
        dims, _ = self._characterize(dataset)
        return dims

    def run_scoring_only(self, dataset: CIPADataset) -> CIPAResult:
        """Run Stages C, I and P. Returns everything except the Action protocol.

        Parameters
        ----------
        dataset : CIPADataset

        Returns
        -------
        CIPAResult
            ``difficulty_score`` (DS, band, weights and the seven dimensions
            with IQR, components and protocol metadata), ``profile``
            (signature and qualifier), ``metadata`` (run record) and
            ``action=None``. ``to_dict()`` is JSON-serializable.
        """
        from cipa.indexing import compute_difficulty_score
        from cipa.profiling import compute_profile

        logger.info("CIPAPipeline: dataset=%s (N=%d, d=%d, IR=%.2f)",
                    dataset.name, dataset.N, dataset.d, dataset.IR)
        dims, metadata = self._characterize(dataset)
        ds = compute_difficulty_score(dims, self._weights)
        profile = compute_profile(ds, tau=self._tau, tau_prime=self._tau_prime)
        return CIPAResult(
            dataset_name=dataset.name,
            difficulty_score=ds,
            profile=profile,
            action=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _characterize(
        self, dataset: CIPADataset
    ) -> tuple[tuple[DimensionResult, ...], dict[str, Any]]:
        """Compute D1–D7 under the 2.0.0 protocol and return them with the run record."""
        from cipa import __version__
        from cipa._knn import _KNNCache, select_knn_algorithm
        from cipa.dimensions import (
            compute_d1,
            compute_d2,
            compute_d3,
            compute_d4_from_minority,
            compute_d5,
            compute_d6,
            compute_d7,
        )
        from cipa.preprocessing import preprocess_dataset

        rs = self._random_state
        t_start = perf_counter()
        prepared, prep_info = preprocess_dataset(dataset, self._scaling)
        t_prep = perf_counter() - t_start
        n = prepared.N

        def cache_for(ds: CIPADataset) -> _KNNCache:
            """Lazily fitted k-NN index on ds."""
            return _KNNCache(
                ds, k=self._k, algorithm=self._algorithm, n_jobs=self._n_jobs,
                chunk_size=self._chunk_size,
            )

        # D1, D5, D6: all N rows
        d1 = _single("D1", lambda: compute_d1(prepared), n_used=n)
        d5 = _single("D5", lambda: compute_d5(prepared, random_state=rs), n_used=n)
        d6 = _single(
            "D6", lambda: compute_d6(prepared, random_state=rs, n_jobs=self._n_jobs), n_used=n
        )

        # D3: every minority row queried against all N rows
        full_cache = cache_for(prepared)
        d3 = _single(
            "D3", lambda: compute_d3(prepared, knn_cache=full_cache, k=self._k), n_used=n,
        )
        d3.metadata["n_queries"] = prepared.n_minority

        # D2, D7: all rows, or IR-preserving draws of n_max rows (shared draws)
        d2_runs: list[DimensionResult] = []
        d7_runs: list[DimensionResult] = []
        t2 = t7 = 0.0
        seeds: list[int] = []
        minority_used: list[int] = []
        if n > self._n_max:
            seeds = [derive_seed(rs, STREAM_D2_D7, i) for i in range(self._n_subsamples)]
        for seed in seeds or [None]:
            t0 = perf_counter()
            if seed is None:
                sub, cache = prepared, full_cache
            else:
                idx = stratified_indices(prepared.minority_mask, self._n_max, seed)
                sub = prepared._subset(idx)
                cache = cache_for(sub)
            minority_used.append(sub.n_minority)
            d2_runs.append(compute_d2(
                sub, knn_cache=cache, k=self._k, weights=self._d2_weights, n_jobs=self._n_jobs,
                chunk_size=self._chunk_size, algorithm=self._algorithm,
            ))
            t1 = perf_counter()
            d7_runs.append(compute_d7(
                sub, svc_max_iter=self._svc_max_iter, random_state=rs, n_jobs=self._n_jobs,
                chunk_size=self._chunk_size, algorithm=self._algorithm,
                svc_tol=self._svc_tol,
            ))
            t2 += t1 - t0
            t7 += perf_counter() - t1
        n_used = n if not seeds else self._n_max
        d2 = _aggregate("D2", d2_runs, seeds, n_used, t2)
        d7 = _aggregate("D7", d7_runs, seeds, n_used, t7)
        for dim in (d2, d7):
            dim.metadata["n_minority_used"] = minority_used

        # D4: all minority rows, or draws of n_max minority rows
        t0 = perf_counter()
        X_min = prepared.X_minority
        d4_seeds: list[int] = []
        if prepared.n_minority > self._n_max:
            d4_seeds = [derive_seed(rs, STREAM_D4, i) for i in range(self._n_subsamples)]
        d4_runs = [
            compute_d4_from_minority(
                X_min if seed is None
                else X_min[minority_indices(prepared.n_minority, self._n_max, seed)],
                dbscan_min_samples=self._dbscan_min_samples,
                dbscan_eps=self._dbscan_eps,
                n_jobs=self._n_jobs,
            )
            for seed in d4_seeds or [None]
        ]
        d4 = _aggregate(
            "D4", d4_runs, d4_seeds,
            prepared.n_minority if not d4_seeds else self._n_max,
            perf_counter() - t0,
        )

        dims = (d1, d2, d3, d4, d5, d6, d7)
        logger.debug(
            "D1=%.4f D2=%.4f D3=%.4f D4=%.4f D5=%.4f D6=%.4f D7=%.4f",
            *(d.value for d in dims),
        )

        not_converged = sum(not r.components["converged"] for r in d7_runs)
        if not_converged:
            logger.warning("L1: %d of %d LinearSVC fits did not converge", not_converged, len(d7_runs))

        metadata: dict[str, Any] = {
            "cipa_version": __version__,
            "random_state": rs,
            "scaling": self._scaling,
            "n_max": self._n_max,
            "n_subsamples": self._n_subsamples,
            "k_neighbors": self._k,
            "svc_max_iter": self._svc_max_iter,
            "svc_tol": self._svc_tol,
            "algorithm": self._algorithm,
            "algorithm_resolved": select_knn_algorithm(prepared.d, self._algorithm),
            "chunk_size": self._chunk_size,
            "N": n,
            "d": prepared.d,
            "n_minority": prepared.n_minority,
            "n_majority": prepared.n_majority,
            "IR": prepared.IR,
            "minority_label": dataset.minority_label,
            "majority_label": dataset.majority_label,
            "minority_is_majority": dataset.minority_is_majority,
            "preprocessing": prep_info,
            "l1_fits": len(d7_runs),
            "l1_not_converged": not_converged,
            "time_seconds": {
                "preprocessing": t_prep,
                **{d.dimension_id: d.metadata["time_seconds"] for d in dims},
                "total": perf_counter() - t_start,
            },
        }
        return dims, metadata


def _single(
    dimension_id: str,
    compute: Callable[[], DimensionResult],
    n_used: int,
) -> DimensionResult:
    """Run a dimension once on all its rows and attach the protocol metadata."""
    t0 = perf_counter()
    result = compute()
    return _aggregate(dimension_id, [result], [], n_used, perf_counter() - t0)


def _aggregate(
    dimension_id: str,
    runs: list[DimensionResult],
    seeds: list[int],
    n_used: int,
    elapsed: float,
) -> DimensionResult:
    """Combine one or more computations of a dimension into a single result.

    A single computation is returned as is (IQR 0). With several draws, the
    value and every numeric component are medians over the draws, with their
    IQRs in ``metadata["components_iqr"]``; boolean components are True only
    if True in every draw; other components (e.g. lists) are kept per draw in
    ``metadata["components_per_subsample"]`` only.

    ``metadata["component_seconds"]`` is **summed**, not averaged, over the
    draws (2.0.0rc3), so it is comparable with ``time_seconds``: both answer
    how long this dimension took in total, one broken down and one not.
    """
    protocol = {
        "n_used": n_used,
        "subsampled": bool(seeds),
        "n_subsamples": len(runs),
        "seeds": list(seeds),
        "values": [r.value for r in runs],
        "time_seconds": elapsed,
    }
    per_component = [r.metadata.get("component_seconds") for r in runs]
    if all(isinstance(c, dict) for c in per_component) and per_component:
        totals: dict[str, float] = {}
        for entry in per_component:
            for name, seconds in entry.items():
                totals[name] = totals.get(name, 0.0) + seconds
        protocol["component_seconds"] = {n: round(v, 4) for n, v in totals.items()}
    if len(runs) == 1:
        run = runs[0]
        assert run.dimension_id == dimension_id
        return DimensionResult(
            value=run.value,
            dimension_id=dimension_id,
            components=dict(run.components),
            metadata={**run.metadata, **protocol},
            iqr=0.0,
        )

    value, iqr = median_iqr(protocol["values"])
    components: dict[str, Any] = {}
    components_iqr: dict[str, float] = {}
    per_subsample: dict[str, list[Any]] = {}
    for key in runs[0].components:
        values = [r.components[key] for r in runs]
        per_subsample[key] = values
        if all(isinstance(v, (bool, np.bool_)) for v in values):
            components[key] = all(values)
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            components[key], components_iqr[key] = median_iqr(values)
    protocol.update({
        "components_iqr": components_iqr,
        "components_per_subsample": per_subsample,
        "run_metadata": [r.metadata for r in runs],
    })
    return DimensionResult(
        value=value,
        dimension_id=dimension_id,
        components=components,
        metadata=protocol,
        iqr=iqr,
    )
