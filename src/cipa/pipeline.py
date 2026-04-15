"""CIPAPipeline: orchestrates the full CIPA framework (C → I → P → A).

See §3 of García Rodríguez et al. (2026) for the data flow.

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

from cipa._constants import (
    DEFAULT_D2_WEIGHTS,
    DEFAULT_DBSCAN_MIN_SAMPLES,
    DEFAULT_K,
    DEFAULT_LARGE_N_SUBSAMPLE,
    DEFAULT_N1_MAX_EXACT,
    DEFAULT_SVC_MAX_ITER,
    DEFAULT_WEIGHTS,
)
from cipa.dataset import CIPADataset
from cipa.types import CIPAResult, DifficultyScore, DimensionResult

logger = logging.getLogger(__name__)


class CIPAPipeline:
    """Orchestrates the four CIPA stages: C → I → P → A.

    Parameters
    ----------
    weights : tuple of 7 floats
        Dimension weights for DS computation. Must sum to 1.
    k_neighbors : int
        k for kDN (D2) and NS-typology (D3). Shared k-NN fit.
    dbscan_min_samples : int
        DBSCAN min_samples for D4.
    dbscan_eps : float or None
        DBSCAN eps for D4. None = adaptive.
    n1_max_exact : int
        Max N for exact N1 computation (MST).
    large_n_subsample : int
        Subsample size for large-N paths in N1, N2.
    random_state : int or None
        For reproducibility across D4, D6, D7.
    d2_weights : tuple of 3 floats
        (alpha, beta, gamma) for (F3, N1, kDN) in D2. Must sum to 1.
    knn_subsample : int or None
        When set and N > knn_subsample, apply selective subsampling:
        D1, D5, D6 use the full dataset (cheap, O(N·d) or less);
        D2, D3, D4, D7 use an asymmetric subsample of knn_subsample rows
        (expensive k-NN / DBSCAN). All minority samples are preserved;
        majority is randomly drawn to fill the remaining budget.
        None = no subsampling (default).
    """

    def __init__(
        self,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
        k_neighbors: int = DEFAULT_K,
        dbscan_min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
        dbscan_eps: float | None = None,
        n1_max_exact: int = DEFAULT_N1_MAX_EXACT,
        large_n_subsample: int = DEFAULT_LARGE_N_SUBSAMPLE,
        random_state: int | None = None,
        d2_weights: tuple[float, float, float] = DEFAULT_D2_WEIGHTS,
        svc_max_iter: int = DEFAULT_SVC_MAX_ITER,
        knn_subsample: int | None = None,
    ) -> None:
        self._weights = weights
        self._k = k_neighbors
        self._dbscan_min_samples = dbscan_min_samples
        self._dbscan_eps = dbscan_eps
        self._n1_max_exact = n1_max_exact
        self._large_n_subsample = large_n_subsample
        self._random_state = random_state
        self._d2_weights = d2_weights
        self._svc_max_iter = svc_max_iter
        self._knn_subsample = knn_subsample

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, dataset: CIPADataset) -> CIPAResult:
        """Execute the full CIPA pipeline: C → I → P → A.

        Fits a single k-NN model shared by D2, D3, and D7.
        All seven dimensions are computed, then aggregated into DS,
        profiled into a Signature, and mapped to Action recommendations.

        Parameters
        ----------
        dataset : CIPADataset

        Returns
        -------
        CIPAResult

        Raises
        ------
        ValueError
            If dataset fails validation (propagated from CIPADataset).
        """
        from cipa.action import compute_action
        from cipa.indexing import compute_difficulty_score
        from cipa.profiling import compute_profile

        logger.info("CIPAPipeline.run: dataset=%s (N=%d, d=%d, IR=%.2f)",
                    dataset.name, dataset.N, dataset.d, dataset.IR)

        dims = self._compute_dimensions(dataset)
        ds = compute_difficulty_score(dims, self._weights)
        profile = compute_profile(ds)
        action = compute_action(profile, ds, dataset)

        return CIPAResult(
            dataset_name=dataset.name,
            difficulty_score=ds,
            profile=profile,
            action=action,
        )

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
        return self._compute_dimensions(dataset)

    def run_scoring_only(self, dataset: CIPADataset) -> DifficultyScore:
        """Run Stages C and I (Characterization + Indexing). Returns the Difficulty Score.

        Skips the Profiling and Action stages. Useful when only the scalar DS
        value is needed.

        Parameters
        ----------
        dataset : CIPADataset

        Returns
        -------
        DifficultyScore
            value   : DS ∈ [0, 1]
            band    : "Low" | "Moderate" | "High" | "Extreme"
            weights : weights used for aggregation
            dimensions : the seven DimensionResult objects
        """
        from cipa.indexing import compute_difficulty_score

        dims = self._compute_dimensions(dataset)
        return compute_difficulty_score(dims, self._weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_dimensions(
        self, dataset: CIPADataset
    ) -> tuple[DimensionResult, ...]:
        """Compute D1-D7, sharing a single k-NN fit for D2, D3, D7.

        When knn_subsample is set and N > knn_subsample, selective subsampling
        is applied: D1, D5, D6 run on the full dataset; D2, D3, D4, D7 run on
        an asymmetrically subsampled dataset (all minority + sampled majority).
        """
        from cipa._knn import _KNNCache
        from cipa.dimensions import (
            compute_d1,
            compute_d2,
            compute_d3,
            compute_d4,
            compute_d5,
            compute_d6,
            compute_d7,
        )

        # Selective subsampling for expensive k-NN / DBSCAN dimensions
        if self._knn_subsample and dataset.N > self._knn_subsample:
            ds_knn = _asymmetric_subsample(
                dataset, self._knn_subsample, self._random_state
            )
            logger.info(
                "knn_subsample: N=%d → %d (minority kept=%d, majority sampled=%d)",
                dataset.N, ds_knn.N, ds_knn.n_minority, ds_knn.n_majority,
            )
        else:
            ds_knn = dataset

        # D1, D5, D6: cheap — use full dataset
        d1 = compute_d1(dataset)
        d5 = compute_d5(dataset)
        d6 = compute_d6(dataset, random_state=self._random_state)

        # D2, D3, D4, D7: expensive k-NN / DBSCAN — use (possibly subsampled) dataset
        cache = _KNNCache(ds_knn, k=self._k)
        d2 = compute_d2(
            ds_knn,
            knn_cache=cache,
            k=self._k,
            weights=self._d2_weights,
            n1_max_exact=self._n1_max_exact,
            n1_subsample_size=self._large_n_subsample,
            random_state=self._random_state,
        )
        d3 = compute_d3(ds_knn, knn_cache=cache, k=self._k)
        d4 = compute_d4(
            ds_knn,
            dbscan_min_samples=self._dbscan_min_samples,
            dbscan_eps=self._dbscan_eps,
            random_state=self._random_state,
        )
        d7 = compute_d7(
            ds_knn,
            knn_cache=cache,
            svc_max_iter=self._svc_max_iter,
            n2_max_exact=self._n1_max_exact,
            n2_subsample_size=self._large_n_subsample,
            random_state=self._random_state,
        )

        logger.debug(
            "D1=%.4f D2=%.4f D3=%.4f D4=%.4f D5=%.4f D6=%.4f D7=%.4f",
            d1.value, d2.value, d3.value, d4.value, d5.value, d6.value, d7.value,
        )

        return (d1, d2, d3, d4, d5, d6, d7)


def _asymmetric_subsample(
    dataset: CIPADataset,
    n_target: int,
    random_state: int | None,
) -> CIPADataset:
    """Return a new CIPADataset with at most n_target rows.

    Strategy: keep all minority samples; randomly draw majority samples to
    fill the remaining budget. If minority alone exceeds n_target, fall back
    to proportional stratified sampling.
    """
    import numpy as np

    rng     = np.random.default_rng(random_state)
    min_idx = np.where(dataset.y == dataset.minority_label)[0]
    maj_idx = np.where(dataset.y == dataset.majority_label)[0]
    n_min   = len(min_idx)

    if n_min >= n_target:
        n_keep_min = max(2, round(n_target * n_min / dataset.N))
        n_keep_maj = n_target - n_keep_min
        sel_min = rng.choice(min_idx, size=n_keep_min, replace=False)
        sel_maj = rng.choice(maj_idx, size=min(n_keep_maj, len(maj_idx)), replace=False)
    else:
        n_keep_maj = n_target - n_min
        sel_min    = min_idx
        sel_maj    = rng.choice(maj_idx, size=min(n_keep_maj, len(maj_idx)), replace=False)

    sel = np.concatenate([sel_min, sel_maj])
    rng.shuffle(sel)

    return CIPADataset(
        X=dataset.X[sel],
        y=dataset.y[sel],
        minority_label=dataset.minority_label,
        majority_label=dataset.majority_label,
        name=dataset.name,
    )
