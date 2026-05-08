# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 2026-05-06

### Added

- `experiments/compute_rq1_stats.py`: reproduces all RQ1 statistics from §5.2
  using CIPA v1.1.0 DS values consistently throughout (Spearman ρ, Pearson r,
  Wilcoxon p for B1–B3). Resolves prior inconsistency where p=0.031 was
  computed with aspirational DS while correlations used v1.1.0 values. Definitive
  results: B1 p=0.040, B2 p=0.001, B3 p=0.0002 (abs_direct, two-sided).
- `experiments/weight_sensitivity.py`: three analyses over the 13-dataset
  benchmark — per-dimension Spearman ρ sensitivity, Monte Carlo search over the
  weight simplex (50,000 Dirichlet samples), and constrained SLSQP optimisation
  with 21 multi-starts. Addresses reviewer comment on manually specified weights:
  principled **w** outperforms 61.9% of random assignments (ρ = −0.71 vs
  median −0.68); optimal **w\*** gains Δ|ρ| = 0.07, attributable to
  overfitting on N=13.
- `experiments/ADDING_DATASETS.md`: step-by-step guide for integrating new
  datasets into the validation script and `datasets/` directory.

### Changed

- Documentation audit and full docstring coverage pass over all modules in
  `src/cipa/`.
- `README.md`: corrected D1 formula, D5 description, and Sig III/IV trigger
  to match the camera-ready paper.
- `experiments/validate_table2.py`: all `TABLE_2` values updated to official
  v1.1.0 computed results (4-digit precision); PaySim IR corrected to 772.7.
- All 22 module docstrings updated to reflect v1.2.0 and camera-ready formulas.

---

## [1.1.0] — 2026-03-30

### Changed

**D1 — Imbalance Distribution (breaking formula change)**

- Previous formula: `D1 = 1 − 1/log₂(IR + 1)`, derived from the imbalance
  ratio IR.
- New formula: `D1 = 1 − H(Y)`, where `H(Y) = −p⁺log₂p⁺ − p⁻log₂p⁻` is the
  normalised binary entropy of the label variable.
- Motivation: the entropy-based formula is grounded in information theory,
  maps naturally to [0, 1] for binary classification, and avoids the
  overestimation of D1 for mild imbalance (IR < 5) that the log-IR formula
  produced.
- Effect: D1 values decrease slightly for extreme IR, bringing them closer to
  the theoretical optimum. All downstream DS values are affected.

**D5 — Effective Dimensionality (breaking formula change)**

- Previous formula: ratio of PCA components needed to explain 95% of variance
  to total features.
- New formula: spectral entropy of the PCA eigenvalue distribution,
  `H(λ) / log(k)`, where `λᵢ = eigenvalueᵢ / Σeigenvalues` and
  `k = min(N−1, d)`.
- Motivation: spectral entropy quantifies the spread of the variance across
  principal components rather than a threshold-dependent count. A uniform
  eigenvalue distribution (all directions equally informative) now yields
  D5 = 1; a single dominant component yields D5 → 0.
- Effect: D5 is now sensitive to the geometry of the feature space rather than
  to the 95% variance threshold hyperparameter.

**Pipeline — selective subsampling (`knn_subsample` parameter)**

- `CIPAPipeline` now accepts a `knn_subsample` parameter (default `None`).
- When set and `N > knn_subsample`, computationally cheap dimensions (D1, D5,
  D6) use the full dataset while expensive k-NN / DBSCAN dimensions (D2, D3,
  D4, D7) use an asymmetrically subsampled dataset (all minority samples
  preserved; majority randomly drawn to fill the remaining budget).
- This preserves the accuracy of D1 (sensitive to IR) and D5 (sensitive to
  feature geometry) while keeping runtime manageable for large-N datasets.

### Fixed

- `ecol/n2.py`: corrected the N2 normalisation formula from `1/(1 + N2_raw)`
  to `N2_raw/(1 + N2_raw)` so that higher values correctly indicate harder
  decision boundaries.

---

## [1.0.0] — 2026-03-01

### Added

Initial public release implementing the four-stage CIPA framework:

**Stage C — Characterization (seven complexity dimensions)**
- D1: Imbalance Distribution — normalised log-IR formula.
- D2: Class Overlap — composite of F3 (Maximum Fisher Discriminant Ratio),
  N1 (MST-based borderline fraction), and kDN (k-Disagreeing Neighbors).
- D3: Instance Hardness — Napierała–Stefanowski typology (safe, borderline,
  rare, outlier) weighted by severity.
- D4: Sub-concept Fragmentation — ECindex × √(n\_clusters / n\_minority).
- D5: Effective Dimensionality — fraction of PCA components explaining ≥ 95%
  of variance.
- D6: Feature Informativeness — KSG mutual-information deficit.
- D7: Boundary Complexity — composite of L1 (LinearSVC error) and N2
  (intra/inter-class distance ratio).

**Stage I — Indexing**
- Weighted aggregation `DS = Σ wᵢ Dᵢ` with default weights
  `(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13)`.
- Interpretation bands: Low [0, 0.25), Moderate [0.25, 0.50),
  High [0.50, 0.75), Extreme [0.75, 1].

**Stage P — Profiling**
- Five named complexity signatures (I–V) assigned by a dominance rule on the
  dimension vector.

**Stage A — Action**
- Signature- and band-driven recommendations across four axes: evaluation
  metrics, preprocessing strategy, model families, and validation protocol.

**Infrastructure**
- `CIPADataset`: validated binary classification dataset wrapper.
- `CIPAPipeline`: end-to-end orchestrator with shared k-NN cache (D2, D3, D7).
- All result types as frozen dataclasses with `to_dict()` / JSON serialization.
- Large-N automatic subsampling in N1 (MST), N2, and DBSCAN (cap at 50 k).
- 212 tests, ≥ 90% coverage gate.
- Pure Python (NumPy, SciPy, scikit-learn); no R dependency.

---

[1.2.0]: https://github.com/luisgarciarodriguez-research/cipa/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/luisgarciarodriguez-research/cipa/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/luisgarciarodriguez-research/cipa/releases/tag/v1.0.0
