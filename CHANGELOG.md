# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0rc2] — 2026-09-17

Second release candidate of 2.0.0. **Only D5 changes**; everything else in
2.0.0rc1 (C1–C9, scaling, subsampling, signatures, action) is unchanged.
Decided in CIPA Extended (protocol decision 5, 2026-09-17) after two pilots
on `v2.0.0rc1` over 28 datasets (`cipa-extended/results/cipa/pilot/`:
`scaling_pilot_summary.md`, `d5_candidates.json`).

### Changed — D5 redefined (breaking value change)

- **D5 = effective dimensionality relative to the minority.**

      r_95 = smallest number of principal components whose cumulative
             explained variance ratio is ≥ 0.95
      ρ    = r_95 / |C+|
      D5   = ρ / (1 + ρ)        ∈ [0, 1); 0.5 when r_95 = |C+|

  Same PCA fit as before (`PCA(n_components=min(N−1, d))` on the
  preprocessed matrix with all N rows), so r_95 ≤ min(N−1, d). ρ/(1+ρ) is the
  map D7 already applies to N2. `r_95 = searchsorted(cumsum(ratios), 0.95) + 1`
  with no tolerance, capped at the number of fitted components if rounding
  keeps the cumulative sum below 0.95. d = 1 or N ≤ 2 still return 0; a single
  non-zero component gives r_95 = 1 and D5 = 1/(1 + |C+|). The threshold is
  `cipa._constants.D5_VARIANCE_THRESHOLD` = 0.95 and is not a pipeline
  parameter (the CIPA Extended protocol fixes it).
- **Why.** With z-score (rc1, C2) the normalised spectral entropy measures
  lack of redundancy, not dimensionality: it exceeded 0.5 in 24 of the 28
  CIPA Extended datasets and made Signature IV cover 15 of the 20 fully
  profiled ones (PIMA, d = 8: D5 = 0.93, "dimensionality-dominated"). Of four
  candidates fixed before running the pilot (effective rank exp(H) or r_95,
  relative to N or to |C+|), r_95/|C+| was chosen. It also answers the
  criticism that D5 ignored the minority class and approached 1 on isotropic
  noise.
- **The spectral entropy remains available** as an informative component
  that does not enter D5: `H_nats`, `H_max_nats`, `spectral_entropy_norm`
  (= the 1.x/rc1 D5) and `n_components_fit`. New components `r_95`,
  `n_minority` and `rho`; new metadata `variance_threshold`.
- *Effect on values.* On the pilot reference datasets (`scaling="standard"`,
  seed 42, full N): tcga_brca 0.758 → 0.785 (r_95 = 536, |C+| = 147),
  secom 0.797 → 0.609, ozone_level 0.513 → 0.215, pima_diabetes 0.927 →
  0.029, abalone_19 0.364 → 0.086, credit_card_fraud 0.983 → 0.052. D5 now
  exceeds 0.5 only when there are more effective dimensions than minority
  instances. DS moves by 0.10·ΔD5 with the default weights.
- *Effect on signatures* (known, rule unchanged). Signature IV becomes rare:
  replacing D5 in the 20 rc1 pilot profiles (`scaling="standard"`) leaves
  only tcga_brca in IV (rc1: IV 15 · I 4 · II 1; rc2: I 10 · III 5 · II 2 ·
  V 2 · IV 1). secom has D5 = 0.609 but D1 is larger, so it is Signature I.
  The 18-case signature table and the 13 COMIA profiles in the tests use
  fixed vectors and do not change.
- *Effect on the action protocol* (known, **not corrected**; out of scope as
  in rc1). `cipa.action` was not modified and still requires D5 ≥ 0.70 for
  mandatory dimensionality reduction; with the new D5 that means
  r_95 ≥ 2.33·|C+|, so the recommendation fires far less often, and Signature
  IV recommendations follow the signature change above.

### Tests

- `tests/unit/test_d5.py` rewritten: exact value where r_95 is known by
  construction (data in a k-dimensional subspace), ρ/(1+ρ) consistency,
  D5 = 0.5 when r_95 = |C+|, monotone decrease with more minority instances
  at the same spectrum, bound below 1 with d > N, cap of r_95 when the
  cumulative ratio never reaches 0.95, degenerate cases (d = 1, N ≤ 2, a
  single component) and the spectral-entropy component.
- `tests/regression/test_regression_v1_2_1.py`: D5 is excluded from the exact
  value comparison; its `spectral_entropy_norm`, `H_nats`, `H_max_nats` and
  `n_components_fit` must equal the v1.2.1 D5 exactly. D1–D4, D6 and D7 stay
  exact. The frozen reference is unchanged.
- `tests/integration/test_d5_reference.py` (new): the six pilot reference
  values (tcga_brca, secom, ozone_level, pima_diabetes, abalone_19,
  credit_card_fraud) with tolerance 1e-3 and exact r_95. Runs only where
  imbdata is installed and the datasets are cached; reproduced bit for bit
  with imbdata 0.3.1.

### Documentation

- README: D5 row, formula notes, PIMA example output (Signature IV → II,
  DS 0.475 → 0.385), Signature IV trigger, D5 action note, `DimensionResult`
  components of D5 and the COMIA reproduction note (the published D5 is now
  `spectral_entropy_norm`).
- `specs/` remain out of date (SPEC-02, 05, 06, 08, 10, the 2.0.0 API and now
  the D5 definition). Pending.

---

## [2.0.0rc1] — 2026-09-16

First release candidate of 2.0.0, prepared for CIPA Extended. **The formulas
of v1.1.0 are unchanged**; the values change because of how the data reach
them (scaling, per-dimension subsampling, duplicates), how L1 reports
non-convergence and how signatures are assigned. `2.0.0` will be tagged after
the CIPA Extended pilot (runtime and scaling sensitivity), which may still
adjust the subsampling protocol.

### Unchanged formulas

D1 = 1 − H(Y); D2 = (F3 + N1 + kDN)/3 with kDN over all instances (k = 5);
D3 = (B + 2R + 3O)/(3·|C+|) (k = 5); D4 = ECindex·√(n_clusters/|C+|) with
DBSCAN on the minority (`min_samples` = 3, eps = median distance to the 2nd
neighbour, noise = singletons, eps doubled once if all noise); D5 = normalised
spectral entropy of PCA; D6 = 1 − Ī/H(Y) with `mutual_info_classif`;
D7 = (L1 + N2/(1+N2))/2 with `LinearSVC(class_weight="balanced")`;
DS = Σ wᵢDᵢ and the Low/Moderate/High/Extreme bands.

`tests/regression/test_regression_v1_2_1.py` checks that, on duplicate-free
data with `scaling="none"`, no subsampling and the same seed, D1–D7 equal the
values frozen from the `v1.2.1` tag (exact equality; D7 too because L1
converged in every case). On the real datasets, the eight COMIA benchmarks
that were not subsampled (Breast Cancer W., PIMA, SVMGUIDE1, Yeast-ME3,
Ecoli-iMU, TCGA-BRCA, CWRU, SEU Gearbox) also reproduce the published D1–D7
and DS to four decimals with `scaling="none"`.

### Changed — corrections

- **C1 · Explicit minority class.** `CIPADataset` takes the class roles as
  declared. If the declared minority outnumbers the declared majority it logs
  and emits a `UserWarning`, never swaps the labels, and the pipeline records
  `minority_is_majority` in `CIPAResult.metadata`.
  `CIPADataset.from_arrays` (minority by frequency) is deprecated with a
  `DeprecationWarning` and no longer used anywhere in the pipeline.
  *Effect:* none on correctly labelled data; prevents the PaySim inversion.
- **C2 · Constant columns out and scaling.** New `scaling` parameter
  (`"standard"` default, `"robust"`, `"none"`) applied once per dataset to all
  N rows, before any dimension and any subsampling (`cipa.preprocessing`).
  Constant columns (max = min) are dropped first and reported by index.
  `"robust"` divides by the IQR, or by σ where IQR = 0 (reported), and is meant
  for sensitivity analysis only.
  *Effect:* N1, kDN, D3, D4, D5, L1 and N2 no longer depend on units.
  D5 changes most: it now measures correlation structure, and on weakly
  correlated feature sets the standardised spectrum is nearly flat, so D5
  rises (PIMA 0.23 → 0.93). With z-score, 7 of the 8 full-N COMIA datasets
  become Signature IV. D6 no longer averages the zero MI of constant columns.
  D1 and F3 are unaffected.
- **C3 · Neighbours that tolerate duplicates.** Self-exclusion is by index
  (`cipa._knn.kneighbors_excluding_self`), not by dropping column 0; exact
  duplicates of other rows remain neighbours. Applies to kDN (D2), D3 and N2.
  *Effect:* only on data with exact duplicates, where kDN and D3 could count
  the instance itself as a neighbour and miss a twin with another label.
- **C4 · N1 without a dense matrix.** New
  `cipa.ecol.euclidean_minimum_spanning_tree`: exact Euclidean MST by Borůvka
  over precomputed neighbour lists, O(N·k) memory (v1.2.1 built
  `squareform(pdist(X))`, about 20 GB at N = 50,000). Zero-length edges are
  valid, so identical rows with different labels are both borderline (scipy's
  MST treated 0 entries as missing edges). Ties are broken by the key
  (length, min(i, j), max(i, j)).
  *Effect:* identical to the dense method on duplicate-free data; on data
  with duplicates N1 can rise.
- **C5 · L1 convergence.** `max_iter` default 2,000 → 10,000 (`svc_max_iter`).
  L1 is always the training error obtained; the fixed 0.5 fallback is gone.
  D7 components include `converged`; `CIPAResult.metadata` reports
  `l1_fits` and `l1_not_converged`.
  *Effect:* D7 changes only where LinearSVC did not converge in v1.2.1.
- **C6 · Per-dimension subsampling.** Replaces `knn_subsample`, the asymmetric
  subsample (all minority + majority fill) and the internal N1/N2 subsampling
  with one protocol (`n_max` = 50,000, `n_subsamples` = 5):
  D1, D5, D6 on all N; D3 with every minority row queried against all N;
  D4 on all minority rows, or `n_subsamples` draws of `n_max` minority rows
  if |C+| > `n_max`; D2 and D7 on all N if N ≤ `n_max`, otherwise on
  `n_subsamples` shared stratified draws of `n_max` rows that keep the IR
  (±1 row). With draws, each dimension and each numeric component is the
  median, with its IQR (`DimensionResult.iqr`,
  `metadata["components_iqr"]`); DS uses the medians. Each `DimensionResult`
  records `n_used`, `n_subsamples`, `seeds`, `values` and `time_seconds`.
  `n_jobs` and `chunk_size` are accepted for neighbour queries.
  *Effect:* D1 reflects the real IR (CreditCard 0.717 published → 0.982 on
  full N); D2/D7 keep the real class proportion instead of an enriched
  minority, so large datasets no longer look easier.
- **C7 · Signatures follow the paper.** `compute_profile` implements
  "Dᵢ dominates if Dᵢ > τ = 0.50 and Dᵢ = max{D1, D2, D4, D5}" (I–IV for D1,
  D2, D4, D5; V when none dominates), with ties resolved D1 > D2 > D4 > D5.
  Signature V carries a qualifier over all seven dimensions — `compound`
  (≥ 2 > τ′ = 0.35), `single` (exactly one) or `low` (none) — which closes the
  case the paper leaves uncovered. **The qualifier was confirmed by the
  author on 2026-09-16, unchanged from this release candidate.** τ and τ′ are
  parameters.
  The 1.x priority rule (0.25/0.55 thresholds, D5 − D2 margin) is removed.
  *Effect:* on the 13 published COMIA profiles, 5 change signature
  (IEEE-CIS V → I, CreditCard V → I, PaySim V → IV, SEU Gearbox V → I,
  CWRU V → I).
- **C8 · Action protocol unchanged.** `cipa.action` was not modified and
  still uses its 1.x thresholds. It consumes the signature, so its
  recommendations change with C7. **Known effect, not corrected** in this
  release.
- **C9 · Determinism.** `random_state` defaults to 42 and must be a
  non-negative integer (`None` raises `ValueError`). Draw seeds derive from it
  through `numpy.random.SeedSequence`; the integer is passed to
  `mutual_info_classif`, `LinearSVC` and PCA.

### Changed — API (breaking)

- `CIPAPipeline`: parameters after `weights` are keyword-only. New
  `scaling`, `n_max`, `n_subsamples`, `n_jobs`, `tau`, `tau_prime`,
  `chunk_size`; removed `knn_subsample`, `n1_max_exact`, `large_n_subsample`.
  `random_state` default `None` → 42.
- `run_scoring_only` returns a `CIPAResult` with `profile` and
  `action=None` (was a `DifficultyScore`), so one call yields DS, band,
  signature and qualifier.
- `CIPAResult`: `action` may be `None`; new `metadata` (class counts,
  `minority_is_majority`, preprocessing, protocol, L1 convergence, timings).
- `DimensionResult`: new `iqr` field (serialised by `to_dict`).
- `ComplexityProfile`: `dominant_dimensions` replaced by
  `dominant_dimension`, `qualifier`, `active_dimensions` (> τ),
  `elevated_dimensions` (> τ′), `tau`, `tau_prime`.
- `compute_n1` and `compute_n2` return a float (no subsampling flag) and take
  `n_jobs`/`chunk_size` instead of subsampling parameters. `compute_d2` and
  `compute_d7` drop their N1/N2 subsampling parameters; `compute_d4`,
  `compute_d6` and `compute_d7` accept `n_jobs`; `compute_d5` accepts
  `random_state`. New `compute_d4_from_minority`.
- `_constants`: removed `ELEVATION_THRESHOLD`, `LOW_THRESHOLD`,
  `DOMINANCE_MARGIN`, `DEFAULT_N1_MAX_EXACT`, `DEFAULT_LARGE_N_SUBSAMPLE`;
  `DEFAULT_SVC_MAX_ITER` 2,000 → 10,000.

### Added

- Tests: v1.2.1 regression (`tests/regression/`, with the script that freezes
  the reference from the tag), exact MST against the dense method and a keyed
  Kruskal reference with duplicates and ties, bounded memory at N = 50,000,
  duplicate-aware neighbours, preprocessing, the subsampling protocol, the
  signature case table and the 13 COMIA profiles under the paper rule.
- `pyproject.toml`: ruff ignores RUF001–RUF003 (the paper's σ, −, × notation)
  and N801 in tests; `ruff check src tests` is clean.

### Notes

- **Version metadata of 1.2.1.** The `v1.2.1` tag (`6ea039f`, 2026-05-08,
  submitted with the COMIA 2026 paper) kept `version = "1.2.0"` in
  `pyproject.toml` and `cipa.__version__` and has no entry of its own: its two
  commits after `v1.2.0` (`4c26fa6`, `6ea039f`) added
  `experiments/weight_sensitivity.py`, docstrings and documentation fixes, and
  are listed under [1.2.0] below. No computed value differs between the two
  tags. Installing `v1.2.1` reports 1.2.0; the tag was not rewritten.
- `experiments/` scripts reproduce the COMIA results and target the 1.x API
  (`from_arrays`, published signatures); run them against the `v1.2.1` tag.
- `specs/` are out of date with respect to the code (SPEC-02, 05, 06, 08, 10
  and the 2.0.0 API). Pending.

### Findings about the values published in COMIA 2026

- **PaySim was computed with the classes inverted.** The 10,000-row
  subsample left fraud as the majority and `from_arrays` picked the minority
  by frequency: D1 = 0.3228 = 1 − H(0.1787).
- **D1 was computed on the 10,000-row subsamples** for the six † datasets,
  not on the real IR: CreditCard published D1 = 0.717; on the full N it is
  0.982.
- **IEEE-CIS kept 350 minority instances**, not all of them, according to the
  review. Not verified.
- **The paper and the code differ** in kDN (paper: minority only; code: all
  instances), DBSCAN parameters (paper: MinPts = max(2, ⌈0.05·|C+|⌉); code:
  `min_samples` = 3 and median 2nd-neighbour eps), the signature rule (code
  before 2.0.0: priority rule with 0.25/0.55 thresholds) and the action
  protocol thresholds. Published Table 3 comes from the code.

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
