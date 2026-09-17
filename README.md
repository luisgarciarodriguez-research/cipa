# CIPA

**A Multi-Domain Statistical Framework for Characterizing Imbalanced Datasets and Computing a Difficulty Score**

CIPA (_Characterization, Indexing, Profiling, Action_) analyzes a binary classification dataset along seven non-redundant complexity dimensions and produces a single **Difficulty Score** (DS ∈ [0, 1]), a structural fingerprint, and actionable preprocessing recommendations — all without training any model.

---

## Why CIPA?

Preprocessing decisions (resampling strategy, evaluation metric, model family, validation protocol) must be made _before_ any model is trained, yet are typically based on the imbalance ratio (IR) alone. IR is a poor predictor of classification difficulty: class overlap, borderline instances, minority fragmentation, effective dimensionality, and feature informativeness all shape learning difficulty independently of IR and must be quantified before intervening.

CIPA formalizes this characterization step as a four-stage pipeline that measures seven structural properties, aggregates them into a normalized index, and maps the resulting profile to concrete, dimension-specific recommendations.

**Empirical validation across 13 benchmark datasets from 5 domains (Finance, Medical, Cybersecurity, Industry, Bioinformatics)**, as published in COMIA 2026 with cipa v1.1.0–v1.2.1 values:

| Predictor | Spearman ρ vs AUC-PR | Wilcoxon p vs DS |
|-----------|---------------------|-----------------|
| **CIPA DS** | **−0.71** | — |
| IR alone (B1) | −0.60 | 0.040 |
| F3 alone (B2) | −0.28 | 0.001 |
| ECoL distance (B3) | −0.55 | 0.0002 |

DS significantly outperforms all three baselines (pairwise Wilcoxon, α = 0.05).

> **cipa 2.0.0 changes the computed values** (scaling, per-dimension subsampling, duplicates, L1 convergence, signature rule, and D5 redefined as effective dimensionality relative to the minority). The published values are reproducible with the `v1.2.1` tag. See [Reproducing COMIA 2026 values](#reproducing-comia-2026-values) and `CHANGELOG.md`.

---

## Applicability

| Requirement | Details |
|---|---|
| **Task** | Binary classification only |
| **Feature matrix X** | Numeric (`float`), shape (N, d) — encode categoricals before passing to CIPA |
| **Label vector y** | 1-D integer array with exactly two distinct values |
| **Class structure** | One class of interest (normally the minority), declared explicitly with `minority_label` and `majority_label` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/luisgarciarodriguez-research/cipa
cd cipa
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install

```bash
pip install -e .          # runtime dependencies only
pip install -e ".[dev]"   # add test and lint tools
```

**Requirements:** Python ≥ 3.11, NumPy ≥ 1.24, SciPy ≥ 1.10, scikit-learn ≥ 1.3.

### 4. Verify

```bash
python -c "import cipa; print(cipa.__version__)"
# 2.0.0rc2
```

To install a tagged release without touching other packages in the environment:

```bash
pip install --no-deps "cipa @ git+https://github.com/luisgarciarodriguez-research/cipa@v2.0.0rc2"
```

---

## Quick start

### From NumPy arrays

```python
import numpy as np
from cipa import CIPADataset, CIPAPipeline

# 1. Wrap your data
X = np.load("features.npy")   # shape (N, d), numeric
y = np.load("labels.npy")     # 1-D integer array

# Declare the class roles explicitly (never inferred from frequency)
dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name="MyDataset")

# 2. Run the full pipeline (z-score scaling, n_max = 50,000, seed 42 by default)
pipeline = CIPAPipeline(random_state=42)
result = pipeline.run(dataset)

# 3. Read the results
ds = result.difficulty_score
print(f"Difficulty Score : {ds.value:.3f}  ({ds.band})")
print(f"Signature        : {result.profile.signature} — {result.profile.signature_name}")
print()
print("Per-dimension breakdown:")
for dim in ds.dimensions:
    contrib = ds.contributions[dim.dimension_id]
    print(f"  {dim.dimension_id}  {dim.value:.3f}  (contributes {contrib:.3f})")
print()
print("Top recommendations:")
print("  Metrics      :", result.action.evaluation_metrics[:2])
print("  Preprocessing:", result.action.preprocessing_strategy[:1])
print("  Models       :", result.action.model_families[:2])
print("  Validation   :", result.action.validation_protocol[:1])
```

**Example output** (PIMA Diabetes, N=768, IR=1.9:1, cipa 2.0.0rc2 defaults):

```
Difficulty Score : 0.385  (Moderate)
Signature        : II — Overlap-dominated

Per-dimension breakdown:
  D1  0.067  (contributes 0.007)
  D2  0.584  (contributes 0.129)
  D3  0.262  (contributes 0.047)
  D4  0.285  (contributes 0.043)
  D5  0.029  (contributes 0.003)
  D6  0.934  (contributes 0.112)
  D7  0.348  (contributes 0.045)

Top recommendations:
  Metrics      : ['AUC-PR', 'F1-score (minority class)']
  Preprocessing: ['Borderline-SMOTE or ADASYN']
  Models       : ['RBF-SVM', 'Kernel methods']
  Validation   : ['Stratified k-fold cross-validation (k=5 or k=10)']
```

PIMA has 8 effective dimensions (r₉₅ = 8) for 268 minority instances, so D5 = (8/268)/(1 + 8/268) = 0.029. The COMIA 2026 value, D5 = 0.2336, was the spectral entropy of the unscaled data; 2.0.0 still reports it as the component `spectral_entropy_norm` (0.927 after z-scoring, where the flat spectrum of eight weakly correlated features made Signature IV almost universal in rc1).

### From a pandas DataFrame

```python
import pandas as pd
from cipa import CIPADataset, CIPAPipeline

df = pd.read_csv("data.csv")
X = df.drop(columns=["label"]).values.astype(float)
y = df["label"].values.astype(int)

dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name="MyDataset")
result = CIPAPipeline(random_state=42).run(dataset)

ds = result.difficulty_score
print(f"DS = {ds.value:.3f}  ({ds.band})  Sig: {result.profile.signature}")
```

---

## The four stages

```
CIPADataset
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  C  Characterization  —  compute D1 through D7        │
├──────────────────────────────────────────────────────┤
│  I  Indexing          —  DS = Σ wᵢ · Dᵢ  ∈ [0, 1]   │
├──────────────────────────────────────────────────────┤
│  P  Profiling         —  Complexity Signature I–V     │
├──────────────────────────────────────────────────────┤
│  A  Action            —  Preprocessing recommendations│
└──────────────────────────────────────────────────────┘
    │
    ▼
CIPAResult
```

### Stage C — Seven complexity dimensions

All dimensions are normalized to [0, 1]; higher values indicate higher difficulty.

| ID | Name | Formula | What it measures |
|----|------|---------|-----------------|
| D1 | Imbalance Distribution | `1 − H(Y)` | Class distribution skew via binary entropy of the label: 0 = balanced, 1 = extreme imbalance |
| D2 | Class Overlap | `(F3 + N1 + kDN) / 3` | Geometric inseparability: F3 (Fisher discriminant ratio), N1 (exact Euclidean MST boundary fraction), kDN (fraction of the k = 5 neighbours with another label, averaged over **all** instances) |
| D3 | Instance Hardness | Napierała–Stefanowski typology | Proportion of minority instances classified as borderline, rare, or outlier, weighted by severity |
| D4 | Sub-concept Fragmentation | `ECindex × (n_clusters / \|C₊\|)^0.5` | How fragmented the minority concept is across DBSCAN sub-regions (`min_samples` = 3, eps = median distance to the 2nd neighbour, noise = singleton groups; ECindex = Error Concentration) |
| D5 | Effective Dimensionality relative to the minority | `ρ / (1 + ρ)`, `ρ = r₉₅ / \|C₊\|` | PCA components needed for 95 % of the variance per minority instance: 0 = many minority instances per effective dimension, 0.5 = as many effective dimensions as minority instances |
| D6 | Feature Informativeness | `1 − Ī(X;Y) / H(Y)` | Mutual information deficit between features and the label (KSG estimator): 0 = fully discriminative, 1 = uninformative |
| D7 | Boundary Complexity | `(LinearSVC error + N2_norm) / 2` | Linear separability error combined with neighbourhood non-linearity (N2 ratio) |

where `H(Y) = −p⁺ log₂ p⁺ − p⁻ log₂ p⁻` is the binary entropy of the label; `p±= |C±| / N`. L1 is the training error of `LinearSVC(class_weight="balanced")`. For D5, PCA keeps min(N−1, d) components, so r₉₅ ≤ min(N−1, d); ρ/(1+ρ) is the same map D7 applies to N2. The normalised spectral entropy of the PCA spectrum, `H(p) / ln(k)` (the D5 of cipa 1.x and COMIA 2026), is still reported as the informative component `spectral_entropy_norm` and does not enter D5.

#### Computation protocol (2.0.0)

1. **Preprocessing, once per dataset on all N rows.** Constant columns (σ = 0) are dropped and reported; the remaining columns are scaled with `scaling="standard"` (z-score, default), `"robust"` (median/IQR, falling back to σ when IQR = 0; for sensitivity analysis only) or `"none"`. Every dimension sees the same matrix.
2. **Each dimension uses the rows that match what it measures:**

   | Dimension | Rows used | Output |
   |---|---|---|
   | D1, D5, D6 | all N | value |
   | D3 | every minority row as a query; neighbours searched among all N | value |
   | D4 | all minority rows; if \|C₊\| > `n_max`, `n_subsamples` draws of `n_max` minority rows | median + IQR |
   | D2 (F3, N1, kDN), D7 (L1, N2) | all N if N ≤ `n_max`; otherwise `n_subsamples` stratified draws of `n_max` rows that keep the IR (±1 row) | median + IQR, per dimension and per component |

   DS is computed from the medians. D2 and D7 share the same draws.
3. **Neighbours tolerate duplicates.** Each instance is excluded from its own neighbourhood by index; exact duplicates of other rows are neighbours (a twin with another label is real overlap), and zero-length MST edges are valid.
4. **Determinism.** `random_state` (default 42, never `None`) seeds every draw through `numpy.random.SeedSequence` and is passed to mutual information, `LinearSVC` and PCA.

### Stage I — Difficulty Score

```
DS = 0.10·D1 + 0.22·D2 + 0.18·D3 + 0.15·D4 + 0.10·D5 + 0.12·D6 + 0.13·D7
```

Weight order reflects empirical evidence from the data complexity literature: class overlap (D2) and instance hardness (D3) are stronger difficulty drivers than imbalance ratio alone (D1). Weights are a principled default; see the paper for calibration rationale.

| Band | DS range | Interpretation |
|------|----------|----------------|
| Low | [0.00, 0.25) | Unlikely to be significantly challenging |
| Moderate | [0.25, 0.50) | Standard imbalanced techniques likely sufficient |
| High | [0.50, 0.75) | Specialized strategies and careful evaluation needed |
| Extreme | [0.75, 1.00] | Fundamental learning challenges; combine multiple strategies |

### Stage P — Complexity Signature

A dimension Dᵢ _dominates_ the profile when **Dᵢ > τ = 0.50 and Dᵢ = max{D1, D2, D4, D5}** (paper §3.3). D3, D6, and D7 activate recommendations (Stage A) when they exceed the threshold but never dominate.

| Sig | Name | Trigger |
|-----|------|---------|
| I | Imbalance-dominated | D1 dominates |
| II | Overlap-dominated | D2 dominates |
| III | Fragmented | D4 dominates |
| IV | Dimensionality-dominated | D5 dominates (more effective dimensions than minority instances) |
| V | Compound | No dimension dominates |

- **Ties** at the maximum resolve in the order D1 > D2 > D4 > D5. A value exactly equal to τ does not dominate.
- **Qualifier of Signature V** (informative, never changes the signature), counting all seven dimensions above τ′ = 0.35: `compound` (≥ 2), `single` (exactly 1), `low` (none). The paper leaves the "exactly one above τ′" case uncovered; this qualifier closes it and was **confirmed by the author on 2026-09-16**.
- τ and τ′ are parameters (`CIPAPipeline(tau=..., tau_prime=...)`).

### Stage A — Action recommendations

Preprocessing and model families are **signature-specific**; evaluation metrics and validation rules are **threshold-triggered** per dimension.

> The Action protocol is unchanged in 2.0.0 and still uses the thresholds below, which differ from the paper's τ = 0.50. Because it depends on the signature, its output can change with the new signature rule. It is out of scope for this release.

**Signature-based recommendations:**

| Sig | Primary preprocessing | Primary model families |
|-----|-----------------------|------------------------|
| I | Random Oversampling or SMOTE | Logistic Regression, Linear SVM |
| II | Borderline-SMOTE or ADASYN | RBF-SVM, Kernel methods |
| III | DBSMOTE or MWMOTE (cluster-aware) | Random Forest, Gradient Boosting |
| IV | Dimensionality reduction (PCA/UMAP) _before_ resampling, then SMOTE | Linear models after dim. reduction, Kernel SVM |
| V | SMOTE+ENN (combined over+under) | Gradient Boosting (XGBoost/LightGBM), Balanced Random Forest |

**Conditional additions** (activated when a dimension exceeds its threshold):

- **D2 ≥ 0.55** → add AUC-ROC to metrics
- **D3 ≥ 0.55** → add Recall (minority) to metrics; add outlier removal before oversampling
- **D4 ≥ 0.55** → validate that each CV fold contains all sub-concepts
- **D5 ≥ 0.70** → mandatory dimensionality reduction before any resampling (with the 2.0.0rc2 D5 this means r₉₅ ≥ 2.33·|C₊|)
- **DS ≥ 0.50** → add G-mean to metrics
- **DS ≥ 0.75** → add MCC; use repeated stratified k-fold (5×10); add cost-sensitive learning

**Validation protocol** (always applied): stratified k-fold (k=5 or k=10) with confidence intervals over folds; bootstrap for N < 500; k=10 minimum when IR > 20.

---

## API reference

### `CIPADataset`

```python
CIPADataset(
    X: np.ndarray,           # feature matrix, shape (N, d)
    y: np.ndarray,           # labels, shape (N,)
    minority_label: int,     # label of the class of interest C+ (taken as declared)
    majority_label: int,     # label of C-
    name: str | None = None,
)
```

Key properties: `.N`, `.d`, `.IR`, `.n_minority`, `.n_majority`, `.X_minority`, `.X_majority`, `.minority_is_majority`.

If the declared minority has more instances than the declared majority, the constructor emits a `UserWarning`, keeps the labels as declared and the pipeline records `minority_is_majority=True` in the result metadata. `CIPADataset.from_arrays(X, y)`, which picks the minority by frequency, is **deprecated** (`DeprecationWarning`) and never used by the pipeline.

### `CIPAPipeline`

```python
CIPAPipeline(
    weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13),  # D1–D7 weights, must sum to 1
    *,
    random_state=42,          # int, never None; seeds every random source
    scaling="standard",       # "standard" | "robust" | "none"
    n_max=50_000,             # rows per computation for D2/D7 (minority rows for D4)
    n_subsamples=5,           # draws when a dimension exceeds n_max
    n_jobs=None,              # parallel neighbour search, DBSCAN, MI (sklearn >= 1.5)
    k_neighbors=5,            # k for D2 (kDN), D3 (NS-typology)
    dbscan_min_samples=3,     # min_samples for D4 DBSCAN
    dbscan_eps=None,          # eps for D4 (None = adaptive)
    d2_weights=(1/3,1/3,1/3), # (alpha, beta, gamma) for D2 sub-components
    svc_max_iter=10_000,      # LinearSVC iterations for L1
    tau=0.50,                 # dominance threshold (signatures)
    tau_prime=0.35,           # Signature V qualifier threshold
    chunk_size=65_536,        # neighbour queries per block
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.run(dataset)` | `CIPAResult` | Full C → I → P → A pipeline |
| `.run_scoring_only(dataset)` | `CIPAResult` with `action=None` | Stages C + I + P |
| `.run_dimensions_only(dataset)` | `tuple[DimensionResult, ...]` | Only Stage C (D1–D7) |

The 1.x parameters `knn_subsample`, `n1_max_exact` and `large_n_subsample` were removed.

### `CIPAResult`

```python
result.dataset_name            # str | None
result.difficulty_score        # DifficultyScore
result.profile                 # ComplexityProfile
result.action                  # ActionRecommendation | None (None from run_scoring_only)
result.metadata                # dict — run record, see below
result.to_dict()               # JSON-serializable dict
```

`metadata` holds `cipa_version`, `random_state`, `scaling`, `n_max`, `n_subsamples`, `N`, `d`, `n_minority`, `n_majority`, `IR`, `minority_label`, `majority_label`, `minority_is_majority`, `preprocessing` (dropped constant columns and, for `"robust"`, the σ-fallback columns), `l1_fits`, `l1_not_converged` and `time_seconds` (preprocessing, each dimension, total).

### `DifficultyScore`

```python
ds = result.difficulty_score
ds.value                       # float in [0, 1], from the dimension medians
ds.band                        # "Low" | "Moderate" | "High" | "Extreme"
ds.weights                     # tuple of 7 floats
ds.dimensions                  # tuple of 7 DimensionResult
ds.contributions               # dict {dimension_id: w_i * D_i}
ds.to_dict()
```

### `DimensionResult`

```python
dim = ds.dimensions[1]         # D2
dim.value                      # median over draws (or the single value)
dim.iqr                        # IQR over draws (0.0 without subsampling)
dim.components                 # e.g. {"F3", "N1", "kDN", ...}: medians over draws
                               # D5: {"r_95", "n_minority", "rho", "H_nats", "H_max_nats",
                               #      "spectral_entropy_norm", "n_components_fit"}
dim.metadata["n_used"]         # rows per computation
dim.metadata["n_subsamples"]   # computations made (1 without subsampling)
dim.metadata["seeds"]          # derived seed of each draw ([] without subsampling)
dim.metadata["values"]         # value of each draw
dim.metadata["time_seconds"]
# with draws also: "components_iqr", "components_per_subsample", "run_metadata"
```

D7's components include `converged` (True only if every `LinearSVC` fit converged); L1 is always the training error obtained, never a fixed fallback.

### `ComplexityProfile`

```python
profile = result.profile
profile.vector                 # tuple of 7 floats — (D1, ..., D7)
profile.signature              # "I" | "II" | "III" | "IV" | "V"
profile.signature_name         # e.g. "Overlap-dominated"
profile.dominant_dimension     # "D1" | "D2" | "D4" | "D5" | None (Signature V)
profile.qualifier              # "compound" | "single" | "low" for V, else None
profile.active_dimensions      # dimensions > tau, sorted descending
profile.elevated_dimensions    # dimensions > tau_prime, sorted descending
profile.tau, profile.tau_prime
profile.to_dict()
```

### `ActionRecommendation`

```python
action = result.action
action.evaluation_metrics      # list[str] — ordered by priority
action.preprocessing_strategy  # list[str] — ordered by priority
action.model_families          # list[str] — ordered by priority
action.validation_protocol     # list[str]
action.rationale               # dict {item: reason} — why each recommendation was activated
action.warnings                # list[str] — edge-case flags (small N, extreme IR, etc.)
action.to_dict()
```

### Individual dimensions

```python
from cipa.dimensions import compute_d1, compute_d2, compute_d3, compute_d4
from cipa.dimensions import compute_d5, compute_d6, compute_d7

result_d1 = compute_d1(dataset)
print(result_d1.value, result_d1.components)
```

Standalone dimension functions compute on the rows they receive, without preprocessing or subsampling; `CIPAPipeline` applies both.

---

## Custom weights

```python
# Put all weight on class overlap and instance hardness
pipeline = CIPAPipeline(weights=(0.05, 0.40, 0.35, 0.05, 0.05, 0.05, 0.05))
result = pipeline.run(dataset)
```

---

## Serialization

All result objects are JSON-serializable:

```python
import json

result_dict = result.to_dict()
print(json.dumps(result_dict, indent=2))
```

---

## Performance

- No computation builds an N×N matrix. N1 uses an exact Euclidean minimum spanning tree (Borůvka over precomputed neighbour lists) with O(N·k) memory; on N = 50,000 random points it takes about 0.3 s with d = 3, 8 s with d = 10 and 100 s with d = 70 on one core, almost all of it in the initial k-NN query.
- The cost of D2/D7 is bounded by `n_max` × `n_subsamples`. D1, D5, D6 and D3 run on all N rows and dominate for very large datasets; use `n_jobs=-1`.
- Timings per dimension are recorded in `result.metadata["time_seconds"]`.

---

## Reproducing COMIA 2026 values

The values published in COMIA 2026 come from cipa v1.1.0–v1.2.1. Install the `v1.2.1` tag to reproduce them:

```bash
pip install --no-deps "cipa @ git+https://github.com/luisgarciarodriguez-research/cipa@v1.2.1"
```

For datasets processed without subsampling in the paper, `CIPAPipeline(scaling="none")` in 2.0.0 returns the same D1–D4, D6 and D7 (those formulas are unchanged). D5 was redefined; the published D5 is the component `spectral_entropy_norm` of D5, and the published DS is recovered by using it in place of D5. The signature follows the paper's dominance rule and the new D5, so it can differ from the published one. The test `tests/regression/test_regression_v1_2_1.py` checks this on synthetic data.

---

## Development

```bash
pip install -e ".[dev]"
pytest                   # ≥90% coverage required
python -m ruff check src tests
```

---

## Citation

If you use CIPA in your research, please cite:

```bibtex
@inproceedings{garcia2026cipa,
  title     = {{CIPA}: A Multi-Domain Statistical Framework for Characterizing
               Imbalanced Datasets and Computing a Difficulty Score},
  author    = {Garc\'{i}a Rodr\'{i}guez, Luis and
               Neme Castillo, Jos\'{e} Antonio and
               G\'{o}mez Adorno, Helena Montserrat and
               Fuentes Pineda, Gibr\'{a}n},
  booktitle = {Proceedings of the XVIII Congreso Mexicano de Inteligencia
               Artificial (COMIA 2026)},
  year      = {2026},
  % doi     = {},  % update when assigned
}
```

---

## License

MIT © Luis García Rodríguez, José Antonio Neme Castillo, Helena Montserrat Gómez Adorno, Gibran Fuentes Pineda — IIMAS-UNAM
