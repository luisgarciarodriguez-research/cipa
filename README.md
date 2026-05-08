# CIPA

**A Multi-Domain Statistical Framework for Characterizing Imbalanced Datasets and Computing a Difficulty Score**

CIPA (_Characterization, Indexing, Profiling, Action_) analyzes a binary classification dataset along seven non-redundant complexity dimensions and produces a single **Difficulty Score** (DS ∈ [0, 1]), a structural fingerprint, and actionable preprocessing recommendations — all without training any model.

---

## Why CIPA?

Preprocessing decisions (resampling strategy, evaluation metric, model family, validation protocol) must be made _before_ any model is trained, yet are typically based on the imbalance ratio (IR) alone. IR is a poor predictor of classification difficulty: class overlap, borderline instances, minority fragmentation, effective dimensionality, and feature informativeness all shape learning difficulty independently of IR and must be quantified before intervening.

CIPA formalizes this characterization step as a four-stage pipeline that measures seven structural properties, aggregates them into a normalized index, and maps the resulting profile to concrete, dimension-specific recommendations.

**Empirical validation across 13 benchmark datasets from 5 domains (Finance, Medical, Cybersecurity, Industry, Bioinformatics):**

| Predictor | Spearman ρ vs AUC-PR | Wilcoxon p vs DS |
|-----------|---------------------|-----------------|
| **CIPA DS** | **−0.71** | — |
| IR alone (B1) | −0.60 | 0.040 |
| F3 alone (B2) | −0.28 | 0.001 |
| ECoL distance (B3) | −0.55 | 0.0002 |

DS significantly outperforms all three baselines (pairwise Wilcoxon, α = 0.05).

---

## Applicability

| Requirement | Details |
|---|---|
| **Task** | Binary classification only |
| **Feature matrix X** | Numeric (`float`), shape (N, d) — encode categoricals before passing to CIPA |
| **Label vector y** | 1-D integer array with exactly two distinct values |
| **Class structure** | One minority class (less frequent); CIPA auto-detects it or accepts explicit labels |

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
# 1.2.0
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

# Auto-detect minority class by frequency:
dataset = CIPADataset.from_arrays(X, y, name="MyDataset")
# Or specify labels explicitly:
# dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name="MyDataset")

# 2. Run the full pipeline
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

**Example output** (PIMA Diabetes, N=768, IR=1.9:1):

```
Difficulty Score : 0.427  (Moderate)
Signature        : II — Overlap-dominated

Per-dimension breakdown:
  D1  0.067  (contributes 0.007)
  D2  0.599  (contributes 0.132)
  D3  0.285  (contributes 0.051)
  D4  0.382  (contributes 0.057)
  D5  0.234  (contributes 0.023)
  D6  0.935  (contributes 0.112)
  D7  0.345  (contributes 0.045)

Top recommendations:
  Metrics      : ['AUC-PR', 'F1-score (minority class)']
  Preprocessing: ['Borderline-SMOTE or ADASYN']
  Models       : ['RBF-SVM', 'Kernel methods']
  Validation   : ['Stratified k-fold cross-validation (k=5 or k=10)']
```

### From a pandas DataFrame

```python
import pandas as pd
from cipa import CIPADataset, CIPAPipeline

df = pd.read_csv("data.csv")
X = df.drop(columns=["label"]).values.astype(float)
y = df["label"].values.astype(int)

dataset = CIPADataset.from_arrays(X, y, name="MyDataset")
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
| D2 | Class Overlap | `(F3 + N1 + kDN) / 3` | Geometric inseparability: F3 (Fisher discriminant ratio), N1 (MST boundary fraction), kDN (minority k-NN majority contamination) |
| D3 | Instance Hardness | Napierała–Stefanowski typology | Proportion of minority instances classified as borderline, rare, or outlier, weighted by severity |
| D4 | Sub-concept Fragmentation | `ECindex × (n_clusters / \|C₊\|)^0.5` | How fragmented the minority concept is across DBSCAN sub-regions (ECindex = Error Concentration) |
| D5 | Effective Dimensionality | `H(PCA eigenvalues) / ln(k)` | Spectral entropy of the covariance matrix: 0 = one dominant component, 1 = uniform variance spread |
| D6 | Feature Informativeness | `1 − Ī(X;Y) / H(Y)` | Mutual information deficit between features and the label (KSG estimator): 0 = fully discriminative, 1 = uninformative |
| D7 | Boundary Complexity | `(LinearSVC error + N2_norm) / 2` | Linear separability error combined with neighbourhood non-linearity (N2 ratio) |

where `H(Y) = −p⁺ log₂ p⁺ − p⁻ log₂ p⁻` is the binary entropy of the label; `p±= |C±| / N`.

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

A dimension Dᵢ _dominates_ the profile when it exceeds the activation threshold (0.50) and is the largest among {D1, D2, D4, D5}. D3, D6, and D7 activate recommendations (Stage A) when they exceed the threshold but do not generate a named signature.

| Sig | Name | Trigger |
|-----|------|---------|
| I | Imbalance-dominated | All of D2–D7 < 0.25 — imbalance is the only active dimension |
| II | Overlap-dominated | D2 > 0.55 and D2 ≥ D1 |
| III | Fragmented | D4 dominates {D1, D2, D4, D5} and D4 > 0.50 |
| IV | Dimensionality-dominated | D5 dominates {D1, D2, D4, D5}, D5 > 0.55, and D5 > D2 + 0.10 |
| V | Compound | Multiple elevated dimensions or no single dominant driver |

### Stage A — Action recommendations

Preprocessing and model families are **signature-specific**; evaluation metrics and validation rules are **threshold-triggered** per dimension.

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
- **D5 ≥ 0.70** → mandatory dimensionality reduction before any resampling
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
    minority_label: int,     # label of the minority class
    majority_label: int,     # label of the majority class
    name: str | None = None,
)

CIPADataset.from_arrays(X, y, name=None)   # auto-detect minority by frequency
```

Key properties: `.N`, `.d`, `.IR`, `.n_minority`, `.n_majority`, `.X_minority`, `.X_majority`.

### `CIPAPipeline`

```python
CIPAPipeline(
    weights=(0.10, 0.22, 0.18, 0.15, 0.10, 0.12, 0.13),  # D1–D7 weights, must sum to 1
    k_neighbors=5,            # k for D2 (kDN), D3 (NS-typology)
    dbscan_min_samples=3,     # min_samples for D4 DBSCAN
    dbscan_eps=None,          # eps for D4 (None = adaptive)
    n1_max_exact=50_000,      # max N for exact N1 MST computation
    large_n_subsample=10_000, # subsample size for large-N paths
    random_state=None,        # reproducibility seed
    d2_weights=(1/3,1/3,1/3), # (alpha, beta, gamma) for D2 sub-components
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.run(dataset)` | `CIPAResult` | Full C → I → P → A pipeline |
| `.run_dimensions_only(dataset)` | `tuple[DimensionResult, ...]` | Only Stage C (D1–D7) |
| `.run_scoring_only(dataset)` | `DifficultyScore` | Stages C + I, skip P and A |

### `CIPAResult`

```python
result.dataset_name            # str | None
result.difficulty_score        # DifficultyScore
result.profile                 # ComplexityProfile
result.action                  # ActionRecommendation
result.to_dict()               # JSON-serializable dict
```

### `DifficultyScore`

```python
ds = result.difficulty_score
ds.value                       # float in [0, 1]
ds.band                        # "Low" | "Moderate" | "High" | "Extreme"
ds.weights                     # tuple of 7 floats
ds.dimensions                  # tuple of 7 DimensionResult
ds.contributions               # dict {dimension_id: w_i * D_i}
ds.to_dict()
```

### `ComplexityProfile`

```python
profile = result.profile
profile.vector                 # tuple of 7 floats — (D1, ..., D7)
profile.signature              # "I" | "II" | "III" | "IV" | "V"
profile.signature_name         # e.g. "Overlap-dominated"
profile.dominant_dimensions    # list of dimension IDs with value ≥ 0.55, sorted descending
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

Single-core runtimes on typical hardware:

| Dataset size | Max features | Expected runtime |
|-------------|-------------|-----------------|
| N ≤ 10,000 | any | ≤ 10 s |
| N ≤ 100,000 | d ≤ 100 | ≤ 60 s |
| N ≤ 1,000,000 | d ≤ 50 | ≤ 300 s |

For large N, the pipeline automatically applies stratified subsampling in the N1 (MST), N2, and DBSCAN computations.

---

## Development

```bash
pip install -e ".[dev]"
pytest                   # 212 tests, ≥90% coverage required
python -m ruff check src/
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
