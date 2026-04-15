# CIPA

**A Multi-Domain Statistical Framework for Characterizing Imbalanced Datasets and Computing a Difficulty Score**

CIPA (_Characterization, Indexing, Profiling, Action_) analyzes a binary classification dataset along seven complexity dimensions and produces a single Difficulty Score, a structural fingerprint, and actionable preprocessing recommendations — all without training any model.

---

## Installation

```bash
pip install -e .          # development install
pip install -e ".[dev]"   # with test/lint tools
```

**Requirements:** Python ≥ 3.11, NumPy ≥ 1.24, SciPy ≥ 1.10, scikit-learn ≥ 1.3.

---

## Quick start

```python
import numpy as np
from cipa import CIPADataset, CIPAPipeline

# 1. Wrap your data
X = np.load("features.npy")   # shape (N, d), numeric
y = np.load("labels.npy")     # 1-D integer array

dataset = CIPADataset(X, y, minority_label=1, majority_label=0, name="MyDataset")
# or auto-detect minority by frequency:
dataset = CIPADataset.from_arrays(X, y, name="MyDataset")

# 2. Run the pipeline
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
print("  Metrics     :", result.action.evaluation_metrics[:2])
print("  Preprocessing:", result.action.preprocessing_strategy[:1])
print("  Models      :", result.action.model_families[:2])
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

| ID | Name | What it measures |
|----|------|-----------------|
| D1 | Imbalance Distribution | How skewed the class ratio is (`1 − 1/log₂(IR+1)`) |
| D2 | Class Overlap | Fraction of instances in the overlap region (F3 + N1 + kDN) |
| D3 | Instance Hardness | Proportion of minority instances that are borderline, rare, or outliers (Napierała–Stefanowski typology) |
| D4 | Sub-concept Fragmentation | How fragmented the minority class is across DBSCAN clusters (ECindex) |
| D5 | Effective Dimensionality | Ratio of informative dimensions to total features (PCA-based) |
| D6 | Feature Informativeness | Fraction of features with low mutual information with the label |
| D7 | Boundary Complexity | Difficulty of the decision boundary (LinearSVC error + N2 ratio) |

All dimensions are clipped to [0, 1]. Higher values mean higher difficulty.

### Stage I — Difficulty Score

```
DS = 0.10·D1 + 0.22·D2 + 0.18·D3 + 0.15·D4 + 0.10·D5 + 0.12·D6 + 0.13·D7
```

Default weights reflect the empirical importance found in the paper (D2 and D3 dominate).

| Band | DS range | Meaning |
|------|----------|---------|
| Low | [0.00, 0.25) | Unlikely to be significantly challenging |
| Moderate | [0.25, 0.50) | Standard imbalanced techniques likely sufficient |
| High | [0.50, 0.75) | Specialized strategies and careful evaluation needed |
| Extreme | [0.75, 1.00] | Fundamental learning challenges; combine multiple strategies |

### Stage P — Complexity Signature

| Sig | Name | Trigger rule |
|-----|------|-------------|
| I | Imbalance-dominated | All of D2–D7 < 0.25 (uniformly low; mild imbalance is the only issue) |
| II | Overlap-dominated | D2 > 0.55 and D2 ≥ D1 |
| III | Fragmented | D4 is highest among D1–D5 and D4 > 0.50 |
| IV | Dimensionality-dominated | D5 is highest among D1–D5, D5 > 0.55, and D5 > D2 + 0.10 |
| V | Compound | Multiple elevated dimensions or no single dominant driver |

### Stage A — Action recommendations

Four axes are populated based on the signature, DS band, IR, and N:

- **Evaluation metrics** — primary: AUC-PR, F1; conditional: G-mean, MCC, AUC-ROC, Recall (minority)
- **Preprocessing strategy** — signature-specific oversampling or dimensionality reduction
- **Model families** — signature-specific classifiers, escalating to ensembles for compound datasets
- **Validation protocol** — stratified k-fold, repeated folds for extreme cases, bootstrap for small N

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
ds.value                       # float in [0, 1]
ds.band                        # "Low" | "Moderate" | "High" | "Extreme"
ds.weights                     # tuple of 7 floats
ds.dimensions                  # tuple of 7 DimensionResult
ds.contributions               # dict {dimension_id: w*D_i}
ds.to_dict()
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
@article{garcia2026cipa,
  title   = {{CIPA}: A Multi-Domain Statistical Framework for Characterizing
             Imbalanced Datasets and Computing a Difficulty Score},
  author  = {Garc\'{i}a Rodr\'{i}guez, Luis and
             Neme Castillo, Jos\'{e} Antonio and
             G\'{o}mez Adorno, Helena Montserrat},
  journal = {TODO: venue},
  year    = {2026},
  doi     = {TODO: doi},
}
```

---

## License

MIT © Luis García Rodríguez, IIMAS-UNAM
