# Adding a New Dataset to CIPA

This guide explains how to integrate a new dataset so that it can be analysed
by the CIPA pipeline and included in the validation script
(`experiments/validate_table2.py`).

All four steps below are confined to that single file plus a new subdirectory
under `datasets/`. No changes to `src/cipa/` are needed.

---

## Prerequisites

- CIPA installed in development mode: `pip install -e ".[dev]"`
- The dataset available locally in a directory you control.

---

## Step 1 — Place the raw files

Create a subdirectory under `datasets/` following the existing naming
convention:

```
datasets/
└── NN-Domain-DatasetName/
    └── datafile.csv          ← raw data file(s)
```

`NN` is the next sequential number (currently the last is `13`). Examples:

| Number | Pattern |
|--------|---------|
| 01–03 | `01-Financial-CreditCardFraudDetection` |
| 04–06 | `04-Medical-BreastCancer-Wisconsin` |
| 07–08 | `07-Cybersecurity-KDD` |

Raw data files are excluded from the repository by `.gitignore`. Document the
download source in [`datasets/README.md`](../datasets/README.md) following the
same table format used for the existing datasets.

---

## Step 2 — Write a loader function

Open `experiments/validate_table2.py` and add a loader function after the
existing ones (before the `LOADERS` registry). The function must:

- Return `(X, y)` as a tuple of `np.ndarray`.
- `X` — shape `(N, d)`, dtype `float64`, **no NaN or Inf**.
- `y` — shape `(N,)`, exactly **two unique values**, minority class = `1`.
- Name the function `load_<dataset_key_lowercase>`.

### Minimal example (CSV)

```python
def load_mi_dataset() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    path = DATASETS_ROOT / "14-Domain-MyDataset" / "data.csv"
    df = pd.read_csv(path)
    y = (df["label"] == "minority_class_name").astype(int).values
    X = df.drop(columns=["label"]).values.astype(float)
    return X, y
```

### Large-N datasets (N > 10,000)

Apply the asymmetric subsampler to keep **all** minority samples and
random-subsample the majority down to `n_target`:

```python
def load_mi_dataset() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    path = DATASETS_ROOT / "14-Domain-MyDataset" / "data.csv"
    df = pd.read_csv(path)
    y = (df["label"] == "minority_class").astype(int).values
    X = df.drop(columns=["label"]).values.astype(float)
    # N = 500k → reduce to 10,000 preserving all minority samples
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)
```

### Common preprocessing patterns

| Situation | Pattern |
|-----------|---------|
| Missing values | `df[col].fillna(df[col].median())` |
| Categorical features | `pd.Categorical(df[col]).codes.astype(float)` |
| Columns with >50% missing | drop: `df.loc[:, df.isnull().sum() <= 0.5 * len(df)]` |
| `±Inf` values | `df.replace([np.inf, -np.inf], np.nan)` then impute |
| LibSVM sparse format | `sklearn.datasets.load_svmlight_file` → `.toarray()` |
| MATLAB `.mat` files | `scipy.io.loadmat` |
| Auto-detect minority | use `CIPADataset.from_arrays` (auto-detects by frequency) |

### Validation rules enforced by `CIPADataset`

The pipeline raises `ValueError` if any of these are violated:

| Rule | Minimum |
|------|---------|
| `X` must be 2-dimensional | — |
| No NaN or Inf in `X` | — |
| `y` must have exactly 2 unique values | — |
| Minority class size | n_minority ≥ 2 |
| Total samples | N ≥ 10 |

---

## Step 3 — Register in `LOADERS` and `TIER`

### `LOADERS` registry

Add an entry mapping the dataset key to `(loader_function, subdirectory_name)`:

```python
LOADERS: dict[str, tuple[callable, str]] = {
    # ... existing entries ...
    "MyDataset": (load_mi_dataset, "14-Domain-MyDataset"),  # ← add here
}
```

The key (`"MyDataset"`) is what you pass to `--datasets` on the command line.

### `TIER` registry

```python
TIER: dict[str, int] = {
    # ... existing entries ...
    "MyDataset": 1,   # ← add here
}
```

Choose the tier based on whether subsampling was applied:

| Tier | When to use | Validation criteria |
|------|-------------|---------------------|
| **1** | Full-N dataset (no `_asymmetric_subsample` call) | DS ± 0.10, band match, signature match |
| **2** | Dataset subsampled to N=10,000 | Band match + signature match only |

Tier 2 is used when the full dataset is too large for exact numeric
reproduction: the asymmetric subsampling changes the effective imbalance ratio
(IR_eff < IR_real), which shifts D1 and DS relative to full-N values.

---

## Step 4 — Add ground-truth values (optional)

If you have reference values to compare against (e.g., from a prior analysis),
add an entry to `TABLE_2`:

```python
TABLE_2: dict[str, dict] = {
    # ... existing entries ...
    "MyDataset": {
        "D1": 0.35, "D2": 0.48, "D3": 0.41,
        "D4": 0.29, "D5": 0.22, "D6": 0.31, "D7": 0.44,
        "DS": 0.37, "band": "M", "sig": "III",
    },
}
```

Valid band codes: `"L"` (Low), `"M"` (Moderate), `"H"` (High), `"E"` (Extreme).  
Valid signature codes: `"I"`, `"II"`, `"III"`, `"IV"`, `"V"`.

If `TABLE_2` has no entry for the dataset, the script runs the pipeline and
prints results but skips the comparison step without raising an error.

---

## Running the new dataset

```bash
# Single dataset
python experiments/validate_table2.py --datasets MyDataset

# With a custom data directory
python experiments/validate_table2.py --datasets MyDataset --data-dir /path/to/datasets

# With debug logging
python experiments/validate_table2.py --datasets MyDataset --verbose

# Save results to JSON
python experiments/validate_table2.py --datasets MyDataset --output my_results.json
```

Expected output (Tier 1, no TABLE_2 entry):

```
[MyDataset] loading... N=1,200, d=12 | running pipeline... done (3.2s)
```

If a `TABLE_2` entry was added, the pass/fail columns will appear in the
summary table.

---

## Checklist

- [ ] Raw files placed in `datasets/NN-Domain-DatasetName/`
- [ ] Download source documented in `datasets/README.md`
- [ ] `load_<name>()` function written and returns `(X, y)` with minority = 1
- [ ] Large-N: `_asymmetric_subsample` applied, Tier set to 2
- [ ] Entry added to `LOADERS`
- [ ] Entry added to `TIER`
- [ ] (Optional) Entry added to `TABLE_2` with reference values
- [ ] `python experiments/validate_table2.py --datasets MyDataset` runs without error

---

## Using the pipeline directly (without the validation script)

For ad-hoc analysis without modifying `validate_table2.py`, use the public API
directly:

```python
import numpy as np
from cipa import CIPADataset, CIPAPipeline

# Load your data however you like
X = np.load("my_features.npy")
y = np.load("my_labels.npy")

# Build the CIPA dataset (minority class auto-detected by frequency)
dataset = CIPADataset.from_arrays(X, y, name="MyDataset")

# Run the full pipeline
pipeline = CIPAPipeline(random_state=42)
result = pipeline.run(dataset)

print(result.difficulty_score.value)   # DS ∈ [0, 1]
print(result.difficulty_score.band)    # "Low" / "Moderate" / "High" / "Extreme"
print(result.profile.signature)        # "I" – "V"

# Per-dimension values
for dim in result.difficulty_score.dimensions:
    print(f"  {dim.dimension_id}: {dim.value:.4f}")
```

If both classes have equal frequency, `from_arrays` raises `ValueError`; in
that case pass `minority_label` explicitly to `CIPADataset(...)`.

---

## Subsampling reference

`_asymmetric_subsample` lives in `validate_table2.py` (not in `src/cipa/`).
The pipeline's own `knn_subsample` parameter handles subsampling internally for
large-N k-NN operations and is independent of this function.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_target` | `10_000` | Total samples after subsampling |
| `minority_label` | `1` | Label of the class to preserve entirely |
| `random_state` | `0` | Reproducibility seed |

If `len(y) <= n_target`, the function returns `(X, y)` unchanged.
