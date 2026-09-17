"""Empirical validation of CIPA against Table 2 of the paper.

Runs CIPAPipeline on all 13 benchmark datasets and compares
dimension scores (D1-D7), DS, band, and complexity signature
against the ground-truth values reported in Table 2.

Requires cipa 1.x (tag v1.2.1)
------------------------------
This script reproduces the values published in COMIA 2026. It relies on the
1.x API and behaviour: ``CIPADataset.from_arrays``, no feature scaling, the
1.x signature rule and the asymmetric 10k subsamples below. cipa 2.0.0
changes those values by design (see CHANGELOG.md), so the script stops if it
imports cipa >= 2. Run it against the ``v1.2.1`` tag, e.g. from a worktree::

    git worktree add ../cipa-v1.2.1 v1.2.1
    cd ../cipa-v1.2.1
    python -m venv .venv && .venv/bin/pip install -e ".[experiments]"
    .venv/bin/python experiments/validate_table2.py --data-dir <this repo>/datasets

Usage
-----
    cd <project_root>
    python experiments/validate_table2.py

    # Point to a custom datasets directory:
    python experiments/validate_table2.py --data-dir /path/to/datasets

    # Or set the environment variable (overridden by --data-dir):
    CIPA_DATA_DIR=/path/to/datasets python experiments/validate_table2.py

    # Run a single dataset:
    python experiments/validate_table2.py --datasets CreditCard

    # Save results to JSON:
    python experiments/validate_table2.py --output results.json

Data directory
--------------
The script looks for datasets in the following order:
  1. --data-dir command-line argument
  2. CIPA_DATA_DIR environment variable
  3. <project_root>/datasets/  (default, relative to this file)

See datasets/README.md for download instructions for each dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Official v1.1.0 results — source: experiments/VALIDATION_REPORT.md §2
# These values are reproduced in Table 2 of the paper.
# Columns: D1-D7, DS, band (E/H/M/L), sig (I-V)
# Tier-2 datasets (marked * in VALIDATION_REPORT) use N_eff=10k subsample.
# ---------------------------------------------------------------------------
TABLE_2: dict[str, dict] = {
    "CreditCard":   {"D1":.7170,"D2":.2997,"D3":.2039,"D4":.5337,"D5":.0038,"D6":.6819,"D7":.1394,"DS":.3547,"band":"M","sig":"V"},
    "PaySim":       {"D1":.3228,"D2":.3565,"D3":.1114,"D4":.4398,"D5":.5328,"D6":.7368,"D7":.0905,"DS":.3502,"band":"M","sig":"V"},
    "IEEE-CIS":     {"D1":.7811,"D2":.3770,"D3":.9219,"D4":.6989,"D5":.0000,"D6":.9642,"D7":.1562,"DS":.5679,"band":"H","sig":"V"},
    "Yeast-ME3":    {"D1":.5006,"D2":.2138,"D3":.2025,"D4":.2714,"D5":.8581,"D6":.9199,"D7":.1994,"DS":.3964,"band":"M","sig":"IV"},
    "CIC-IDS":      {"D1":.2841,"D2":.2351,"D3":.0504,"D4":.4648,"D5":.2694,"D6":.7447,"D7":.1023,"DS":.2885,"band":"M","sig":"V"},
    "TCGA-BRCA":    {"D1":.1746,"D2":.2130,"D3":.0654,"D4":.2421,"D5":.7326,"D6":.8792,"D7":.2265,"DS":.3206,"band":"M","sig":"IV"},
    "CWRU":         {"D1":.5310,"D2":.0078,"D3":.0319,"D4":.4446,"D5":.0520,"D6":.3670,"D7":.0172,"DS":.1787,"band":"L","sig":"V"},
    "PIMA":         {"D1":.0669,"D2":.5992,"D3":.2848,"D4":.3817,"D5":.2336,"D6":.9346,"D7":.3449,"DS":.4274,"band":"M","sig":"II"},
    "SEU-Gearbox":  {"D1":.5310,"D2":.0213,"D3":.0000,"D4":.2424,"D5":.4037,"D6":.4666,"D7":.1077,"DS":.2045,"band":"L","sig":"V"},
    "Ecoli-iMU":    {"D1":.5179,"D2":.2345,"D3":.2286,"D4":.2617,"D5":.6974,"D6":.8192,"D7":.1739,"DS":.3744,"band":"M","sig":"IV"},
    "NSL-KDD":      {"D1":.0035,"D2":.3412,"D3":.0117,"D4":.6158,"D5":.1291,"D6":.7875,"D7":.2241,"DS":.3064,"band":"M","sig":"III"},
    "SVMGUIDE1":    {"D1":.0637,"D2":.2739,"D3":.0300,"D4":.5702,"D5":.3527,"D6":.5362,"D7":.0709,"DS":.2664,"band":"M","sig":"III"},
    "BreastCancer": {"D1":.0474,"D2":.2357,"D3":.0818,"D4":.4940,"D5":.0284,"D6":.6773,"D7":.0875,"DS":.2409,"band":"L","sig":"V"},
}

BAND_MAP = {"E": "Extreme", "H": "High", "M": "Moderate", "L": "Low"}
DIM_TOL  = 0.10   # ±0.10 regression tolerance per dimension
DS_TOL   = 0.10   # ±0.10 regression tolerance on overall DS

# ---------------------------------------------------------------------------
# Tier classification
# Tier 1: full-N datasets (N ≤ ~10k, no subsampling distortion)
#          → exact numeric reproduction expected (DS ± DS_TOL, band, sig)
# Tier 2: large-N datasets subsampled to N=10k using asymmetric strategy
#          → qualitative reproduction expected (band + sig match only)
# ---------------------------------------------------------------------------
TIER: dict[str, int] = {
    "CreditCard":   2,   # N=284k,  IR=577  → asymmetric subsample
    "PaySim":       2,   # N=6.3M,  IR=772.7 → asymmetric subsample
    "IEEE-CIS":     2,   # N=590k,  IR=29   → asymmetric subsample
    "NSL-KDD":      2,   # N=125k,  IR=1.8  → asymmetric subsample
    "CIC-IDS":      2,   # N=2.8M,  IR=~5   → asymmetric subsample
    "BreastCancer": 1,   # N=569,   full-N
    "PIMA":         1,   # N=768,   full-N
    "SVMGUIDE1":    1,   # N=3089,  full-N
    "CWRU":         1,   # N=~1750, full-N
    "SEU-Gearbox":  1,   # N=10000, engineered (controlled extraction)
    "TCGA-BRCA":    1,   # N=826,   full-N
    "Yeast-ME3":    1,   # N=1484,  full-N
    "Ecoli-iMU":    1,   # N=336,   full-N
}

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_ROOT: Path = Path(os.environ["CIPA_DATA_DIR"]) if "CIPA_DATA_DIR" in os.environ else _DEFAULT_DATA_DIR

# ---------------------------------------------------------------------------
# Loaders  — each returns (X: np.ndarray, y: np.ndarray)
#            X shape (N, d), y shape (N,), minority label = 1
# ---------------------------------------------------------------------------

def _asymmetric_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_target: int = 10_000,
    minority_label: int = 1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dataset to n_target preserving ALL minority samples.

    Strategy (Tier-2 datasets with extreme IR):
    - Keep every minority-class sample (they are few and precious for D2/D3/D4).
    - Fill remaining budget (n_target - n_minority) by random-sampling the majority.
    - If n_minority > n_target, fall back to proportional stratified sampling.
    """
    if len(y) <= n_target:
        return X, y
    rng = np.random.default_rng(random_state)
    min_idx = np.where(y == minority_label)[0]
    maj_idx = np.where(y != minority_label)[0]
    n_min   = len(min_idx)

    if n_min >= n_target:
        # Pathological: more minority than budget — proportional fallback
        n_keep_min = round(n_target * n_min / len(y))
        n_keep_maj = n_target - n_keep_min
        sel_min = rng.choice(min_idx, size=n_keep_min, replace=False)
        sel_maj = rng.choice(maj_idx, size=n_keep_maj, replace=False)
    else:
        # Normal case: keep all minority, subsample majority
        n_keep_maj = n_target - n_min
        sel_min = min_idx
        sel_maj = rng.choice(maj_idx, size=min(n_keep_maj, len(maj_idx)), replace=False)

    sel = np.concatenate([sel_min, sel_maj])
    rng.shuffle(sel)
    return X[sel], y[sel]


def load_creditcard() -> tuple[np.ndarray, np.ndarray]:
    """Load and asymmetrically subsample CreditCard fraud detection (Kaggle/ULB, N=284k, IR=577)."""
    import pandas as pd
    path = DATASETS_ROOT / "01-Financial-CreditCardFraudDetection" / "creditcard.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["Time", "Class"]).values.astype(float)
    y = df["Class"].values.astype(int)
    # Full N=284k, IR=577 → keep all 492 minority + sample 9508 majority
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)


def load_paysim() -> tuple[np.ndarray, np.ndarray]:
    """Load and asymmetrically subsample PaySim synthetic financial transactions (N=6.3M, IR=772.7)."""
    import pandas as pd
    path = DATASETS_ROOT / "02-Financial-PaySim" / "PS_20174392719_1491204439457_log.csv"
    df = pd.read_csv(path)
    # Drop string ID columns and leakage flag; encode transaction type
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    df["type"] = pd.Categorical(df["type"]).codes
    X = df.drop(columns=["isFraud"]).values.astype(float)
    y = df["isFraud"].values.astype(int)
    # Full N=6.3M, IR=772.7 → keep all fraud samples + sample majority
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)


def load_ieee_cis() -> tuple[np.ndarray, np.ndarray]:
    """Load, preprocess, and subsample IEEE-CIS Fraud (N=590k): drop sparse cols, label-encode, median-impute."""
    import pandas as pd
    path = DATASETS_ROOT / "03-Financial-IEEE_CIS_FD" / "train_transaction.csv"
    df = pd.read_csv(path)
    y = df["is_fraud"].values.astype(int)
    df = df.drop(columns=["TransactionID", "is_fraud"])
    # Drop columns with >50% missing
    thresh = 0.50 * len(df)
    df = df.loc[:, df.isnull().sum() <= thresh]
    # Encode object columns with label encoding
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes.astype(float)
        df[col] = df[col].replace(-1, np.nan)
    # Median imputation for remaining NaN
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    X = df.values.astype(float)
    # Full N=590k, IR=29 → keep all fraud + sample legitimate
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Load Breast Cancer Wisconsin (Diagnostic) from UCI WDBC format; M=minority."""
    import pandas as pd
    path = DATASETS_ROOT / "04-Medical-BreastCancer-Wisconsin" / "wdbc.data"
    df = pd.read_csv(path, header=None)
    # col0=ID (drop), col1=label (M=minority=1, B=0), cols2-31=features
    y = (df.iloc[:, 1] == "M").astype(int).values
    X = df.iloc[:, 2:].values.astype(float)
    return X, y


def load_pima() -> tuple[np.ndarray, np.ndarray]:
    """Load the PIMA Indians Diabetes dataset from CSV (UCI/KEEL)."""
    import pandas as pd
    path = DATASETS_ROOT / "05-Medical-Diabetes" / "diabetes.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"]).values.astype(float)
    y = df["Outcome"].values.astype(int)
    return X, y


def load_svmguide1() -> tuple[np.ndarray, np.ndarray]:
    """Load SVMGuide1 from libSVM format; minority class auto-detected by frequency."""
    from sklearn.datasets import load_svmlight_file
    path = DATASETS_ROOT / "06-Medical-SVMGuide1" / "svmguide1.txt"
    X, y = load_svmlight_file(str(path))
    X = X.toarray().astype(float)
    # Labels are 1 and 2; make minority=1 (smaller class)
    classes, counts = np.unique(y, return_counts=True)
    minority_label = classes[np.argmin(counts)]
    y = (y == minority_label).astype(int)
    return X, y


def load_nsl_kdd() -> tuple[np.ndarray, np.ndarray]:
    """Load and subsample NSL-KDD (KDDTrain+); encode categorical cols; binary label: attack=1."""
    import pandas as pd
    path = DATASETS_ROOT / "07-Cybersecurity-KDD" / "KDDTrain+.txt"
    df = pd.read_csv(path, header=None)
    # Col 41 = label (normal/attack name), col 42 = difficulty score (drop)
    label_col = df.iloc[:, 41]
    df = df.drop(columns=[41, 42])
    # Encode 3 categorical columns (1=protocol, 2=service, 3=flag)
    for col in [1, 2, 3]:
        df[col] = pd.Categorical(df[col]).codes.astype(float)
    X = df.values.astype(float)
    # Binary: normal=0, any attack=1 (minority: attack is actually majority here,
    # but paper defines minority by frequency — normal is majority)
    y = (label_col != "normal").astype(int).values
    # N=125k, IR~1.8 → keep all attack samples + subsample normal
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)


def load_cic_ids() -> tuple[np.ndarray, np.ndarray]:
    """Concatenate, clean (inf/NaN), and subsample CIC-IDS-2017 CSV shards; attack=1."""
    import pandas as pd
    pattern = DATASETS_ROOT / "08-Cybersecurity-CIC_IDS"
    csvs = sorted(pattern.glob("*.csv"))
    dfs = []
    for f in csvs:
        try:
            chunk = pd.read_csv(f, encoding="utf-8", low_memory=False)
            chunk.columns = chunk.columns.str.strip()
            dfs.append(chunk)
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True)
    label_col = "Label"
    y = (df[label_col].str.strip() != "BENIGN").astype(int).values
    df = df.drop(columns=[label_col])
    # Drop non-numeric and infinite values
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    X = df.values.astype(float)
    # N=2.8M, IR~5 → keep all attack + subsample benign
    return _asymmetric_subsample(X, y, n_target=10_000, minority_label=1, random_state=0)


def load_cwru() -> tuple[np.ndarray, np.ndarray]:
    """Load CWRU Bearing fault diagnosis dataset; normal=majority, any fault=minority."""
    import pandas as pd
    path = DATASETS_ROOT / "09-Industrial-CWRU" / "feature_time_48k_2048_load_1.csv"
    df = pd.read_csv(path)
    fault_col = df.columns[-1]
    y = (~df[fault_col].str.contains("Normal", na=False)).astype(int).values
    X = df.drop(columns=[fault_col]).values.astype(float)
    return X, y


def load_seu_gearbox() -> tuple[np.ndarray, np.ndarray]:
    """Load SEU Gearbox via FFT feature extraction.

    Strategy (matches paper: N=10,000, d=128, IR=9:1):
    - Window size = 256 samples → 128 FFT magnitude bins (d=128)
    - 250 non-overlapping windows per .mat file (40 files × 250 = 10,000)
    - Drive End (DE) accelerometer signal
    - Minority (label=1) = normal operation (4 files)
    - Majority (label=0) = all fault conditions (36 files)
    """
    import scipy.io as sio

    path = DATASETS_ROOT / "10-Industrial-SEU_Gearbox"
    mat_files = sorted(path.glob("*.mat"))

    WINDOW_SIZE = 256       # FFT of 256 → 128 real-valued features
    N_PER_FILE  = 250       # 40 files × 250 = 10,000 total
    rng = np.random.default_rng(42)

    X_parts, y_parts = [], []
    for mat_file in mat_files:
        class_name = mat_file.stem.rsplit("_", 1)[0]
        is_normal = class_name == "normal"

        data = sio.loadmat(str(mat_file))
        # Drive End accelerometer signal
        sig_key = next(k for k in data if not k.startswith("_") and "DE" in k)
        signal = data[sig_key].ravel().astype(float)

        # Non-overlapping windows
        n_windows = len(signal) // WINDOW_SIZE
        segments = signal[: n_windows * WINDOW_SIZE].reshape(n_windows, WINDOW_SIZE)

        # FFT magnitude spectrum → 128 features per window
        fft_mag = np.abs(np.fft.rfft(segments, n=WINDOW_SIZE))[:, :WINDOW_SIZE // 2]

        # Subsample to N_PER_FILE windows
        n_take = min(N_PER_FILE, len(fft_mag))
        idx = rng.choice(len(fft_mag), size=n_take, replace=False)
        X_parts.append(fft_mag[idx])
        # normal=minority=1, fault=majority=0
        y_parts.append(np.ones(n_take, dtype=int) if is_normal else np.zeros(n_take, dtype=int))

    return np.vstack(X_parts), np.concatenate(y_parts)


def load_tcga_brca() -> tuple[np.ndarray, np.ndarray]:
    """Load TCGA-BRCA expression data with PAM50 binary labels.

    Features : top 2,000 genes by variance across HiSeq samples (d=2,000)
    Labels   : LumA + LumB = majority (0), Basal + Her2 = minority (1)
    """
    import pandas as pd

    base = DATASETS_ROOT / "11-Bioinformatics-TCGA_BRCA"
    expr_file = base / "Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct"
    clin_file = base / "Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi.txt"

    # Expression matrix: rows=genes, cols=samples → transpose to samples×genes
    expr = pd.read_csv(expr_file, sep="\t", index_col=0).T  # shape (N_samples, N_genes)

    # Clinical: parse PAM50 row
    clin = pd.read_csv(clin_file, sep="\t", index_col=0).T  # shape (N_samples, N_attrs)
    pam50 = clin["PAM50"].dropna()
    pam50 = pam50[pam50 != "NA"]

    # Keep only samples with expression + known PAM50
    common = expr.index.intersection(pam50.index)
    expr_sub = expr.loc[common]
    labels = pam50.loc[common]

    # Binary: minority=1 = Basal + Her2 (aggressive), majority=0 = LumA + LumB
    keep = labels.isin(["LumA", "LumB", "Basal", "Her2"])
    expr_sub = expr_sub.loc[keep]
    labels = labels.loc[keep]
    y = labels.isin(["Basal", "Her2"]).astype(int).values

    # Top 2,000 genes by variance
    variances = expr_sub.var(axis=0)
    top_genes = variances.nlargest(2000).index
    X = expr_sub[top_genes].values.astype(float)

    return X, y


def load_yeast_me3() -> tuple[np.ndarray, np.ndarray]:
    """Load Yeast ME3 imbalanced dataset from UCI/KEEL whitespace-separated format."""
    import pandas as pd
    path = DATASETS_ROOT / "12-BioInformatics-YEAST" / "yeast.data"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # col0=sequence name (drop), cols1-8=features, col9=class
    y = (df.iloc[:, -1] == "ME3").astype(int).values
    X = df.iloc[:, 1:-1].values.astype(float)
    return X, y


def load_ecoli_imu() -> tuple[np.ndarray, np.ndarray]:
    """Load Ecoli iMU imbalanced dataset from UCI/KEEL whitespace-separated format."""
    import pandas as pd
    path = DATASETS_ROOT / "13-Bioinformatics-Ecoli" / "ecoli.data"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # col0=sequence name (drop), cols1-7=features, col8=class
    y = (df.iloc[:, -1] == "imU").astype(int).values
    X = df.iloc[:, 1:-1].values.astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
LOADERS: dict[str, tuple[callable, str]] = {
    "CreditCard":   (load_creditcard,  "01-Financial-CreditCardFraudDetection"),
    "PaySim":       (load_paysim,      "02-Financial-PaySim"),
    "IEEE-CIS":     (load_ieee_cis,    "03-Financial-IEEE_CIS_FD"),
    "BreastCancer": (load_breast_cancer,"04-Medical-BreastCancer-Wisconsin"),
    "PIMA":         (load_pima,        "05-Medical-Diabetes"),
    "SVMGUIDE1":    (load_svmguide1,   "06-Medical-SVMGuide1"),
    "NSL-KDD":      (load_nsl_kdd,     "07-Cybersecurity-KDD"),
    "CIC-IDS":      (load_cic_ids,     "08-Cybersecurity-CIC_IDS"),
    "CWRU":         (load_cwru,        "09-Industrial-CWRU"),
    "SEU-Gearbox":  (load_seu_gearbox, "10-Industrial-SEU_Gearbox"),
    "TCGA-BRCA":    (load_tcga_brca,   "11-Bioinformatics-TCGA_BRCA"),
    "Yeast-ME3":    (load_yeast_me3,   "12-BioInformatics-YEAST"),
    "Ecoli-iMU":    (load_ecoli_imu,   "13-Bioinformatics-Ecoli"),
}

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _band_letter(band: str) -> str:
    """Return the single-letter abbreviation for a difficulty band name (E/H/M/L)."""
    return {"Extreme": "E", "High": "H", "Moderate": "M", "Low": "L"}[band]


def compare(name: str, result, elapsed: float) -> dict:
    """Compare pipeline output for one dataset against expected Table 2 values; return diff record."""
    expected = TABLE_2[name]
    ds = result.difficulty_score
    dim_values = {d.dimension_id: d.value for d in ds.dimensions}

    dim_results = {}
    n_dim_ok = 0
    for i in range(1, 8):
        did = f"D{i}"
        actual = dim_values[did]
        exp    = expected[did]
        diff   = actual - exp
        ok     = abs(diff) <= DIM_TOL
        dim_results[did] = {"actual": round(actual, 4), "expected": exp, "diff": round(diff, 4), "ok": ok}
        if ok:
            n_dim_ok += 1

    ds_actual   = round(ds.value, 4)
    ds_expected = expected["DS"]
    ds_ok       = abs(ds_actual - ds_expected) <= DS_TOL

    band_actual   = _band_letter(ds.band)
    band_expected = expected["band"]
    band_ok       = band_actual == band_expected

    sig_actual   = result.profile.signature
    sig_expected = expected["sig"]
    sig_ok       = sig_actual == sig_expected

    tier = TIER.get(name, 1)
    if tier == 1:
        overall_pass = ds_ok and band_ok and sig_ok
    else:
        # Tier 2: qualitative only
        overall_pass = band_ok and sig_ok

    return {
        "dataset":       name,
        "tier":          tier,
        "N":             sum(d.value for d in ds.dimensions),   # placeholder; overwritten in main
        "elapsed_s":     round(elapsed, 1),
        "dimensions":    dim_results,
        "DS":            {"actual": ds_actual,   "expected": ds_expected, "ok": ds_ok},
        "band":          {"actual": band_actual, "expected": band_expected, "ok": band_ok},
        "signature":     {"actual": sig_actual,  "expected": sig_expected, "ok": sig_ok},
        "dims_within_tol": n_dim_ok,
        "overall_pass":  overall_pass,
    }


def _print_section(
    records: list[dict],
    tier: int,
    header: str,
    pass_label: str,
    pass_criteria_fn,
) -> int:
    """Print one tier section. Returns number of passes."""
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    W = 135

    subset = [r for r in records if TIER.get(r["dataset"], 1) == tier]
    if not subset:
        return 0

    print()
    print(f"{'─' * W}")
    print(f"  {header}")
    print(f"{'─' * W}")
    print(f"  {'Dataset':<16} {'N':>7} {'time':>6}  "
          f"{'D1':>7} {'D2':>7} {'D3':>7} {'D4':>7} {'D5':>7} {'D6':>7} {'D7':>7}  "
          f"{'DS':>7}  {'Band':>4}  {'Sig':>3}  {pass_label}")
    print(f"  {'-' * (W-2)}")

    n_pass = 0
    for r in subset:
        passed = pass_criteria_fn(r)
        mark = PASS if passed else FAIL
        print(f"  {r['dataset']:<16} {r['N']:>7,} {r['elapsed_s']:>5.1f}s  "
              f"{r['dimensions']['D1']['actual']:.4f} "
              f"{r['dimensions']['D2']['actual']:.4f} "
              f"{r['dimensions']['D3']['actual']:.4f} "
              f"{r['dimensions']['D4']['actual']:.4f} "
              f"{r['dimensions']['D5']['actual']:.4f} "
              f"{r['dimensions']['D6']['actual']:.4f} "
              f"{r['dimensions']['D7']['actual']:.4f}  "
              f"{r['DS']['actual']:.4f}  "
              f"{r['band']['actual']:>4}  "
              f"{r['signature']['actual']:>3}  "
              f"{mark}")
        if passed:
            n_pass += 1

    print(f"  {'-' * (W-2)}")
    print(f"  {'Expected (Table 2)':<16} {'':>7} {'':>6}  "
          f"{'D1':>7} {'D2':>7} {'D3':>7} {'D4':>7} {'D5':>7} {'D6':>7} {'D7':>7}  "
          f"{'DS':>7}  {'Band':>4}  {'Sig':>3}")
    for r in subset:
        name = r["dataset"]
        if name not in TABLE_2:
            continue
        t = TABLE_2[name]
        print(f"    {name:<14} {'':>7} {'':>6}  "
              f"{t['D1']:.4f} {t['D2']:.4f} {t['D3']:.4f} {t['D4']:.4f} "
              f"{t['D5']:.4f} {t['D6']:.4f} {t['D7']:.4f}  "
              f"{t['DS']:.4f}  {BAND_MAP[t['band']]:>4}  {t['sig']:>3}")

    print(f"\n  → {n_pass}/{len(subset)} pass")
    return n_pass


def print_report(records: list[dict]) -> None:
    """Print the full tier-grouped validation report with per-dataset diffs and pass/fail summary."""
    W = 135
    print()
    print("=" * W)
    print("  CIPA — Empirical Validation against Table 2")
    print("=" * W)

    # Tier 1: exact numeric reproduction (DS ± 0.10, band, sig)
    def tier1_pass(r):
        """Return True if result passes Tier 1: DS ±tol, band match, and signature match."""
        return r["DS"]["ok"] and r["band"]["ok"] and r["signature"]["ok"]

    n1 = _print_section(
        records, tier=1,
        header="TIER 1  — Full-N datasets  (exact numeric reproduction expected: DS ± 0.10, band match, sig match)",
        pass_label="Pass",
        pass_criteria_fn=tier1_pass,
    )

    # Tier 2: qualitative reproduction (band + sig); DS numeric not expected due to subsampling
    def tier2_pass(r):
        """Return True if result passes Tier 2: band and signature match (DS numeric not required)."""
        return r["band"]["ok"] and r["signature"]["ok"]

    n2 = _print_section(
        records, tier=2,
        header=(
            "TIER 2  — Large-N subsampled (asymmetric: all minority + majority sampled to N=10k)\n"
            "          Qualitative reproduction expected: band match + sig match\n"
            "          Note: D2/D3 may still differ from paper due to extreme IR (e.g. CreditCard IR=577)"
        ),
        pass_label="Band+Sig",
        pass_criteria_fn=tier2_pass,
    )

    n_total   = len(records)
    n_pass    = n1 + n2
    t1_count  = sum(1 for r in records if TIER.get(r["dataset"], 1) == 1)
    t2_count  = sum(1 for r in records if TIER.get(r["dataset"], 1) == 2)

    print()
    print("=" * W)
    print(f"  SUMMARY")
    print(f"  Tier 1 (full-N, exact):      {n1}/{t1_count}  (DS ± {DS_TOL}, band, sig)")
    print(f"  Tier 2 (subsampled, qualit.): {n2}/{t2_count}  (band + sig match)")
    print(f"  Overall:                      {n_pass}/{n_total}")
    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run CIPA validation over selected benchmark datasets."""
    parser = argparse.ArgumentParser(
        description="Validate CIPA against Table 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(LOADERS),
        choices=list(LOADERS),
        help="Datasets to run (default: all)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help=(
            "Root directory containing the dataset subdirectories. "
            "Overrides the CIPA_DATA_DIR environment variable. "
            f"Default: {_DEFAULT_DATA_DIR}"
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Resolve data directory: CLI > env var > default (already set at module level)
    global DATASETS_ROOT
    if args.data_dir is not None:
        DATASETS_ROOT = args.data_dir.expanduser().resolve()
    if not DATASETS_ROOT.is_dir():
        parser.error(
            f"Data directory not found: {DATASETS_ROOT}\n"
            "Use --data-dir or set CIPA_DATA_DIR. "
            "See datasets/README.md for download instructions."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore")

    import cipa
    if int(cipa.__version__.split(".")[0]) >= 2:
        parser.error(
            f"cipa {cipa.__version__} imported from {cipa.__file__}.\n"
            "This script reproduces the COMIA 2026 values and requires cipa 1.x: "
            "run it against the v1.2.1 tag (see the module docstring)."
        )

    from cipa import CIPADataset, CIPAPipeline

    pipeline = CIPAPipeline(random_state=42)
    records = []

    for name in args.datasets:
        loader_fn, dir_name = LOADERS[name]
        print(f"[{name}] loading...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            X, y = loader_fn()
            print(f"N={len(y):,}, d={X.shape[1]}", end=" | running pipeline...", flush=True)
            dataset = CIPADataset.from_arrays(X, y, name=name)
            result  = pipeline.run(dataset)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")
            rec = compare(name, result, elapsed)
            rec["N"] = int(len(y))
            records.append(rec)
        except Exception as exc:
            print(f"ERROR: {exc}")
            records.append({"dataset": name, "error": str(exc), "overall_pass": False})

    print_report([r for r in records if "error" not in r])

    failed_load = [r for r in records if "error" in r]
    if failed_load:
        print("Load/runtime errors:")
        for r in failed_load:
            print(f"  {r['dataset']}: {r['error']}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(records, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
