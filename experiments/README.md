# Experiments

This directory contains the empirical validation of CIPA against the 13
benchmark datasets reported in Table 2 of the paper, plus the corresponding
result files.

```
experiments/
├── validate_table2.py    — main validation script
├── VALIDATION_REPORT.md  — full numeric results and analysis (v1.1.0)
└── README.md             — this file
```

Result JSON files (`results*.json`) are excluded from the repository via
`.gitignore`; they are regenerated locally by running the script.

---

## Prerequisites

**1. Install CIPA and its dependencies**

```bash
pip install -e ".[dev]"
```

**2. Download the datasets**

The 13 benchmark datasets are not included in the repository. Follow the
download instructions in [`datasets/README.md`](../datasets/README.md) and
place each dataset in its corresponding subdirectory.

---

## Reproducing Table 2

### All 13 datasets (default)

```bash
cd <project_root>
python experiments/validate_table2.py
```

### Pointing to a custom data directory

If your datasets live outside the project tree, pass `--data-dir` or set the
`CIPA_DATA_DIR` environment variable:

```bash
# via argument
python experiments/validate_table2.py --data-dir /path/to/datasets

# via environment variable
export CIPA_DATA_DIR=/path/to/datasets
python experiments/validate_table2.py
```

The resolution order is: `--data-dir` > `CIPA_DATA_DIR` > `<project_root>/datasets/`.

### Single dataset

```bash
python experiments/validate_table2.py --datasets BreastCancer
python experiments/validate_table2.py --datasets CreditCard PaySim IEEE-CIS
```

Available dataset keys:

| Key | Description |
|-----|-------------|
| `BreastCancer` | Breast Cancer Wisconsin (Diagnostic) |
| `PIMA` | PIMA Indians Diabetes |
| `SVMGUIDE1` | SVMGuide1 (LibSVM) |
| `CWRU` | CWRU Bearing Fault |
| `SEU-Gearbox` | SEU Gearbox Fault |
| `TCGA-BRCA` | TCGA Breast Cancer (RNA-seq) |
| `Yeast-ME3` | Yeast ME3 class (UCI) |
| `Ecoli-iMU` | Ecoli imU class (UCI) |
| `CreditCard` | Credit Card Fraud Detection *(Tier 2)* |
| `PaySim` | PaySim Mobile Money Fraud *(Tier 2)* |
| `IEEE-CIS` | IEEE-CIS Fraud Detection *(Tier 2)* |
| `NSL-KDD` | NSL-KDD Intrusion Detection *(Tier 2)* |
| `CIC-IDS` | CIC-IDS 2017 Network Traffic *(Tier 2)* |

### Saving results to JSON

```bash
python experiments/validate_table2.py --output my_results.json
```

### Verbose logging (debug)

```bash
python experiments/validate_table2.py --verbose --datasets BreastCancer
```

---

## Validation criteria

The script uses a two-tier validation scheme:

**Tier 1 — Full-N datasets** (N ≤ 10,000; no subsampling distortion)

Exact numeric reproduction is expected:
- Each dimension Dᵢ within ±0.10 of Table 2.
- DS within ±0.10 of Table 2.
- Band (Low / Moderate / High / Extreme) must match exactly.
- Complexity Signature (I–V) must match exactly.

**Tier 2 — Large-N datasets** (N > 10,000; asymmetric subsample to N=10,000)

Only qualitative reproduction is expected:
- Band must match exactly.
- Complexity Signature must match exactly.
- DS numeric value may differ from Table 2 due to asymmetric subsampling
  (all minority samples are kept; majority is subsampled), which changes the
  effective imbalance ratio relative to the full dataset.

---

## Official results (CIPA v1.1.0)

The canonical results are documented in
[`VALIDATION_REPORT.md`](VALIDATION_REPORT.md). The table below summarises
the computed values; see the report for full per-dimension breakdowns and
analysis of deviations from Table 2.

| Dataset | DS | Band | Sig |
|---------|----|------|-----|
| BreastCancer | 0.2409 | Low | V |
| SVMGUIDE1 | 0.2664 | Moderate | III |
| CWRU | 0.1787 | Low | V |
| SEU-Gearbox | 0.2045 | Low | V |
| TCGA-BRCA | 0.3206 | Moderate | IV |
| Yeast-ME3 | 0.3964 | Moderate | IV |
| Ecoli-iMU | 0.3744 | Moderate | IV |
| PIMA | 0.4274 | Moderate | II |
| CreditCard* | 0.3547 | Moderate | V |
| PaySim* | 0.3502 | Moderate | V |
| IEEE-CIS* | 0.5679 | High | V |
| NSL-KDD* | 0.3064 | Moderate | III |
| CIC-IDS* | 0.2885 | Moderate | V |

\* Tier 2: N=10,000 asymmetric subsample; IR_eff < IR_real.

> **Note on Table 2 discrepancies.** The values in Table 2 of the paper were
> computed with preliminary formulas for D1 and D5. Version 1.1.0 uses the
> improved formulas (`D1 = 1 − H(Y)` and D5 based on PCA spectral entropy),
> which produce results that are more grounded theoretically but differ
> numerically from Table 2. The VALIDATION_REPORT.md documents both sets of
> values.
