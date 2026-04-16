# Datasets

This directory holds the raw data files used to validate CIPA against the 13
benchmark datasets described in Table 2 of the paper. **The data files are not
included in this repository** (see `.gitignore`). Download each dataset from
its original source and place the required file(s) in the corresponding
subdirectory.

All experiments were run with `experiments/validate_table2.py`. See
`experiments/README.md` for reproduction instructions.

---

## Directory layout

```
datasets/
├── 01-Financial-CreditCardFraudDetection/
├── 02-Financial-PaySim/
├── 03-Financial-IEEE_CIS_FD/
├── 04-Medical-BreastCancer-Wisconsin/
├── 05-Medical-Diabetes/
├── 06-Medical-SVMGuide1/
├── 07-Cybersecurity-KDD/
├── 08-Cybersecurity-CIC_IDS/
├── 09-Industrial-CWRU/
├── 10-Industrial-SEU_Gearbox/
├── 11-Bioinformatics-TCGA_BRCA/
├── 12-BioInformatics-YEAST/
└── 13-Bioinformatics-Ecoli/
```

---

## Dataset details

### 01 · Credit Card Fraud Detection

| Field | Value |
|-------|-------|
| IR (full) | ~577:1 |
| N | 284,807 |
| Features | 30 (PCA-anonymised) |
| Source | Kaggle — ULB Machine Learning Group |
| URL | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| License | Open Database License (ODbL) |

**Required file:** `creditcard.csv`

> Note: experiments use a stratified subsample of N=10,000.

---

### 02 · PaySim Mobile Money Fraud

| Field | Value |
|-------|-------|
| IR (full) | ~745:1 |
| N | 6,362,620 |
| Features | 9 |
| Source | Kaggle — Edgar Lopez-Rojas |
| URL | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| License | CC BY-SA 4.0 |

**Required file:** `PS_20174392719_1491204439457_log.csv`

> Note: experiments use a stratified subsample of N=10,000.
> The directory also contains `EMSS2016_249.pdf` (the original paper describing
> the simulator); this file is excluded from the repository by `.gitignore`.

---

### 03 · IEEE-CIS Fraud Detection

| Field | Value |
|-------|-------|
| IR (full) | ~28:1 |
| N | 590,540 |
| Features | 394 |
| Source | Kaggle — IEEE Computational Intelligence Society |
| URL | https://www.kaggle.com/competitions/ieee-fraud-detection/data |
| License | Competition rules — research use only |

**Required file:** `train_transaction.csv`

> Note: experiments use a stratified subsample of N=10,000.

---

### 04 · Breast Cancer Wisconsin (Diagnostic)

| Field | Value |
|-------|-------|
| IR | ~1.7:1 |
| N | 569 |
| Features | 30 |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| License | CC BY 4.0 |

**Required files:** `wdbc.data`, `wdbc.names`

---

### 05 · PIMA Indians Diabetes

| Field | Value |
|-------|-------|
| IR | ~1.9:1 |
| N | 768 |
| Features | 8 |
| Source | Kaggle — UCI ML (original: National Institute of Diabetes) |
| URL | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| License | CC0: Public Domain |

**Required file:** `diabetes.csv`

---

### 06 · SVMGuide1

| Field | Value |
|-------|-------|
| IR | ~1.6:1 |
| N | 3,089 |
| Features | 4 |
| Source | LibSVM datasets (Chang & Lin, 2011) |
| URL | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html |
| License | Publicly available for research |

**Required file:** `svmguide1.txt` (LibSVM sparse format)

---

### 07 · NSL-KDD

| Field | Value |
|-------|-------|
| IR | ~variable by attack type |
| N | 125,973 (KDDTrain+) |
| Features | 41 |
| Source | Canadian Institute for Cybersecurity |
| URL | https://www.unb.ca/cic/datasets/nsl.html |
| License | Publicly available for research |

**Required file:** `KDDTrain+.txt`

> Note: experiments use a stratified subsample of N=10,000.

---

### 08 · CIC-IDS 2017

| Field | Value |
|-------|-------|
| IR | ~variable by traffic type |
| N | ~2,800,000 (all days combined) |
| Features | 78 |
| Source | Canadian Institute for Cybersecurity |
| URL | https://www.unb.ca/cic/datasets/ids-2017.html |
| License | Publicly available for research |

**Required files:** all `*.pcap_ISCX.csv` day files (Monday through Friday).

Place all CSV files directly in `08-Cybersecurity-CIC_IDS/`.

> Note: experiments use a stratified subsample of N=10,000.

---

### 09 · CWRU Bearing Fault

| Field | Value |
|-------|-------|
| IR | ~variable by fault configuration |
| N | ~48,000 samples per condition |
| Features | time-domain / frequency-domain (preprocessed) |
| Source | Case Western Reserve University Bearing Data Center |
| URL | https://engineering.case.edu/bearingdatacenter |
| License | Publicly available for research |

**Required files:** `CWRU_48k_load_1_CNN_data.npz`,
`feature_time_48k_2048_load_1.csv`

---

### 10 · SEU Gearbox

| Field | Value |
|-------|-------|
| IR | ~variable by fault type |
| N | ~variable |
| Features | vibration signals (raw `.mat`) |
| Source | Southeast University, School of Mechanical Engineering |
| URL | https://github.com/cathysiyu/Mechanical-datasets |
| License | Publicly available for research |

**Required files:** all `.mat` files (normal and fault conditions).

Place all `.mat` files directly in `10-Industrial-SEU_Gearbox/`.

---

### 11 · TCGA-BRCA

| Field | Value |
|-------|-------|
| IR | ~variable by molecular subtype |
| N | ~1,100 |
| Features | ~20,000 (RNA-seq gene expression) |
| Source | The Cancer Genome Atlas via Broad GDAC Firehose |
| URL | https://gdac.broadinstitute.org/ |
| License | NIH Genomic Data Sharing Policy — open-access tier |

**Required files:**
- `Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi.txt`
- `Human__TCGA_BRCA__UNC__RNAseq__GA_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct`
- `Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct`

Download from the Firehose BRCA cohort (stddata run 2016-01-28).

---

### 12 · Yeast (ME3 class)

| Field | Value |
|-------|-------|
| IR | ~28:1 (ME3 vs rest) |
| N | 1,484 |
| Features | 8 |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/110/yeast |
| License | CC BY 4.0 |

**Required files:** `yeast.data`, `yeast.names`

> The binary task targets class ME3 (minority) vs. all other classes (majority).

---

### 13 · Ecoli (imU class)

| Field | Value |
|-------|-------|
| IR | ~8.6:1 (imU vs rest) |
| N | 336 |
| Features | 7 |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/39/ecoli |
| License | CC BY 4.0 |

**Required files:** `ecoli.data`, `ecoli.names`

> The binary task targets class imU (minority) vs. all other classes (majority).
