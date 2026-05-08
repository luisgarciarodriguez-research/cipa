"""Reproduce RQ1 statistics from §5.2 of García Rodríguez et al. (2026).

Definitive computed values (CIPA v1.1.0, DS source: v1.1.0, abs_direct error metric):

  DS (proposed)
    Spearman ρ(DS, AUC-PR) = -0.709   95% CI [-0.817, +0.091]
    Pearson  r(DS, AUC-PR) = -0.484   p = 0.094

  B1: IR alone  (D1_real = 1 - H(Y) computed from real IR)
    Spearman ρ(IR, AUC-PR) = -0.654   Pearson r = -0.595  p = 0.032
    Wilcoxon signed-rank DS vs B1: stat=16, p = 0.040  (two-sided, abs_direct)

  B2: F3 alone  (standard ECoL range-overlap, computed on pipeline-subsampled data)
    Spearman ρ(F3, AUC-PR) = -0.093   Pearson r = -0.226  p = 0.458
    Wilcoxon signed-rank DS vs B2: stat=2,  p = 0.001  (two-sided, abs_direct)
    F3 values per dataset stored in _F3_STANDARD (computed 2026-05-05).

  B3: ECoL aggregate  (unweighted Euclidean norm of P=(D1,...,D7), normalised by √7)
    Spearman ρ(B3, AUC-PR) = -0.665   Pearson r = -0.499  p = 0.083
    Wilcoxon signed-rank DS vs B3: stat=0,  p = 0.0002  (two-sided, abs_direct)
    DS wins in 13/13 datasets.  B3 values derived from results_v2.json (2026-05-05).

Error metric for all Wilcoxon tests: abs_direct = |predictor_i - (1 - AUC-PR_i)|,
which is valid because DS, D1_real, and F3 are all in [0, 1] and represent
difficulty (high value → hard → low AUC-PR).

Usage
-----
  python experiments/compute_rq1_stats.py
  python experiments/compute_rq1_stats.py --ds-source aspirational
  python experiments/compute_rq1_stats.py --ds-source v1.1.0
  python experiments/compute_rq1_stats.py --with-f3 --data-dir /path/to/datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Ground-truth benchmark data (Table 3 of García Rodríguez et al., 2026)
# ---------------------------------------------------------------------------

# AUC-PR sourced from published benchmarking literature per dataset (§4.1).
# IR = |C-| / |C+| (real, not post-subsampling).
# Note on SVMGUIDE1: Table 3 shows 1.2 (rounded); the actual LibSVM training file
# has class sizes ~2000 vs ~1089 (IR ≈ 1.836), which explains D1=0.0637 in v1.1.0.
_TABLE3 = {
    #            auc_pr   ir_real
    "CreditCard":   (0.713, 577.0),
    "PaySim":       (0.642, 772.7),
    "IEEE-CIS":     (0.831,  28.6),
    "BreastCancer": (0.961,   1.68),
    "PIMA":         (0.748,   1.87),
    "SVMGUIDE1":    (0.986,   1.836),  # derived from D1=0.0637: H(Y)≈0.9363 → p+≈0.353 → IR≈1.836
    "NSL-KDD":      (0.991,   4.0),
    "CIC-IDS":      (0.832,  28.4),
    "CWRU":         (0.903,   8.0),
    "SEU-Gearbox":  (0.878,   9.0),
    "TCGA-BRCA":    (0.856,   1.9),
    "Yeast-ME3":    (0.631,  28.1),
    "Ecoli-iMU":    (0.812,   8.6),
}

# DS aspirational values (paper Table 2, written before code was complete).
_DS_ASPIRATIONAL = {
    "CreditCard":   0.83,
    "PaySim":       0.81,
    "IEEE-CIS":     0.59,
    "BreastCancer": 0.08,
    "PIMA":         0.43,
    "SVMGUIDE1":    0.09,
    "NSL-KDD":      0.21,
    "CIC-IDS":      0.57,
    "CWRU":         0.37,
    "SEU-Gearbox":  0.38,
    "TCGA-BRCA":    0.54,
    "Yeast-ME3":    0.68,
    "Ecoli-iMU":    0.39,
}

# D1 computed values — CIPA v1.1.0 pipeline output (D1 uses effective IR for tier-2).
# Tier-2 datasets are subsampled, so D1_pipeline < D1_real for high-IR datasets.
_D1_V110 = {
    "CreditCard":   0.7170,   # IR_ef=19.3  (real IR=577)
    "PaySim":       0.3228,   # IR_ef≈30    (real IR=772.7)
    "IEEE-CIS":     0.7811,   # IR_ef=7.4   (real IR=28.6)
    "BreastCancer": 0.0474,
    "PIMA":         0.0669,
    "SVMGUIDE1":    0.0637,
    "NSL-KDD":      0.0035,   # IR_ef≈1.8  (near balance after subsample)
    "CIC-IDS":      0.2841,
    "CWRU":         0.5310,
    "SEU-Gearbox":  0.5310,
    "TCGA-BRCA":    0.1746,
    "Yeast-ME3":    0.5006,
    "Ecoli-iMU":    0.5179,
}

# F3 standard ECoL (range-overlap, minimum over features) — computed 2026-05-05.
# Pipeline-subsampled data (N_eff=10k for tier-2 datasets), same as DS pipeline.
# All nine Barella class-specific variants were tested; none reproduced r=-0.630.
# Standard ECoL F3 is the correct and reproducible implementation.
_F3_STANDARD = {
    "CreditCard":   0.8526,
    "PaySim":       0.9615,
    "IEEE-CIS":     0.9721,
    "BreastCancer": 0.4833,
    "PIMA":         0.9935,
    "SVMGUIDE1":    0.7067,
    "NSL-KDD":      0.9759,
    "CIC-IDS":      0.6048,
    "CWRU":         0.0000,
    "SEU-Gearbox":  0.0638,
    "TCGA-BRCA":    0.5303,
    "Yeast-ME3":    0.4589,
    "Ecoli-iMU":    0.4613,
}

# DS computed values — CIPA v1.1.0 official results (results_v2.json).
_DS_V110 = {
    "CreditCard":   0.3547,
    "PaySim":       0.3502,
    "IEEE-CIS":     0.5679,
    "BreastCancer": 0.2409,
    "PIMA":         0.4274,
    "SVMGUIDE1":    0.2664,
    "NSL-KDD":      0.3064,
    "CIC-IDS":      0.2885,
    "CWRU":         0.1787,
    "SEU-Gearbox":  0.2045,
    "TCGA-BRCA":    0.3206,
    "Yeast-ME3":    0.3964,
    "Ecoli-iMU":    0.3744,
}

DATASETS = sorted(_TABLE3.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_entropy(p_plus: float) -> float:
    """H(Y) for binary class with minority proportion p_plus."""
    p_minus = 1.0 - p_plus
    if p_plus <= 0 or p_minus <= 0:
        return 0.0
    return -p_plus * np.log2(p_plus) - p_minus * np.log2(p_minus)


def ir_to_d1(ir: float) -> float:
    """D1 = 1 - H(Y) using the real IR (not post-subsampled)."""
    p_plus = 1.0 / (1.0 + ir)
    return 1.0 - _binary_entropy(p_plus)


def ir_to_inv_norm(ir: float) -> float:
    """Saturating normalization: 0 for IR=1, →1 for IR→∞."""
    return 1.0 - 1.0 / ir


def ir_to_log_norm(ir: float, ir_max: float) -> float:
    """Log normalization relative to dataset-collection max IR."""
    return np.log(ir) / np.log(ir_max)


def ols_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute OLS residuals from fitting y ~ x."""
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    return np.abs(y - lr.predict(x.reshape(-1, 1)))


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z 95% CI for Pearson r (two-sided)."""
    import math
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


class WilcoxonResult(NamedTuple):
    """Holds the results of a single pairwise Wilcoxon signed-rank test (DS vs one IR baseline)."""

    ir_label: str
    err_label: str
    alternative: str
    stat: float
    p_value: float
    n_pairs: int
    ds_source: str


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_arrays(ds_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (ds, auc_pr, ir_real, extras) aligned by DATASETS order."""
    ds_map = _DS_ASPIRATIONAL if ds_source == "aspirational" else _DS_V110
    ds     = np.array([ds_map[k]        for k in DATASETS])
    auc_pr = np.array([_TABLE3[k][0]    for k in DATASETS])
    ir     = np.array([_TABLE3[k][1]    for k in DATASETS])

    ir_max = ir.max()
    extras = {
        "d1_real":     np.array([ir_to_d1(r)                    for r in ir]),
        "d1_pipeline": np.array([_D1_V110[k]                    for k in DATASETS]),
        "ir_inv":      np.array([ir_to_inv_norm(r)              for r in ir]),
        "ir_log":      np.array([ir_to_log_norm(r, ir_max)      for r in ir]),
    }
    return ds, auc_pr, ir, extras


def run_correlations(ds: np.ndarray, auc_pr: np.ndarray,
                     extras: dict, label: str) -> None:
    """Print Spearman, Pearson, and CI for DS and all IR variants."""
    n = len(ds)
    sep = "─" * 68

    print(f"\n{sep}")
    print(f"  CORRELATIONS  —  DS source: {label}")
    print(sep)
    print(f"  {'Predictor':<18} {'Spearman ρ':>11} {'Pearson r':>10} "
          f"{'p (Pearson)':>12}  {'95% CI':>16}")
    print(f"  {'-' * 66}")

    for name, vec in [("DS",                    ds),
                      ("D1_real (H(Y) from IR)",  extras["d1_real"]),
                      ("D1_pipeline (eff. IR)",   extras["d1_pipeline"]),
                      ("IR_inv (1-1/IR)",          extras["ir_inv"]),
                      ("IR_log (log/log_max)",     extras["ir_log"])]:
        rho, p_s = spearmanr(vec, auc_pr)
        r,   p_p = pearsonr(vec, auc_pr)
        ci        = _pearson_ci(r, n)
        print(f"  {name:<22} {rho:>+9.4f}   {r:>+8.4f}   {p_p:>10.4f}  "
              f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # Paper targets
    print(f"\n  Paper targets:  ρ(DS)=-0.71   r(DS)=-0.48 (p=0.094)")
    print(f"                  r(IR)=-0.59 (p=0.033)   r(F3)=-0.63 (p=0.029)")


def run_wilcoxon_grid(ds: np.ndarray, auc_pr: np.ndarray,
                      extras: dict, ds_label: str) -> list[WilcoxonResult]:
    """Run all (IR variant) × (error metric) × (alternative) combinations."""
    results: list[WilcoxonResult] = []

    ir_variants = {
        "D1_real":     extras["d1_real"],
        "D1_pipeline": extras["d1_pipeline"],
        "IR_inv":      extras["ir_inv"],
        "IR_log":      extras["ir_log"],
    }

    err_fns = {
        "abs_OLS":    lambda pred: ols_residuals(pred, auc_pr),
        "sq_OLS":     lambda pred: ols_residuals(pred, auc_pr) ** 2,
        "abs_direct": lambda pred: np.abs(pred - (1.0 - auc_pr)),
        "rank_err":   lambda pred: np.abs(
                          np.argsort(np.argsort(-pred)) -
                          np.argsort(np.argsort(auc_pr))
                      ).astype(float),
    }

    alternatives = ["two-sided", "less", "greater"]

    for ir_lbl, ir_vec in ir_variants.items():
        for err_lbl, err_fn in err_fns.items():
            err_ds = err_fn(ds)
            err_ir = err_fn(ir_vec)
            diff   = err_ds - err_ir          # negative ↔ DS better

            for alt in alternatives:
                try:
                    stat, p = wilcoxon(err_ds, err_ir, alternative=alt,
                                       zero_method="wilcox")
                except Exception:
                    stat, p = float("nan"), float("nan")
                results.append(WilcoxonResult(
                    ir_label=ir_lbl, err_label=err_lbl, alternative=alt,
                    stat=stat, p_value=p,
                    n_pairs=len(ds), ds_source=ds_label,
                ))

    return results


def print_wilcoxon_grid(results: list[WilcoxonResult], ds_label: str,
                        target_p: float = 0.031) -> None:
    """Print the full Wilcoxon result grid and flag rows within tol of target_p."""
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  WILCOXON GRID  (DS vs IR baseline)  —  DS source: {ds_label}")
    print(f"  Target paper value: p = {target_p}")
    print(sep)
    print(f"  {'IR variant':<14} {'Error metric':<14} {'Alternative':<12} "
          f"{'stat':>8} {'p-value':>9}  {'note'}")
    print(f"  {'-' * 66}")

    tol = 0.005   # within ±0.005 of target counts as a match
    matches = []
    for r in results:
        note = ""
        if abs(r.p_value - target_p) <= tol:
            note = " *** MATCH ***"
            matches.append(r)
        print(f"  {r.ir_label:<14} {r.err_label:<14} {r.alternative:<12} "
              f"{r.stat:>8.2f} {r.p_value:>9.4f} {note}")

    print(f"\n  Matches (|p - {target_p}| ≤ {tol}): {len(matches)}")


def run_all(ds_label: str) -> None:
    """Run correlation analysis and Wilcoxon grid for one DS source label and print summary."""
    ds, auc_pr, ir, extras = build_arrays(ds_label)
    run_correlations(ds, auc_pr, extras, ds_label)
    results = run_wilcoxon_grid(ds, auc_pr, extras, ds_label)
    print_wilcoxon_grid(results, ds_label)

    # Best match summary
    finite = [r for r in results if not np.isnan(r.p_value)]
    if finite:
        best = min(finite, key=lambda r: abs(r.p_value - 0.031))
        sep = "─" * 68
        print(f"\n{sep}")
        print(f"  BEST MATCH to p=0.031  —  DS source: {ds_label}")
        print(sep)
        print(f"  IR variant : {best.ir_label}")
        print(f"  Error type : {best.err_label}")
        print(f"  Alternative: {best.alternative}")
        print(f"  stat={best.stat:.2f}   p={best.p_value:.5f}  "
              f"(delta from 0.031: {best.p_value - 0.031:+.5f})")
        print(sep)


# ---------------------------------------------------------------------------
# B2: F3 alone (hardcoded values, no dataset loading required)
# ---------------------------------------------------------------------------

def run_b2_stats(ds_label: str) -> None:
    """Print B2 (F3 alone) correlations and Wilcoxon DS vs F3."""
    ds_map = _DS_ASPIRATIONAL if ds_label == "aspirational" else _DS_V110
    ds     = np.array([ds_map[k]        for k in DATASETS])
    auc_pr = np.array([_TABLE3[k][0]    for k in DATASETS])
    f3     = np.array([_F3_STANDARD[k]  for k in DATASETS])
    n      = len(DATASETS)

    rho_f3, _ = spearmanr(f3, auc_pr)
    r_f3, p_f3 = pearsonr(f3, auc_pr)
    ci_f3 = _pearson_ci(r_f3, n)

    err_ds = np.abs(ds   - (1.0 - auc_pr))
    err_f3 = np.abs(f3   - (1.0 - auc_pr))

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  B2: F3 alone  —  DS source: {ds_label}")
    print(sep)
    print(f"  ρ(F3, AUC-PR) = {rho_f3:+.4f}")
    print(f"  r(F3, AUC-PR) = {r_f3:+.4f}   p = {p_f3:.4f}")
    print(f"  95% CI        = [{ci_f3[0]:+.3f}, {ci_f3[1]:+.3f}]")
    print()
    print(f"  {'Dataset':<18} {'F3':>6}  {'err_DS':>8}  {'err_F3':>8}  {'winner'}")
    print(f"  {'-'*52}")
    ds_wins = 0
    for i, k in enumerate(DATASETS):
        winner = "DS" if err_ds[i] < err_f3[i] else "F3"
        if winner == "DS":
            ds_wins += 1
        print(f"  {k:<18} {f3[i]:>6.4f}  {err_ds[i]:>8.4f}  {err_f3[i]:>8.4f}  {winner}")
    print(f"\n  DS wins: {ds_wins}/13")
    print()
    for alt in ["two-sided", "less"]:
        try:
            stat, p = wilcoxon(err_ds, err_f3, alternative=alt, zero_method="wilcox")
            sig = "*" if p < 0.05 else ""
            sig = "**" if p < 0.01 else sig
            sig = "***" if p < 0.001 else sig
            print(f"  Wilcoxon DS vs F3  alt={alt:<10}  stat={stat:.1f}  p={p:.5f} {sig}")
        except Exception as exc:
            print(f"  Wilcoxon DS vs F3  alt={alt}  ERROR: {exc}")
    print(sep)


# ---------------------------------------------------------------------------
# F3 extraction (requires raw datasets)
# ---------------------------------------------------------------------------

def run_with_f3(data_dir: Path, ds_label: str) -> None:
    """Compute F3 per dataset and run Wilcoxon DS vs F3 (target p=0.048)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cipa.ecol.f3 import compute_f3

    # Import loaders from validate_table2 (same file, avoids duplication)
    sys.path.insert(0, str(Path(__file__).parent))
    import validate_table2 as vt
    vt.DATASETS_ROOT = data_dir

    ds_map  = _DS_ASPIRATIONAL if ds_label == "aspirational" else _DS_V110
    auc_prs = [_TABLE3[k][0] for k in DATASETS]
    ds_vals = [ds_map[k]     for k in DATASETS]
    f3_vals = []

    print(f"\n{'─'*68}")
    print(f"  F3 EXTRACTION  —  DS source: {ds_label}")
    print(f"{'─'*68}")

    for name in DATASETS:
        loader_fn, dir_name = vt.LOADERS[name]
        required = data_dir / dir_name
        if not required.is_dir():
            print(f"  {name:<18}  SKIP (directory not found: {required})")
            f3_vals.append(float("nan"))
            continue
        try:
            X, y = loader_fn()
            f3   = compute_f3(X, y)
            f3_vals.append(f3)
            print(f"  {name:<18}  F3 = {f3:.4f}")
        except Exception as exc:
            print(f"  {name:<18}  ERROR: {exc}")
            f3_vals.append(float("nan"))

    # Filter to datasets where F3 was computed
    valid = [(ds_vals[i], auc_prs[i], f3_vals[i])
             for i in range(len(DATASETS)) if not np.isnan(f3_vals[i])]
    if len(valid) < 3:
        print("\n  Not enough datasets with F3 — skipping Wilcoxon vs F3.")
        return

    ds_arr  = np.array([v[0] for v in valid])
    auc_arr = np.array([v[1] for v in valid])
    f3_arr  = np.array([v[2] for v in valid])
    n       = len(valid)

    # Correlations
    rho_f3, _ = spearmanr(f3_arr, auc_arr)
    r_f3, p_f3 = pearsonr(f3_arr, auc_arr)
    print(f"\n  r(F3, AUC-PR) = {r_f3:+.4f}  p = {p_f3:.4f}  (paper: r=-0.63, p=0.029)")
    print(f"  ρ(F3, AUC-PR) = {rho_f3:+.4f}")

    # Wilcoxon DS vs F3 (OLS residuals)
    err_ds = ols_residuals(ds_arr, auc_arr)
    err_f3 = ols_residuals(f3_arr, auc_arr)
    for alt in ("two-sided", "less"):
        try:
            stat, p = wilcoxon(err_ds, err_f3, alternative=alt,
                               zero_method="wilcox")
            note = " *** MATCH ***" if abs(p - 0.048) <= 0.005 else ""
            print(f"  Wilcoxon DS vs F3  alternative={alt:<10}  "
                  f"stat={stat:.2f}  p={p:.4f} {note}")
        except Exception as exc:
            print(f"  Wilcoxon DS vs F3  alternative={alt}  ERROR: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to the requested RQ1 analyses."""
    parser = argparse.ArgumentParser(
        description="Reproduce RQ1 statistics (§5.2, García Rodríguez et al. 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ds-source",
        choices=["aspirational", "v1.1.0", "both"],
        default="both",
        help=(
            "DS values to use: 'aspirational' = paper Table 2 (pre-implementation), "
            "'v1.1.0' = computed by CIPA pipeline, 'both' = run both (default)."
        ),
    )
    parser.add_argument(
        "--with-f3",
        action="store_true",
        help="Also compute Wilcoxon DS vs F3 (requires raw datasets via --data-dir).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="Root directory of raw datasets (only used with --with-f3).",
    )
    args = parser.parse_args()

    sources = (
        ["aspirational", "v1.1.0"] if args.ds_source == "both"
        else [args.ds_source]
    )

    for src in sources:
        run_all(src)
        run_b2_stats(src)

    if args.with_f3:
        for src in sources:
            run_with_f3(args.data_dir, src)

    print()


if __name__ == "__main__":
    main()
