"""Stage A: Action Protocol — recommendation engine.

Maps a Complexity Profile and Difficulty Score to concrete recommendations
across four decision axes: evaluation metrics, preprocessing strategy,
model families, and validation protocol.

See §3.4 of García Rodríguez et al. (2026).

This module is part of the CIPA software package, companion implementation to:

    García Rodríguez, L., Neme Castillo, J. A., Gómez Adorno, H. M., & Fuentes Pineda, G. (2026).
    CIPA: A Multi-Domain Statistical Framework for Characterizing Imbalanced
    Datasets and Computing a Difficulty Score.
    COMIA 2026 — XVIII Congreso Mexicano de Inteligencia Artificial.
    DOI: TODO (pending publication)

Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Universidad Nacional Autónoma de México (UNAM)

Development supported by SECIHTI (researcher ID (CVU) 905206, Luis García Rodríguez).

License: MIT — see LICENSE file for full terms.
"""

from __future__ import annotations

import logging

from cipa.dataset import CIPADataset
from cipa.types import ActionRecommendation, ComplexityProfile, DifficultyScore

logger = logging.getLogger(__name__)

# Primary model recommendations per signature (§3.4)
_SIG_MODELS: dict[str, list[str]] = {
    "I":   ["Logistic Regression", "Linear SVM"],
    "II":  ["RBF-SVM", "Kernel methods"],
    "III": ["Random Forest", "Gradient Boosting"],
    "IV":  ["Linear models after dimensionality reduction", "Kernel SVM"],
    "V":   ["Gradient Boosting (XGBoost/LightGBM)", "Balanced Random Forest"],
}


def _add(lst: list[str], rationale: dict[str, str], item: str, reason: str) -> None:
    """Append *item* to *lst* if not already present; record *reason* in *rationale*."""
    if item not in lst:
        lst.append(item)
        rationale[item] = reason


def compute_action(
    profile: ComplexityProfile,
    difficulty_score: DifficultyScore,
    dataset: CIPADataset,
) -> ActionRecommendation:
    """Generate actionable recommendations for this dataset's profile.

    Parameters
    ----------
    profile : ComplexityProfile
        Signature and dominant dimensions.
    difficulty_score : DifficultyScore
        DS value and band (needed for threshold-based rules).
    dataset : CIPADataset
        Needed for N and IR (size/scale-based rules).

    Returns
    -------
    ActionRecommendation
        evaluation_metrics     : ordered list of recommended metrics
        preprocessing_strategy : ordered list of recommended strategies
        model_families         : ordered list of recommended models
        validation_protocol    : ordered list of protocol requirements
        rationale              : dict mapping each recommendation to the
                                 rule that triggered it
        warnings               : list of flagged conditions
    """
    ds = difficulty_score.value
    sig = profile.signature
    _d1, d2, d3, d4, d5, _d6, _d7 = profile.vector
    ir = dataset.IR
    n = dataset.N

    metrics: list[str] = []
    preprocessing: list[str] = []
    models: list[str] = []
    validation: list[str] = []
    rationale: dict[str, str] = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Evaluation Metrics (§3.4)
    # ------------------------------------------------------------------
    _add(metrics, rationale, "AUC-PR", "Always recommended for imbalanced data")
    _add(metrics, rationale, "F1-score (minority class)", "Always recommended for imbalanced data")

    if ds >= 0.50:
        _add(metrics, rationale, "G-mean", "DS >= 0.50: high difficulty requires balanced recall")
    if ds >= 0.75:
        _add(metrics, rationale, "MCC", "DS >= 0.75: extreme difficulty requires robust metric")
    if d2 >= 0.55:
        _add(metrics, rationale, "AUC-ROC", "D2 >= 0.55: overlap present, track global discrimination")
    if d3 >= 0.55:
        _add(metrics, rationale, "Recall (minority)", "D3 >= 0.55: rare/outlier sensitivity needed")

    if ir > 5:
        warnings.append(
            f"Accuracy is not a reliable metric for this dataset (IR={ir:.1f}). Use AUC-PR."
        )

    # ------------------------------------------------------------------
    # Preprocessing Strategy (§3.4)
    # ------------------------------------------------------------------
    if ds < 0.25:
        _add(preprocessing, rationale, "No resampling recommended; standard training",
             "DS < 0.25: low difficulty")
    else:
        _sig_preprocessing = {
            "I":   "Random Oversampling or SMOTE",
            "II":  "Borderline-SMOTE or ADASYN",
            "III": "DBSMOTE or MWMOTE (cluster-aware oversampling)",
            "IV":  "PCA or UMAP dimensionality reduction BEFORE resampling, then SMOTE",
            "V":   "Ensemble of strategies; start with SMOTE+ENN (combined over+under)",
        }
        _add(preprocessing, rationale, _sig_preprocessing[sig],
             f"Signature {sig}: {profile.signature_name}")

        if d3 >= 0.60:
            _add(preprocessing, rationale,
                 "Remove outlier instances before oversampling",
                 "D3 >= 0.60: high outlier/rare fraction")
        if d5 >= 0.70:
            _add(preprocessing, rationale,
                 "Mandatory: dimensionality reduction before any resampling",
                 "D5 >= 0.70: severe dimensionality burden")
        if ir > 100:
            _add(preprocessing, rationale,
                 "Under-sampling of majority class (RandomUnderSampler or NearMiss)",
                 "IR > 100: extreme imbalance")

    # ------------------------------------------------------------------
    # Model Families (§3.4)
    # ------------------------------------------------------------------
    for model in _SIG_MODELS[sig]:
        _add(models, rationale, model, f"Signature {sig}: {profile.signature_name}")

    if ds >= 0.75:
        _add(models, rationale,
             "Cost-sensitive learning (class_weight='balanced' or custom)",
             "DS >= 0.75: extreme difficulty")
    if d3 >= 0.60:
        _add(models, rationale,
             "One-class classification or isolation forest (baseline comparison)",
             "D3 >= 0.60: high outlier fraction")
    if n > 100_000:
        _add(models, rationale,
             "Prefer linear or tree-based models (scalability)",
             "N > 100,000: large dataset")

    # ------------------------------------------------------------------
    # Validation Protocol (§3.4)
    # ------------------------------------------------------------------
    _add(validation, rationale,
         "Stratified k-fold cross-validation (k=5 or k=10)",
         "Always: preserves IR in folds")
    _add(validation, rationale,
         "Report confidence intervals over folds",
         "Always: assess stability")

    if ir > 20:
        _add(validation, rationale,
             "Use k=10 folds minimum (minority class too small for k=5)",
             "IR > 20: minority too small for k=5")
    if ds >= 0.75:
        _add(validation, rationale,
             "Repeated stratified k-fold (5x10) for stability",
             "DS >= 0.75: extreme difficulty requires stability")
    if n < 500:
        _add(validation, rationale,
             "Leave-one-out CV or stratified 5-fold with bootstrap",
             "N < 500: very small dataset")
    if d4 >= 0.55:
        _add(validation, rationale,
             "Validate that each fold contains all sub-concepts",
             "D4 >= 0.55: fragmented minority")
    if d2 >= 0.70:
        _add(validation, rationale,
             "Compare with and without resampling to quantify overlap impact",
             "D2 >= 0.70: high overlap")

    if ir > 5:
        warnings.append(
            f"Random (non-stratified) train/test split not recommended (IR={ir:.1f})."
        )

    # ------------------------------------------------------------------
    # Additional warnings (§3.4)
    # ------------------------------------------------------------------
    if ds >= 0.75 and d4 >= 0.60:
        warnings.append(
            "High fragmentation detected. Standard SMOTE may generate noisy synthetic "
            "samples between disjunct sub-concepts. Use cluster-aware oversampling."
        )
    if d5 >= 0.70:
        warnings.append(
            "Effective dimensionality is high relative to minority density. k-NN based "
            "methods (SMOTE, kDN) may be unreliable without prior dimensionality reduction."
        )
    if n < 200 and ds >= 0.50:
        warnings.append(
            "Very small dataset with high difficulty. Results may be highly variable "
            "across validation folds."
        )
    if d3 >= 0.70:
        warnings.append(
            "Majority of minority instances are rare or outliers. Consider whether these "
            "are genuine minority instances or mislabeled noise."
        )

    logger.debug(
        "Action: %d metrics, %d strategies, %d models, %d protocols, %d warnings",
        len(metrics), len(preprocessing), len(models), len(validation), len(warnings),
    )

    return ActionRecommendation(
        evaluation_metrics=metrics,
        preprocessing_strategy=preprocessing,
        model_families=models,
        validation_protocol=validation,
        rationale=rationale,
        warnings=warnings,
    )
