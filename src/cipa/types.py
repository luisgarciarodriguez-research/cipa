"""Result types for the CIPA framework. See §3 of García Rodríguez et al. (2026).

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

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


def _to_json_safe(value: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python builtins."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass
class DimensionResult:
    """Result of a single complexity dimension (D1–D7). See §3.1 of García Rodríguez et al. (2026).

    Attributes
    ----------
    value : float
        Normalized dimension score in [0, 1]. Higher = higher difficulty.
    dimension_id : str
        Identifier string, one of "D1"–"D7".
    components : dict[str, Any]
        Intermediate computed quantities specific to this dimension
        (e.g. ``{"F3": ..., "N1": ..., "kDN": ...}`` for D2). Keys vary by dimension;
        see each ``compute_d*`` function for the exact set.
    metadata : dict[str, Any]
        Auxiliary information (e.g. hyperparameter values, subsample flags)
        that does not contribute directly to the dimension value.
    """

    value: float
    dimension_id: str
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_IDS: ClassVar[set[str]] = {f"D{i}" for i in range(1, 8)}

    def __post_init__(self) -> None:
        """Coerce value to float and validate dimension_id membership and value bounds."""
        self.value = float(self.value)
        if self.dimension_id not in self._VALID_IDS:
            raise ValueError(
                f"dimension_id must be one of {sorted(self._VALID_IDS)}, "
                f"got {self.dimension_id!r}"
            )
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"DimensionResult.value must be in [0, 1], got {self.value}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with value, dimension_id, components, and metadata."""
        return {
            "value": self.value,
            "dimension_id": self.dimension_id,
            "components": _to_json_safe(self.components),
            "metadata": _to_json_safe(self.metadata),
        }


@dataclass
class DifficultyScore:
    """Aggregated Difficulty Score DS ∈ [0, 1]. See §3.2 of García Rodríguez et al. (2026).

    Attributes
    ----------
    value : float
        DS = Σ wᵢ · Dᵢ, clipped to [0, 1].
    band : str
        Interpretation band: "Low" | "Moderate" | "High" | "Extreme".
    weights : tuple of 7 floats
        Dimension weights used for aggregation (w₁, …, w₇).
    dimensions : tuple of 7 DimensionResult
        Raw dimension results ordered D1–D7.
    contributions : dict[str, float]
        Read-only property mapping each dimension ID to its weighted
        contribution wᵢ · Dᵢ.
    """

    value: float
    band: str
    weights: tuple[float, ...]
    dimensions: tuple[DimensionResult, ...]

    _VALID_BANDS: ClassVar[set[str]] = {"Low", "Moderate", "High", "Extreme"}

    def __post_init__(self) -> None:
        """Coerce value to float and validate value bounds, band membership, and tuple lengths."""
        self.value = float(self.value)
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"DifficultyScore.value must be in [0, 1], got {self.value}")
        if self.band not in self._VALID_BANDS:
            raise ValueError(f"band must be one of {self._VALID_BANDS}, got {self.band!r}")
        if len(self.weights) != 7:
            raise ValueError(f"weights must have length 7, got {len(self.weights)}")
        if len(self.dimensions) != 7:
            raise ValueError(f"dimensions must have length 7, got {len(self.dimensions)}")

    @property
    def contributions(self) -> dict[str, float]:
        """Per-dimension contribution w_i * D_i to the total DS."""
        return {
            d.dimension_id: w * d.value
            for d, w in zip(self.dimensions, self.weights, strict=False)
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with value, band, weights, dimensions, and contributions."""
        return {
            "value": self.value,
            "band": self.band,
            "weights": list(self.weights),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "contributions": self.contributions,
        }


@dataclass
class ComplexityProfile:
    """Complexity Profile and Signature. See §3.3 of García Rodríguez et al. (2026).

    Attributes
    ----------
    vector : tuple of 7 floats
        Dimension scores (D1, D2, …, D7) as a flat tuple.
    signature : str
        Assigned complexity signature: "I" | "II" | "III" | "IV" | "V".
    signature_name : str
        Human-readable signature name (e.g. "Overlap-dominated").
    dominant_dimensions : list[str]
        Dimension IDs with value ≥ elevation_threshold (0.55), sorted
        in descending order of their value.
    """

    vector: tuple[float, ...]
    signature: str
    signature_name: str
    dominant_dimensions: list[str]

    _VALID_SIGS: ClassVar[set[str]] = {"I", "II", "III", "IV", "V"}

    def __post_init__(self) -> None:
        """Validate vector length and signature membership."""
        if len(self.vector) != 7:
            raise ValueError(f"vector must have length 7, got {len(self.vector)}")
        if self.signature not in self._VALID_SIGS:
            raise ValueError(f"signature must be one of {self._VALID_SIGS}, got {self.signature!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with vector, signature, signature_name, and dominant_dimensions."""
        return {
            "vector": list(self.vector),
            "signature": self.signature,
            "signature_name": self.signature_name,
            "dominant_dimensions": self.dominant_dimensions,
        }


@dataclass
class ActionRecommendation:
    """Preprocessing and modeling recommendations. See §3.4 of García Rodríguez et al. (2026).

    Attributes
    ----------
    evaluation_metrics : list[str]
        Recommended evaluation metrics, ordered by priority (AUC-PR is always first).
    preprocessing_strategy : list[str]
        Recommended preprocessing steps, ordered by priority.
    model_families : list[str]
        Recommended model families, ordered by priority.
    validation_protocol : list[str]
        Required validation protocol steps (stratified k-fold is always included).
    rationale : dict[str, str]
        Maps each recommendation string to the triggering rule or condition
        (e.g. "D2 >= 0.55: overlap present, track global discrimination").
    warnings : list[str]
        Dataset-specific flags (e.g. small N, extreme IR, high outlier fraction).
    """

    evaluation_metrics: list[str]
    preprocessing_strategy: list[str]
    model_families: list[str]
    validation_protocol: list[str]
    rationale: dict[str, str]
    warnings: list[str]

    def __post_init__(self) -> None:
        """No field invariants to enforce — all lists may be empty."""
        pass  # No invariants to enforce — lists may be empty

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with all six recommendation fields."""
        return {
            "evaluation_metrics": self.evaluation_metrics,
            "preprocessing_strategy": self.preprocessing_strategy,
            "model_families": self.model_families,
            "validation_protocol": self.validation_protocol,
            "rationale": self.rationale,
            "warnings": self.warnings,
        }


@dataclass
class CIPAResult:
    """Top-level result of the full CIPA pipeline. See §3 of García Rodríguez et al. (2026).

    Attributes
    ----------
    dataset_name : str or None
        The name passed to CIPADataset (None if not set).
    difficulty_score : DifficultyScore
        Aggregated DS value, band, weights, and per-dimension contributions.
    profile : ComplexityProfile
        Complexity Signature and dominant dimension vector.
    action : ActionRecommendation
        Four-axis recommendation set with rationale and warnings.
    """

    dataset_name: str | None
    difficulty_score: DifficultyScore
    profile: ComplexityProfile
    action: ActionRecommendation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with dataset_name, difficulty_score, profile, and action."""
        return {
            "dataset_name": self.dataset_name,
            "difficulty_score": self.difficulty_score.to_dict(),
            "profile": self.profile.to_dict(),
            "action": self.action.to_dict(),
        }
