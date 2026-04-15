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
    """Result of a single complexity dimension (D1-D7). See §3.1 of García Rodríguez et al. (2026)."""

    value: float
    dimension_id: str
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_IDS: ClassVar[set[str]] = {f"D{i}" for i in range(1, 8)}

    def __post_init__(self) -> None:
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
        return {
            "value": self.value,
            "dimension_id": self.dimension_id,
            "components": _to_json_safe(self.components),
            "metadata": _to_json_safe(self.metadata),
        }


@dataclass
class DifficultyScore:
    """Aggregated Difficulty Score DS ∈ [0, 1]. See §3.2 of García Rodríguez et al. (2026)."""

    value: float
    band: str
    weights: tuple[float, ...]
    dimensions: tuple[DimensionResult, ...]

    _VALID_BANDS: ClassVar[set[str]] = {"Low", "Moderate", "High", "Extreme"}

    def __post_init__(self) -> None:
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
        return {
            "value": self.value,
            "band": self.band,
            "weights": list(self.weights),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "contributions": self.contributions,
        }


@dataclass
class ComplexityProfile:
    """Complexity Profile and Signature. See §3.3 of García Rodríguez et al. (2026)."""

    vector: tuple[float, ...]
    signature: str
    signature_name: str
    dominant_dimensions: list[str]

    _VALID_SIGS: ClassVar[set[str]] = {"I", "II", "III", "IV", "V"}

    def __post_init__(self) -> None:
        if len(self.vector) != 7:
            raise ValueError(f"vector must have length 7, got {len(self.vector)}")
        if self.signature not in self._VALID_SIGS:
            raise ValueError(f"signature must be one of {self._VALID_SIGS}, got {self.signature!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector),
            "signature": self.signature,
            "signature_name": self.signature_name,
            "dominant_dimensions": self.dominant_dimensions,
        }


@dataclass
class ActionRecommendation:
    """Preprocessing and modeling recommendations. See §3.4 of García Rodríguez et al. (2026)."""

    evaluation_metrics: list[str]
    preprocessing_strategy: list[str]
    model_families: list[str]
    validation_protocol: list[str]
    rationale: dict[str, str]
    warnings: list[str]

    def __post_init__(self) -> None:
        pass  # No invariants to enforce — lists may be empty

    def to_dict(self) -> dict[str, Any]:
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
    """Top-level result of the full CIPA pipeline. See §3 of García Rodríguez et al. (2026)."""

    dataset_name: str | None
    difficulty_score: DifficultyScore
    profile: ComplexityProfile
    action: ActionRecommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "difficulty_score": self.difficulty_score.to_dict(),
            "profile": self.profile.to_dict(),
            "action": self.action.to_dict(),
        }
