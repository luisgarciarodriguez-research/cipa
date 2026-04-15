"""CIPA: A Multi-Domain Statistical Framework for Characterizing
Imbalanced Datasets and Computing a Difficulty Score.

Implements the four-stage CIPA framework (Characterization, Indexing,
Profiling, Action) as described in:

    García Rodríguez, L., Neme Castillo, J. A., & Gómez Adorno, H. M. (2026).
    CIPA: A Multi-Domain Statistical Framework for Characterizing Imbalanced
    Datasets and Computing a Difficulty Score.
    COMIA 2026 — XVIII Congreso Mexicano de Inteligencia Artificial.
    DOI: TODO (pending publication)

Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Universidad Nacional Autónoma de México (UNAM)

Development supported by SECIHTI (researcher ID (CVU) 905206, Luis García Rodríguez).

License: MIT — see LICENSE file for full terms.

Quick start
-----------
>>> import numpy as np
>>> from cipa import CIPADataset, CIPAPipeline
>>>
>>> X = np.load("features.npy")
>>> y = np.load("labels.npy")
>>>
>>> dataset = CIPADataset.from_arrays(X, y, name="MyDataset")
>>> pipeline = CIPAPipeline(random_state=42)
>>> result = pipeline.run(dataset)
>>>
>>> print(result.difficulty_score.value, result.difficulty_score.band)
>>> print(result.profile.signature, result.profile.signature_name)
"""

from cipa._constants import DEFAULT_WEIGHTS
from cipa._logging import *  # noqa: F403 — activates NullHandler
from cipa.dataset import CIPADataset
from cipa.pipeline import CIPAPipeline
from cipa.types import (
    ActionRecommendation,
    CIPAResult,
    ComplexityProfile,
    DifficultyScore,
    DimensionResult,
)

__version__ = "1.1.0"

__all__ = [
    "DEFAULT_WEIGHTS",
    "ActionRecommendation",
    "CIPADataset",
    "CIPAPipeline",
    "CIPAResult",
    "ComplexityProfile",
    "DifficultyScore",
    "DimensionResult",
]
