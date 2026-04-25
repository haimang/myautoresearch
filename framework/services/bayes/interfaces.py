"""Protocols for generic Bayesian refinement services."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CandidateEncoder(Protocol):
    def __call__(
        self,
        candidates: list[dict],
        feature_schema: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]: ...


class CandidateScorer(Protocol):
    def __call__(self, result: dict) -> float: ...


class CandidatePrior(Protocol):
    def __call__(self, candidate: dict) -> float: ...
