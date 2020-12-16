"""Definitions and collection of problems solved by probabilistic numerical methods."""

from ._problems import (
    InitialValueProblem,
    LinearSystem,
    QuadratureProblem,
    RegressionProblem,
)

__all__ = [
    "RegressionProblem",
    "InitialValueProblem",
    "LinearSystem",
    "QuadratureProblem",
]
