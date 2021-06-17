"""Definitions and collection of problems solved by probabilistic numerical methods."""

from ._problems import (
    InitialValueProblem,
    LinearSystem,
    QuadratureProblem,
    TimeSeriesRegressionProblem,
)

__all__ = [
    "TimeSeriesRegressionProblem",
    "InitialValueProblem",
    "LinearSystem",
    "QuadratureProblem",
]

# Set correct module paths. Corrects links and module paths in documentation.
TimeSeriesRegressionProblem.__module__ = "probnum.problems"
InitialValueProblem.__module__ = "probnum.problems"
LinearSystem.__module__ = "probnum.problems"
QuadratureProblem.__module__ = "probnum.problems"
