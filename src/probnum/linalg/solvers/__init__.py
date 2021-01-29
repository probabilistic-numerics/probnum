"""Probabilistic linear solvers.

Compositional implementation of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
iterative methods for the solution of linear systems. Some combinations
generalizing and recovering classic iterative methods are listed below.


+---------------------------------------------------+-------------------------------+------------------+---------------------------------+
| **(Prior) Belief**                                | **Policy**                    | **Observation**  |    **Classic Iterative Method** |
+===================================================+===============================+==================+=================================+
| :class:`.SymmetricLinearSystemBelief`             | :class:`.ConjugateDirections` | :class:`.MatVec` |     Conjugate Directions Method |
+---------------------------------------------------+-------------------------------+------------------+---------------------------------+
| :meth:`.WeakMeanCorrespondenceBelief.from_scalar` | :class:`.ConjugateDirections` | :class:`.MatVec` |  Conjugate Gradient Method (CG) |
+---------------------------------------------------+-------------------------------+------------------+---------------------------------+
| :meth:`.WeakMeanCorrespondenceBelief.from_inverse`| :class:`.ConjugateDirections` | :class:`.MatVec` |               preconditioned CG |
+---------------------------------------------------+-------------------------------+------------------+---------------------------------+
| :class:`.SymmetricLinearSystemBelief`             | :class:`.MaxSupNormColumn`    | :class:`.MatVec` |            Gaussian Elimination |
+---------------------------------------------------+-------------------------------+------------------+---------------------------------+
"""

from ._probabilistic_linear_solver import ProbabilisticLinearSolver
from ._state import LinearSolverCache, LinearSolverInfo, LinearSolverState
from .data import LinearSolverAction, LinearSolverData, LinearSolverObservation

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverInfo",
    "LinearSolverCache",
    "LinearSolverState",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.solvers"
LinearSolverInfo.__module__ = "probnum.linalg.solvers"
LinearSolverCache.__module__ = "probnum.linalg.solvers"
LinearSolverState.__module__ = "probnum.linalg.solvers"
