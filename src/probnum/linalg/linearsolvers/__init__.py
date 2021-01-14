"""Probabilistic linear solvers.

Implementation of components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
iterative methods for the solution of linear systems. Some combinations
generalizing and recovering classic iterative methods are listed below.


+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| **(Prior) Belief**                                | **Policy**                    | **Observation**             |    **Classic Iterative Method** |
+===================================================+===============================+=============================+=================================+
| :class:`.LinearSystemBelief`                      | :class:`.ConjugateDirections` | :class:`.MatVecObservation` |     Conjugate Directions Method |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| :meth:`.WeakMeanCorrespondenceBelief.from_scalar` | :class:`.ConjugateDirections` | :class:`.MatVecObservation` |  Conjugate Gradient Method (CG) |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| :class:`.WeakMeanCorrespondenceBelief`            | :class:`.ConjugateDirections` | :class:`.MatVecObservation` |               preconditioned CG |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| :class:`.LinearSystemBelief`                      | Unit vectors                  | :class:`.MatVecObservation` |            Gaussian Elimination |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| :class:`.LinearSystemBelief`                      | :math:`s=x_k`                 | :math:`y=SAs`               | (randomized) Kaczmarz algorithm |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
| :class:`.LinearSystemBelief`                      | sparse                        | :math:`y=SAs`               |          :math:`L_1` sketching? |
+---------------------------------------------------+-------------------------------+-----------------------------+---------------------------------+
"""

from ._probabilistic_linear_solver import LinearSolverState, ProbabilisticLinearSolver

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ProbabilisticLinearSolver", "LinearSolverState"]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
LinearSolverState.__module__ = "probnum.linalg.linearsolvers"
