"""Base class for linear solver stopping criteria."""

import abc

import probnum  # pylint: disable="unused-import"


class LinearSolverStoppingCriterion(abc.ABC):
    r"""Stopping criterion of a (probabilistic) linear solver.

    Checks whether quantities tracked by the :class:`~probnum.linalg.solvers.LinearSolverState` meet a desired terminal condition.

    See Also
    --------
    ResidualNormStoppingCriterion : Stop based on the norm of the residual.
    PosteriorContractionStoppingCriterion : Stop based on the uncertainty about the quantity of interest.
    MaxIterationsStoppingCriterion : Stop after a maximum number of iterations.
    """

    @abc.abstractmethod
    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> bool:
        """Check whether tracked quantities meet a desired terminal condition.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
