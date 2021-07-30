"""Base class for policies of probabilistic linear solvers returning actions."""
import abc

import numpy as np

import probnum  # pylint: disable="unused-import"


class LinearSolverPolicy(abc.ABC):
    r"""Policy of a (probabilistic) linear solver.

    The policy :math:`\pi(s \mid \mathsf{A}, \mathsf{H}, \mathsf{x}, A, b)` of a
    linear solver returns a vector to probe the linear system with, typically via
    multiplication, resulting in an observation. Policies can either be deterministic or
    stochastic depending on the application.

    See Also
    --------
    ConjugateDirectionsPolicy : Policy returning :math:`A`-conjugate actions.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:
        """Return an action for a given solver state.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
