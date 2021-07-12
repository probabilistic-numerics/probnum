"""Policy returning randomly drawn standard unit vectors."""
import numpy as np

import probnum  # pylint: disable="unused-import"

from . import _linear_solver_policy


class RandomUnitVectorPolicy(_linear_solver_policy.LinearSolverPolicy):
    r"""Policy returning randomly drawn standard unit vectors.

    Draw a standard unit vector :math:`e_i` at random and return it. This policy corresponds to selecting columns of the matrix as observations.
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:

        n = solver_state.problem.A.shape[1]
        idx = solver_state.rng.choice(n, 1)
        action = np.zeros(n)
        action[idx] = 1.0
        return action
