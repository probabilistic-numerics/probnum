import copy
from typing import Optional, Tuple

import numpy as np

import probnum
from probnum.linalg.solvers.policies._policy import Policy
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ThompsonSampling"]


class ThompsonSampling(Policy):
    r"""Maximize the expected reward with respect to a random sample from the belief.

    Returns an action given by :math:`s_i = -H_ir_i`, where :math:`r = A_i x_i - b_i`
    and :math:`(x_i, A_i, H_i, b_i)` are drawn from the current belief about the
    corresponding linear system components. The resulting action is the exact step to
    the solution of the linear system assuming :math:`H_i = A^{-1}`, :math:`A_i=A`, and
    :math:`b_i=b`.

    Parameters
    ----------
    random_state
        Random state of the policy. If None (or :mod:`numpy.random`), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """

    def __init__(self, random_state=None):
        super().__init__(
            policy=self.__call__, is_deterministic=False, random_state=random_state
        )

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> np.ndarray:

        # Set seeds
        if self.random_state != belief.x.random_state:
            x = copy.copy(belief.x)
            A = copy.copy(belief.A)
            Ainv = copy.copy(belief.Ainv)
            b = copy.copy(belief.b)

            x.random_state = self.random_state
            A.random_state = self.random_state
            Ainv.random_state = self.random_state
            b.random_state = self.random_state
        else:
            x, A, Ainv, b = belief.x, belief.A, belief.Ainv, belief.b

        # A-conjugate action under sampled belief
        action = -Ainv.sample() @ (A.sample() @ x.sample() - b.sample())

        return action
