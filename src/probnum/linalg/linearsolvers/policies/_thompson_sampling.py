from typing import Optional, Tuple

import numpy as np

import probnum
from probnum.linalg.linearsolvers.policies._policy import Policy
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ThompsonSampling"]


class ThompsonSampling(Policy):
    r"""Maximize the expected reward with respect to a random sample from the belief.

    Returns an action given by :math:`s_i = -H_ir_i`, where :math:`r = A_i x_i - b_i`
    and :math:`(x_i, A_i, H_i, b_i)` are drawn from the current belief over the
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
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, Optional["probnum.linalg.linearsolvers.LinearSolverState"]]:

        # Set seeds
        belief.x.random_state = self.random_state
        belief.A.random_state = self.random_state
        belief.Ainv.random_state = self.random_state
        belief.b.random_state = self.random_state

        # Sample from current belief
        x_sample = belief.x.sample()
        A_sample = belief.A.sample()
        Ainv_sample = belief.Ainv.sample()
        b_sample = belief.b.sample()

        # A-conjugate search direction / action (assuming exact arithmetic)
        action = -Ainv_sample @ (A_sample @ x_sample - b_sample)

        # Update solver state
        if solver_state is not None:
            solver_state.actions.append(action)

        return action, solver_state
