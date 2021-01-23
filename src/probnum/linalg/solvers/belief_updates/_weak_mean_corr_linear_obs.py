from functools import cached_property
from typing import Optional

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers.belief_updates._symmetric_normal_linear_obs import (
    SymmetricNormalLinearObsBeliefUpdate,
)
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["WeakMeanCorrLinearObsBeliefUpdate"]


class WeakMeanCorrLinearObsBeliefUpdate(SymmetricNormalLinearObsBeliefUpdate):
    r"""Weak mean correspondence belief update assuming linear observations.

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    solver_state :
        Current state of the linear solver.

    Examples
    --------
    Efficient updating of the solution covariance trace.

    >>> from probnum.linalg.solvers.belief_updates import WeakMeanCorrLinearObsBeliefUpdate
    >>>
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.WeakMeanCorrespondenceBelief",
        actions: np.ndarray,
        observations: np.ndarray,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ):

        super().__init__(
            problem=problem,
            belief=belief,
            actions=actions,
            observations=observations,
            hyperparams=None,
            solver_state=solver_state,
        )

    @cached_property
    def A(self) -> rvs.Normal:
        # Update belief about A assuming WS=Y
        mean_update, covfactor_update = self.A_update_terms(belief_A=self.belief.A_eps)
        if self.belief._A_covfactor_update_op is not None:
            self.belief._A_covfactor_update_op += covfactor_update
        else:
            self.belief._A_covfactor_update_op = covfactor_update

        # Action observation inner products
        if self.solver_state is not None:
            try:
                action_obs_innerprod = self.solver_state.action_obs_innerprods[
                    self.solver_state.iteration
                ]
            except IndexError:
                action_obs_innerprod = self.action.T @ self.observation
        else:
            action_obs_innerprod = self.action.T @ self.observation

        return rvs.Normal(
            mean=linops.aslinop(self.belief.A_eps.mean) + mean_update,
            cov=linops.SymmetricKronecker(
                self.belief._cov_factor_matrix(
                    action_obs_innerprods=action_obs_innerprod
                )
                - self.belief._A_covfactor_update_op
            ),
        )

    @cached_property
    def Ainv(self) -> rvs.Normal:
        # Update belief about Ainv assuming WY=H_0Y (Theorem 3, eqn. 1+2, Wenger2020)
        u, v, Wy = self._matrix_model_update_components(
            belief_matrix=self.belief.Ainv,
            action=self.observation,
            observation=self.action,
        )
        # Rank 2 mean update (+= uv' + vu')
        mean_update = self._matrix_model_mean_update_op(u=u, v=v)
        # Rank 1 covariance Kronecker factor update (-= u(Wy)')
        covfactor_update = self._matrix_model_covariance_factor_update_op(u=u, Ws=Wy)
        if self.belief._Ainv_covfactor_update_op is not None:
            self.belief._Ainv_covfactor_update_op += covfactor_update
        else:
            self.belief._Ainv_covfactor_update_op = covfactor_update

        covfactor_op = (
            self.belief._cov_factor_inverse() - self.belief._Ainv_covfactor_update_op
        )

        # Update trace efficiently
        covfactor_op.trace = lambda: self._Ainv_covfactor_trace(
            y=self.observation, Wy=Wy
        )

        return rvs.Normal(
            mean=linops.aslinop(self.belief.Ainv.mean) + mean_update,
            cov=linops.SymmetricKronecker(covfactor_op),
        )

    def _Ainv_covfactor_trace(self, y: np.ndarray, Wy: np.ndarray):
        r"""Trace of the covariance factor of the inverse model.

        Implements the recursive trace update for the covariance factor of the inverse
        model given by

        .. math::
            \tr(W_k^H) = tr(W_{k-1}^H) - \frac{1}{y_k^\top W_{k-1}^H y_k} \lVert W_{
            k-1}^H y_k \rVert^2.

        See section S4.3 of Wenger and Hennig, 2020 for details.

        Parameters
        ----------
        y : Observation
        Wy : Inverse model covariance factor applied to observation.
        """
        return (
            self.belief.Ainv.cov.A_eps.trace()
            - 1 / (y.T @ Wy).item() * (Wy.T @ Wy).item()
        )
