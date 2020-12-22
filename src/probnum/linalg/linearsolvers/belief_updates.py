"""Belief updates for probabilistic linear solvers."""
from typing import Callable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
import probnum.random_variables as rvs

# Public classes and functions. Order is reflected in documentation.
__all__ = ["BeliefUpdate", "LinearGaussianBeliefUpdate"]

# pylint: disable="invalid-name"


class BeliefUpdate:
    """Belief update of a probabilistic linear solver.

    Computes a new belief over the quantities of interest of the linear system based
    on the current state of the linear solver.

    Parameters
    ----------
    belief_update
        Callable defining how to update the belief.

    Examples
    --------

    See Also
    --------
    LinearGaussianBeliefUpdate: Belief update given linear observations :math:`y=As`.
    """

    def __init__(
        self,
        belief_update: Callable[
            [
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                np.ndarray,
                np.ndarray,
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
            Tuple[
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                "probnum.linalg.linearsolvers.LinearSolverState",
            ],
        ],
    ):
        self._belief_update = belief_update

    def __call__(
        self,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        "probnum.linalg.linearsolvers.LinearSolverState",
    ]:
        """Update belief over quantities of interest of the linear system.

        Parameters
        ----------
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        action :
            Action of the solver to probe the linear system with.
        observation :
            Observation of the linear system for the given action.
        solver_state :
            Current state of the linear solver.
        """
        return self._belief_update(belief, action, observation, solver_state)

    def update_solution(
        self, belief_x: rvs.RandomVariable, action: np.ndarray, observation: np.ndarray
    ) -> rvs.RandomVariable:
        """Update the belief over the solution :math:`x` of the linear system."""
        raise NotImplementedError

    def update_matrix(
        self, belief_A: rvs.RandomVariable, action: np.ndarray, observation: np.ndarray
    ) -> rvs.Normal:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def update_inverse(
        self,
        belief_Ainv: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.Normal:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def update_rhs(self, belief_b: rvs.RandomVariable) -> rvs.RandomVariable:
        """Update the belief over the right hand side of the linear system."""
        raise NotImplementedError

    # TODO: make update functions with kwargs part of this function and then
    #  implement different variants of it depending on the type of solver (
    #  matrixbased/solutionbased) etc.


class LinearGaussianBeliefUpdate(BeliefUpdate):
    """Belief update assuming (symmetric) Gaussianity and linear observations."""

    def __init__(self):
        super().__init__(belief_update=self.__call__)

    def __call__(
        self,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        "probnum.linalg.linearsolvers.LinearSolverState",
    ]:

        # Compute step size
        sy = action.T @ observation
        step_size = -np.squeeze((action.T @ solver_state.residual) / sy)

        # Rayleigh quotient
        log_rayleigh_quotient = self.log_rayleigh_quotient(
            action_obs_innerprod=sy, action=action
        )
        solver_state.log_rayleigh_quotients.append(log_rayleigh_quotient)

        # Belief updates
        belief.x = self.update_solution(
            belief_x=belief.x, action=action, step_size=step_size
        )
        belief.A = self.update_matrix(
            belief_A=belief.A, action=action, observation=observation
        )
        belief.Ainv = self.update_inverse(
            belief_Ainv=belief.Ainv, action=action, observation=observation
        )
        belief.b = self.update_rhs(belief_b=belief.b)

        # Update residual
        solver_state.residual = self.update_residual(
            residual=solver_state.residual, step_size=step_size, observation=observation
        )

        return belief, solver_state

    def update_solution(
        self,
        belief_x: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        step_size: float,
    ) -> rvs.RandomVariable:
        return belief_x + step_size * action

    def update_matrix(
        self, belief_A: rvs.RandomVariable, action: np.ndarray, observation: np.ndarray
    ) -> rvs.Normal:
        Vs = belief_A.cov.A @ action
        delta_A = observation - belief_A.mean @ action
        u_A = Vs / (action.T @ Vs)
        v_A = delta_A - 0.5 * (action.T @ delta_A) * u_A

        # Rank 2 mean update (+= uv' + vu')
        A_mean = linops.aslinop(belief_A.mean) + self._matrix_model_mean_update(
            u=u_A, v=v_A
        )

        # Rank 1 covariance Kronecker factor update (-= u_A(Vs)')
        if solver_state.iteration == 0:
            _A_covfactor_update_term = self._matrix_model_covariance_factor_update(
                u=u_A, Ws=Vs
            )
        else:
            _A_covfactor_update_term = (
                _A_covfactor_update_term
                + self._matrix_model_covariance_factor_update(u=u_A, Ws=Vs)
            )
        A_covfactor = linops.aslinop(self.A_covfactor0) - _A_covfactor_update_term

        return rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

    def update_inverse(
        self,
        belief_Ainv: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.Normal:
        Wy = belief_Ainv.cov.A @ observation
        delta_Ainv = action - belief_Ainv.mean @ observation
        yWy = np.squeeze(observation.T @ Wy)
        u_Ainv = Wy / yWy
        v_Ainv = delta_Ainv - 0.5 * (observation.T @ delta_Ainv) * u_Ainv

        # Rank 2 mean update (+= uv' + vu')
        Ainv_mean = linops.aslinop(belief_Ainv.mean) + self._matrix_model_mean_update(
            u=u_Ainv, v=v_Ainv
        )

        # Rank 1 covariance Kronecker factor update (-= u_Ainv(Wy)')
        if solver_state.iteration == 0:
            _Ainv_covfactor_update_term = self._matrix_model_covariance_factor_update(
                u=u_Ainv, Ws=Wy
            )
        else:
            _Ainv_covfactor_update_term = (
                _Ainv_covfactor_update_term
                + self._matrix_model_covariance_factor_update(u=u_Ainv, Ws=Wy)
            )
        Ainv_covfactor = (
            linops.aslinop(self.Ainv_covfactor0) - _Ainv_covfactor_update_term
        )

        return rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

    def update_rhs(self, belief_b: rvs.RandomVariable) -> rvs.RandomVariable:
        return belief_b

    def update_residual(
        self, residual: np.ndarray, step_size: float, observation: np.ndarray
    ) -> np.ndarray:
        """Update the residual :math:`r_i = Ax_i - b`."""
        return residual + step_size * observation

    def log_rayleigh_quotient(
        self, action_obs_innerprod: float, action: np.ndarray
    ) -> float:
        r"""Compute the log-Rayleigh quotient :math:`\ln R(A, s_i) = \ln(s_i^\top A
        s_i) -\ln(s_i^\top s_i)` for the current action."""
        return np.log(action_obs_innerprod) - np.log(action @ action)

    def _matrix_model_mean_update(self, u, v):
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' +
        vu')."""

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )

    def _matrix_model_covariance_factor_update(self, u, Ws):
        """Linear operator implementing the symmetric rank 2 covariance factor downdate
        (-= Ws u^T)."""

        def mv(x):
            return Ws @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )
