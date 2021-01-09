"""Belief updates for probabilistic linear solvers."""
import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["BeliefUpdate", "SymMatrixNormalLinearObsBeliefUpdate"]

# pylint: disable="invalid-name,too-many-arguments"


class BeliefUpdate(abc.ABC):
    """Belief update of a probabilistic linear solver.

    Computes a new belief over the quantities of interest of the linear system based
    on the current state of the linear solver.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given linear observations
                                          :math:`y=As`.
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):
        """Belief update over quantities of interest of the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        solver_state :
            Current state of the linear solver.
        """
        self.problem = problem
        self.belief = belief
        self.solver_state = solver_state

    def __call__(
        self, action: np.ndarray, observation: np.ndarray
    ) -> Tuple[
        Tuple[rvs.RandomVariable, ...],
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """Update the belief over the quantities of interest of the linear system.

        Parameters
        ----------
        action :
            Action of the solver to probe the linear system with.
        observation :
            Observation of the linear system for the given action.
        """

        updated_beliefs = []
        for belief_update in [self.solution, self.inverse, self.matrix, self.rhs]:
            try:
                updated_beliefs.append(
                    belief_update(action=action, observation=observation)
                )
            except NotImplementedError:
                updated_beliefs.append(None)

        return tuple(updated_beliefs), self.solver_state

    def solution(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        """Update the belief over the solution :math:`x` of the linear system."""
        raise NotImplementedError

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def rhs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        """Update the belief over the right hand side :math:`b` of the linear system."""
        raise NotImplementedError


class SymMatrixNormalLinearObsBeliefUpdate(BeliefUpdate):
    r"""Belief update for a symmetric matrix-variate Normal prior and linear
    observations.

    Updates the posterior beliefs over the quantities of interest of the linear system
    under symmetric matrix-variate Gaussian prior(s) on :math:`A` and / or :math:`H`.
    Observations are assumed to be linear

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    solver_state :
        Current state of the linear solver.
    noise_cov
        Covariance matrix :math:`\Lambda` of the noise term :math:`E \sim \mathcal{
        N}(0, \Lambda)` assumed for matrix evaluations :math:`v \mapsto (A + E)v`.

    Examples
    --------

    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
        noise_cov: Optional[np.ndarray] = None,
    ):
        self.noise_cov = noise_cov
        super().__init__(problem=problem, belief=belief, solver_state=solver_state)

    def solution(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        if self.noise_cov is None:
            # Current residual
            try:
                residual = self.solver_state.residual
            except AttributeError:
                residual = self.problem.A @ self.belief.x.mean - self.problem.b
                if self.solver_state is not None:
                    self.solver_state.residual = residual

            # Step size
            step_size, solver_state = self._step_size(
                residual=residual,
                action=action,
                observation=observation,
            )
            # Solution update
            x = self.belief.x + step_size * action

            # Update residual
            _, solver_state = self._residual(
                residual=residual,
                step_size=step_size,
                observation=observation,
            )
            return x
        else:
            raise NotImplementedError

    def _solution_noise_free_obs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:

        """Update belief over the solution assuming noise-free observations
        :math:`y=As`."""

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        # Compute update terms
        Vs = self.belief.A.cov.A @ action
        delta_A = observation - self.belief.A.mean @ action
        u_A = Vs / (action.T @ Vs)
        v_A = delta_A - 0.5 * (action.T @ delta_A) * u_A

        # Rank 2 mean update (+= uv' + vu')
        A_mean = linops.aslinop(self.belief.A.mean) + self._matrix_model_mean_update_op(
            u=u_A, v=v_A
        )

        # Rank 1 covariance Kroknecker factor update (-= u_A(Vs)')
        A_covfactor = linops.aslinop(
            self.belief.A.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u_A, Ws=Vs)

        return rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        # Compute update terms
        Wy = self.belief.Ainv.cov.A @ observation
        delta_Ainv = action - self.belief.Ainv.mean @ observation
        u_Ainv = Wy / (observation.T @ Wy)
        v_Ainv = delta_Ainv - 0.5 * (observation.T @ delta_Ainv) * u_Ainv

        # Rank 2 mean update (+= uv' + vu')
        Ainv_mean = linops.aslinop(
            self.belief.Ainv.mean
        ) + self._matrix_model_mean_update_op(u=u_Ainv, v=v_Ainv)

        # Rank 1 covariance Kronecker factor update (-= u_Ainv(Wy)')
        Ainv_covfactor = linops.aslinop(
            self.belief.Ainv.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u_Ainv, Ws=Wy)

        return rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

    def rhs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> rvs.RandomVariable:
        return self.belief.b

    def _residual(
        self,
        residual: np.ndarray,
        step_size: float,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Update the residual :math:`r_i = Ax_i - b`."""
        new_residual = residual + step_size * observation
        if self.solver_state is not None:
            self.solver_state.residual = new_residual
        return new_residual

    def _matrix_model_mean_update_op(
        self, u: np.ndarray, v: np.ndarray
    ) -> linops.LinearOperator:
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' +
        vu')."""

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )

    def _matrix_model_covariance_factor_update_op(
        self, u: np.ndarray, Ws: np.ndarray
    ) -> linops.LinearOperator:
        """Linear operator implementing the symmetric rank 2 covariance factor downdate
        (-= Ws u^T)."""

        def mv(x):
            return Ws @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )

    def _step_size(
        self,
        residual: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> float:
        r"""Compute the step size :math:`\alpha` such that :math:`x_{i+1} = x_i +
        \alpha_i s_i`, where :math:`s_i` is the current action."""
        # Compute step size
        action_obs_innerprod = action.T @ observation
        step_size = (-action.T @ residual / action_obs_innerprod).item()

        # Update solver state
        if self.solver_state is not None:
            try:
                self.solver_state.step_sizes.append(step_size)
                self.solver_state.log_rayleigh_quotients.append(
                    _log_rayleigh_quotient(
                        action_obs_innerprod=action_obs_innerprod, action=action
                    )
                )
            except AttributeError:
                pass

        return step_size


def _log_rayleigh_quotient(action_obs_innerprod: float, action: np.ndarray) -> float:
    r"""Compute the log-Rayleigh quotient :math:`\ln R(A, s_i) = \ln(s_i^\top A
    s_i) -\ln(s_i^\top s_i)` for the current action."""
    return (np.log(action_obs_innerprod) - np.log(action.T @ action)).item()


# TODO: implement specific belief update for the CG equivalence class (maybe as a
#  subclass?) (and other
#  linear system beliefs, where inference may be done more efficiently, e.g. when only
#  a prior on the solution is specified.)
class WeakMeanCorrLinearObsBeliefUpdate(SymMatrixNormalLinearObsBeliefUpdate):
    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        phi: float,
        psi: float,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):
        self.phi = phi
        self.psi = psi
        super().__init__(
            problem=problem, belief=belief, solver_state=solver_state, noise_cov=None
        )
