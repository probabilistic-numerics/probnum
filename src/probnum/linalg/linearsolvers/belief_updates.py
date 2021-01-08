"""Belief updates for probabilistic linear solvers."""
import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

from .beliefs import LinearSystemBelief

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

    def __init__(self, hyperparams: Optional[Tuple[np.ndarray]] = None):
        self._hyperparams = hyperparams

    @property
    def hyperparameters(self) -> Tuple[np.ndarray]:
        return self._hyperparams

    @hyperparameters.setter
    def hyperparameters(self, hyperparams: Tuple[np.ndarray]):
        self._hyperparams = hyperparams

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        "probnum.linalg.linearsolvers.LinearSolverState",
    ]:
        """Update belief over quantities of interest of the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.
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
        raise NotImplementedError

    def update_solution(
        self,
        problem: LinearSystem,
        belief_x: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        """Update the belief over the solution :math:`x` of the linear system."""
        raise NotImplementedError

    def update_matrix(
        self,
        problem: LinearSystem,
        belief_A: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def update_inverse(
        self,
        problem: LinearSystem,
        belief_Ainv: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def update_rhs(
        self,
        problem: LinearSystem,
        belief_b: rvs.RandomVariable,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
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
    noise_cov
        Covariance matrix :math:`\Lambda` of the noise term :math:`E \sim \mathcal{
        N}(0, \Lambda)` assumed for matrix evaluations :math:`v \mapsto (A + E)v`.

    Examples
    --------

    """

    def __init__(
        self,
        noise_cov: Union[
            np.ndarray, linops.LinearOperator, scipy.sparse.spmatrix
        ] = None,
    ):
        self.noise_cov = noise_cov
        super().__init__()

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        "probnum.linalg.linearsolvers.LinearSolverState",
    ]:

        # Belief updates
        belief_A, solver_state = self.update_matrix(
            problem=problem,
            belief_A=belief.A,
            action=action,
            observation=observation,
            solver_state=solver_state,
        )
        belief_Ainv, solver_state = self.update_inverse(
            problem=problem,
            belief_Ainv=belief.Ainv,
            action=action,
            observation=observation,
            solver_state=solver_state,
        )
        belief_b, solver_state = self.update_rhs(
            problem=problem, belief_b=belief.b, solver_state=solver_state
        )
        belief_x, solver_state = self.update_solution(
            problem=problem,
            belief_Ainv=belief_Ainv,
            belief_x=belief.x,
            action=action,
            observation=observation,
            solver_state=solver_state,
        )

        belief = LinearSystemBelief(
            x=belief_x, A=belief_A, Ainv=belief_Ainv, b=belief_b
        )

        return belief, solver_state

    def update_solution(
        self,
        problem: LinearSystem,
        belief_Ainv: rvs.RandomVariable,
        belief_x: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        if self.noise_cov is None:
            return self._update_solution_noise_free_obs(
                problem=problem,
                belief_x=belief_x,
                action=action,
                observation=observation,
                solver_state=solver_state,
            )
        else:
            return self._update_solution_noisy_obs(
                problem=problem, belief_Ainv=belief_Ainv, solver_state=solver_state
            )

    def _update_solution_noise_free_obs(
        self,
        problem: LinearSystem,
        belief_x: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        """Update belief over the solution assuming noise-free observations
        :math:`y=As`."""
        # Current residual
        try:
            residual = solver_state.residual
        except AttributeError:
            residual = problem.A @ belief_x.mean - problem.b
            if solver_state is not None:
                solver_state.residual = residual

        # Step size
        step_size, solver_state = _step_size(
            residual=residual,
            action=action,
            observation=observation,
            solver_state=solver_state,
        )
        # Solution update
        x = belief_x + step_size * action

        # Update residual
        _, solver_state = self._update_residual(
            residual=residual,
            step_size=step_size,
            observation=observation,
            solver_state=solver_state,
        )
        return x, solver_state

    def _update_solution_noisy_obs(
        self,
        problem: LinearSystem,
        belief_Ainv: rvs.RandomVariable,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        """Update belief over the solution given noisy observations :math:`y=(A+E)s`."""
        return belief_Ainv @ problem.b, solver_state

    def update_matrix(
        self,
        problem: LinearSystem,
        belief_A: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        # Compute update terms
        Vs = belief_A.cov.A @ action
        delta_A = observation - belief_A.mean @ action
        u_A = Vs / (action.T @ Vs)
        v_A = delta_A - 0.5 * (action.T @ delta_A) * u_A

        # Rank 2 mean update (+= uv' + vu')
        A_mean = linops.aslinop(belief_A.mean) + self._matrix_model_mean_update_op(
            u=u_A, v=v_A
        )

        # Rank 1 covariance Kroknecker factor update (-= u_A(Vs)')
        A_covfactor = linops.aslinop(
            belief_A.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u_A, Ws=Vs)

        A = rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

        return A, solver_state

    def update_inverse(
        self,
        problem: LinearSystem,
        belief_Ainv: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        # Compute update terms
        Wy = belief_Ainv.cov.A @ observation
        delta_Ainv = action - belief_Ainv.mean @ observation
        u_Ainv = Wy / (observation.T @ Wy)
        v_Ainv = delta_Ainv - 0.5 * (observation.T @ delta_Ainv) * u_Ainv

        # Rank 2 mean update (+= uv' + vu')
        Ainv_mean = linops.aslinop(
            belief_Ainv.mean
        ) + self._matrix_model_mean_update_op(u=u_Ainv, v=v_Ainv)

        # Rank 1 covariance Kronecker factor update (-= u_Ainv(Wy)')
        Ainv_covfactor = linops.aslinop(
            belief_Ainv.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u_Ainv, Ws=Wy)

        Ainv = rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

        return Ainv, solver_state

    def update_rhs(
        self,
        problem: LinearSystem,
        belief_b: rvs.RandomVariable,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        rvs.RandomVariable, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
    ]:
        return belief_b, solver_state

    def _update_residual(
        self,
        residual: np.ndarray,
        step_size: float,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, "probnum.linalg.linearsolvers.LinearSolverState"]:
        """Update the residual :math:`r_i = Ax_i - b`."""
        # pylint: disable="no-self-use"
        new_residual = residual + step_size * observation
        if solver_state is not None:
            solver_state.residual = new_residual
        return new_residual, solver_state

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
    residual: np.ndarray,
    action: np.ndarray,
    observation: np.ndarray,
    solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
) -> Tuple[float, "probnum.linalg.linearsolvers.LinearSolverState"]:
    r"""Compute the step size :math:`\alpha` such that :math:`x_{i+1} = x_i +
    \alpha_i s_i`, where :math:`s_i` is the current action."""
    # Compute step size
    action_obs_innerprod = action.T @ observation
    step_size = (-action.T @ residual / action_obs_innerprod).item()

    # Update solver state
    if solver_state is not None:
        try:
            solver_state.step_sizes.append(step_size)
            solver_state.log_rayleigh_quotients.append(
                _log_rayleigh_quotient(
                    action_obs_innerprod=action_obs_innerprod, action=action
                )
            )
        except AttributeError:
            pass

    return step_size, solver_state


def _log_rayleigh_quotient(action_obs_innerprod: float, action: np.ndarray) -> float:
    r"""Compute the log-Rayleigh quotient :math:`\ln R(A, s_i) = \ln(s_i^\top A
    s_i) -\ln(s_i^\top s_i)` for the current action."""
    return (np.log(action_obs_innerprod) - np.log(action.T @ action)).item()


# TODO: implement specific belief update for the CG equivalence class (maybe as a
#  subclass?) (and other
#  linear system beliefs, where inference may be done more efficiently, e.g. when only
#  a prior on the solution is specified.)
class WeakMeanCorrLinearObsBeliefUpdate(SymMatrixNormalLinearObsBeliefUpdate):
    pass
