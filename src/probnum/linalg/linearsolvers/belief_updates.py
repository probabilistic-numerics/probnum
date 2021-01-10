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
        "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
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

        updated_belief = self.belief
        try:
            updated_belief.x = self.solution(action=action, observation=observation)
        except NotImplementedError:
            updated_belief.x = None
        try:
            updated_belief.Ainv = self.inverse(action=action, observation=observation)
        except NotImplementedError:
            updated_belief.Ainv = None
        try:
            updated_belief.A = self.matrix(action=action, observation=observation)
        except NotImplementedError:
            updated_belief.A = None
        try:
            updated_belief.b = self.rhs(action=action, observation=observation)
        except NotImplementedError:
            updated_belief.b = None

        return updated_belief, self.solver_state

    def solution(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_x: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        """Update the belief over the solution :math:`x` of the linear system."""
        raise NotImplementedError

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_Ainv: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def rhs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_b: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        """Update the belief over the right hand side :math:`b` of the linear system."""
        raise NotImplementedError


class NormalLinearObsBeliefUpdate(BeliefUpdate):
    """Belief update assuming Gaussian random variables."""

    def solution(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_x: rvs.Normal = None,
    ) -> rvs.Normal:
        """Update the belief over the solution :math:`x` of the linear system."""
        return self.prior.x + self.solution_matheron(
            action=action, observation=observation, belief_x=self.prior.x
        )

    def solution_matheron_update_term(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_x: rvs.Normal = None,
    ) -> rvs.Normal:
        """Matheron update term for the solution."""
        raise NotImplementedError

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.Normal = None,
    ) -> rvs.Normal:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def matrix_matheron_update_term(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.Normal = None,
    ) -> rvs.Normal:
        """Matheron update term for the system matrix."""
        raise NotImplementedError

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_Ainv: rvs.Normal = None,
    ) -> rvs.Normal:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def inverse_matheron_update_term(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.Normal = None,
    ) -> rvs.Normal:
        """Matheron update term for the system matrix."""
        raise NotImplementedError

    def rhs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_b: rvs.Normal = None,
    ) -> rvs.Normal:
        """Update the belief over the right hand side :math:`b` of the linear system."""
        raise NotImplementedError

    def rhs_matheron_update_term(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        """Matheron update term for the system matrix."""
        raise NotImplementedError


class SymMatrixNormalLinearObsBeliefUpdate(NormalLinearObsBeliefUpdate):
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
        belief_x: rvs.RandomVariable = None,
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
            step_size = self._step_size(
                residual=residual,
                action=action,
                observation=observation,
            )
            # Solution update
            x = self.belief.x + step_size * action

            # Update residual
            self._residual(
                residual=residual,
                step_size=step_size,
                observation=observation,
            )
            return x
        else:
            raise NotImplementedError

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        u, v, Ws = self._matrix_model_update_components(
            belief_matrix=self.belief.Ainv, action=action, observation=observation
        )

        # Rank 2 mean update (+= uv' + vu')
        A_mean = linops.aslinop(self.belief.A.mean) + self._matrix_model_mean_update_op(
            u=u, v=v
        )

        # Rank 1 covariance Kronecker factor update (-= u(Ws)')
        A_covfactor = linops.aslinop(
            self.belief.A.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u, Ws=Ws)

        return rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_Ainv: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        u, v, Wy = self._matrix_model_update_components(
            belief_matrix=self.belief.Ainv, action=observation, observation=action
        )

        # Rank 2 mean update (+= uv' + vu')
        Ainv_mean = linops.aslinop(
            self.belief.Ainv.mean
        ) + self._matrix_model_mean_update_op(u=u, v=v)

        # Rank 1 covariance Kronecker factor update (-= u(Wy)')
        Ainv_covfactor = linops.aslinop(
            self.belief.Ainv.cov.A
        ) - self._matrix_model_covariance_factor_update_op(u=u, Ws=Wy)

        return rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

    def rhs(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_b: rvs.RandomVariable = None,
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

    def _matrix_model_update_components(
        self,
        belief_matrix: rvs.RandomVariable,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Computes the components :math:`u=Ws(s^\top Ws)^{-1}` and :math:`v=\Delta
        - \frac{1}{2}(y^\top \Delta) u` of the update."""
        Ws = belief_matrix.cov.A @ action
        delta_A = observation - belief_matrix.mean @ action
        u = Ws / (action.T @ Ws)
        v = delta_A - 0.5 * (action.T @ delta_A) * u

        return u, v, Ws

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
        (-= Ws u')."""

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
    """Belief update for the weak mean correspondence covariance class under linear
    observations."""

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.WeakMeanCorrespondenceBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):
        self.prior_belief = belief
        super().__init__(
            problem=problem, belief=belief, solver_state=solver_state, noise_cov=None
        )

    def matrix(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_A: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        # TODO implement this under the assumption that W = A
        raise NotImplementedError

    def inverse(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        belief_Ainv: rvs.RandomVariable = None,
    ) -> rvs.RandomVariable:
        raise NotImplementedError

    def _matrix_trace_update(self):
        A.trace = None
        return A

    def _inverse_trace_update(self):
        raise NotImplementedError
