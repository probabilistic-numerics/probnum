"""Belief updates for probabilistic linear solvers."""
import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum  # pylint: disable="unused-import"
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["BeliefUpdate", "SymMatrixNormalLinearObsBeliefUpdate"]

# pylint: disable="invalid-name,too-many-arguments"


class BeliefUpdate(abc.ABC):
    r"""Belief update of a probabilistic linear solver.

    Computes the updated beliefs over quantities of interest of a linear system after
    making observations about the system given a prior belief.

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    solver_state :
        Current state of the linear solver.

    See Also
    --------
    SymMatrixNormalLinearObsBeliefUpdate: Belief update given a symmetric
        matrix-variate normal belief and linear observations.
    """

    def __init__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: np.ndarray,
        observations: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):
        self.problem = problem
        self.belief = belief
        self.actions = actions
        self.observations = observations
        self._x = None
        self._Ainv = None
        self._A = None
        self._b = None
        self.solver_state = solver_state

    def __call__(
        self, action: np.ndarray, observation: np.ndarray
    ) -> Tuple[
        rvs.RandomVariable,
        rvs.RandomVariable,
        rvs.RandomVariable,
        rvs.RandomVariable,
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
        return (
            self.x,
            self.Ainv,
            self.A,
            self.b,
            self.solver_state,
        )

    @cached_property
    def x(
        self,
    ) -> rvs.RandomVariable:
        """Updated belief over the solution :math:`x` of the linear system."""
        raise NotImplementedError

    @cached_property
    def A(
        self,
    ) -> rvs.RandomVariable:
        """Updated belief over the system matrix :math:`A`."""
        raise NotImplementedError

    @cached_property
    def Ainv(
        self,
    ) -> rvs.RandomVariable:
        """Updated belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    @cached_property
    def b(
        self,
    ) -> rvs.RandomVariable:
        """Updated belief over the right hand side :math:`b` of the linear system."""
        raise NotImplementedError


class SymMatrixNormalLinearObsBeliefUpdate(BeliefUpdate):
    r"""Belief update for a symmetric matrix-variate Normal belief and linear
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
        actions: np.ndarray,
        observations: np.ndarray,
        noise_cov: Optional[np.ndarray] = None,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):
        self.noise_cov = noise_cov
        super().__init__(
            problem=problem,
            belief=belief,
            actions=actions,
            observations=observations,
            solver_state=solver_state,
        )

    @cached_property
    def x(self) -> rvs.Normal:
        if self.noise_cov is None:
            x_mean_update, _ = self.x_update_terms
            return rvs.Normal(
                mean=self.belief.x.mean + x_mean_update,
                cov=self.belief._induced_solution_cov(Ainv=self.Ainv, b=self.b),
            )
        else:
            return self.Ainv @ self.b

    @cached_property
    def x_update_terms(
        self,
    ) -> Tuple[np.ndarray, Union[np.ndarray, linops.LinearOperator]]:
        """
        Computes the update terms for the mean and covariance of a quantity of
        interest of the linear system based on the current belief and state of the linear
        solver. Formally, for a prior belief :math:`q \sim \mathcal{N}(\mu, \Sigma)` over a
        quantity of interest, compute the update terms :math:`\mu_{\text{update}(y)}` and
        :math:`\Sigma_{\text{update}}(y)` given observations :math:`y`, such that :math:`q
        \mid y \sim \mathcal{N}(\mu +\mu_{\text{update}}(y), \Sigma - \Sigma_{\text{
        update}}(y))`.
        """
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
                action=self.actions,
                observation=self.observations,
            )
            # Solution estimate update
            x_mean = self.belief.x.mean + step_size * self.actions

            # Update residual
            self._residual(
                residual=residual,
                step_size=step_size,
                observation=self.observations,
            )
            return x_mean, None
        else:
            raise NotImplementedError

    @cached_property
    def A(self) -> rvs.Normal:
        mean_update, cov_update = self.A_update_terms
        A_mean = linops.aslinop(self.belief.A.mean) + mean_update
        A_covfactor = linops.aslinop(self.belief.A.cov.A) - cov_update

        return rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

    @cached_property
    def A_update_terms(
        self,
    ) -> Tuple[
        Union[np.ndarray, linops.LinearOperator],
        Union[np.ndarray, linops.LinearOperator],
    ]:
        u, v, Ws = self._matrix_model_update_components(
            belief_matrix=self.belief.Ainv,
            action=self.actions,
            observation=self.observations,
        )
        # Rank 2 mean update (+= uv' + vu')
        mean_update = self._matrix_model_mean_update_op(u=u, v=v)
        # Rank 1 covariance Kronecker factor update (-= u(Ws)')
        cov_update = self._matrix_model_covariance_factor_update_op(u=u, Ws=Ws)

        return mean_update, cov_update

    @cached_property
    def Ainv(self) -> rvs.Normal:
        mean_update, cov_update = self.Ainv_update_terms
        Ainv_mean = linops.aslinop(self.belief.A.mean) + mean_update
        Ainv_covfactor = linops.aslinop(self.belief.A.cov.A) - cov_update

        return rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

    def Ainv_update_terms(
        self,
    ) -> Tuple[
        Union[np.ndarray, linops.LinearOperator],
        Union[np.ndarray, linops.LinearOperator],
    ]:
        u, v, Wy = self._matrix_model_update_components(
            belief_matrix=self.belief.Ainv,
            action=self.observations,
            observation=self.actions,
        )
        # Rank 2 mean update (+= uv' + vu')
        mean_update = self._matrix_model_mean_update_op(u=u, v=v)
        # Rank 1 covariance Kronecker factor update (-= u(Wy)')
        cov_update = self._matrix_model_covariance_factor_update_op(u=u, Ws=Wy)
        return mean_update, cov_update

    @cached_property
    def b(self) -> Union[rvs.Normal, rvs.Constant]:
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
                self.solver_state.action_obs_innerprods.append(step_size)
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
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: np.ndarray,
        observations: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ):

        super().__init__(
            problem=problem,
            belief=belief,
            actions=actions,
            observations=observations,
            noise_cov=None,
            solver_state=solver_state,
        )

    @cached_property
    def A(self) -> rvs.Normal:
        # TODO implement this under the assumption that W = A
        raise NotImplementedError

    @cached_property
    def Ainv(self) -> rvs.Normal:
        raise NotImplementedError

    def _A_cov_trace_update(self):
        A.trace = None
        return A

    def _Ainv_cov_trace_update(self):
        raise NotImplementedError

    def _x_cov_trace_update(self):
        raise NotImplementedError
