from functools import cached_property
from typing import Optional, Tuple, Union

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.belief_updates._belief_update import BeliefUpdate
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["SymmetricNormalLinearObsBeliefUpdate"]


class SymmetricNormalLinearObsBeliefUpdate(BeliefUpdate):
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
    actions :
        Actions to probe the linear system with.
    observations :
        Observations of the linear system for the given actions.
    solver_state :
        Current state of the linear solver.
    noise_cov :
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
        """Updated Gaussian belief over the solution :math:`x` of the linear system."""
        if self.noise_cov is None:
            x_mean_update, _ = self.x_update_terms(belief_x=self.belief.x)
            return rvs.Normal(
                mean=self.belief.x.mean + x_mean_update,
                cov=self.belief._induced_solution_cov(Ainv=self.Ainv, b=self.b),
            )
        else:
            return self.belief._induced_solution_belief(Ainv=self.Ainv, b=self.b)

    def x_update_terms(
        self,
        belief_x: rvs.Normal,
    ) -> Tuple[np.ndarray, Union[np.ndarray, linops.LinearOperator]]:
        r"""Mean and covariance update terms for the solution.

        For a prior belief :math:`\mathsf{x} \sim \mathcal{N}(\mu, \Sigma)`, computes
        the update terms :math:`\mu_{\text{update}}(y)` and
        :math:`\Sigma_{\text{update}}(y)` given observations :math:`y`, such that
        :math:`\mathsf{x} \mid y \sim \mathcal{N}(\mu +\mu_{\text{update}}(y), \Sigma -
        \Sigma_{\text{update}}(y))`.

        Parameters
        ----------
        belief_x : Belief over the solution of the linear system.
        """
        if self.noise_cov is None:
            # Current residual
            try:
                residual = self.solver_state.residual
            except AttributeError:
                residual = self.problem.A @ belief_x.mean - self.problem.b
                if self.solver_state is not None:
                    self.solver_state.residual = residual

            # Step size
            step_size = self._step_size(
                residual=residual,
                action=self.actions,
                observation=self.observations,
            )
            # Solution estimate update
            x_mean_update = step_size * self.actions

            # Update residual
            self._residual(
                residual=residual,
                step_size=step_size,
                observation=self.observations,
            )
            return x_mean_update, None
        else:
            raise NotImplementedError

    @cached_property
    def A(self) -> rvs.Normal:
        """Updated Gaussian belief over the system matrix :math:`A`."""
        mean_update, cov_update = self.A_update_terms(belief_A=self.belief.A)
        A_mean = linops.aslinop(self.belief.A.mean) + mean_update
        A_covfactor = linops.aslinop(self.belief.A.cov.A) - cov_update

        return rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A_covfactor))

    def A_update_terms(
        self,
        belief_A: rvs.Normal,
    ) -> Tuple[
        Union[np.ndarray, linops.LinearOperator],
        Union[np.ndarray, linops.LinearOperator],
    ]:
        r"""Mean and covariance update terms for the system matrix.

        For a prior belief :math:`\mathsf{A} \sim \mathcal{N}(A_0, W \otimes_s W)`,
        computes the update terms :math:`A_{\text{update}}(y)` and
        :math:`W_{\text{update}}(y)` given observations :math:`y`, such that
        :math:`\mathsf{A} \mid y \sim \mathcal{N}\big(A_0 +A_{\text{update}}(y), (W -
        W_{\text{update}}(y)) \otimes_s (W - W_{\text{update}}(y))\big)`.

        Parameters
        ----------
        belief_A : Belief over the system matrix.
        """
        u, v, Ws = self._matrix_model_update_components(
            belief_matrix=belief_A,
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
        """Updated Gaussian belief over the inverse of the system matrix
        :math:`H=A^{-1}`."""
        mean_update, cov_update = self.Ainv_update_terms(belief_Ainv=self.belief.Ainv)
        Ainv_mean = linops.aslinop(self.belief.Ainv.mean) + mean_update
        Ainv_covfactor = linops.aslinop(self.belief.Ainv.cov.A) - cov_update

        return rvs.Normal(mean=Ainv_mean, cov=linops.SymmetricKronecker(Ainv_covfactor))

    def Ainv_update_terms(
        self, belief_Ainv: rvs.Normal
    ) -> Tuple[
        Union[np.ndarray, linops.LinearOperator],
        Union[np.ndarray, linops.LinearOperator],
    ]:
        r"""Mean and covariance update terms for the inverse.

        For a prior belief :math:`\mathsf{H} \sim \mathcal{N}(H_0, W \otimes_s W)`,
        computes the update terms :math:`H_{\text{update}}(y)` and
        :math:`W_{\text{update}}(y)` given observations :math:`y`, such that
        :math:`\mathsf{H} \mid y \sim \mathcal{N}\big(H_0 +H_{\text{update}}(y), (W -
        W_{\text{update}}(y)) \otimes_s (W - W_{\text{update}}(y))\big)`.

        Parameters
        ----------
        belief_Ainv : Belief over the inverse.
        """
        u, v, Wy = self._matrix_model_update_components(
            belief_matrix=belief_Ainv,
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
        """Updated belief over the right hand side :math:`b` of the linear system."""
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

    @staticmethod
    def _matrix_model_update_components(
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

    @staticmethod
    def _matrix_model_mean_update_op(
        u: np.ndarray, v: np.ndarray
    ) -> linops.LinearOperator:
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' +
        vu')."""

        def mv(x):
            return u * (v.T @ x) + v * (u.T @ x)

        def mm(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mm
        )

    @staticmethod
    def _matrix_model_covariance_factor_update_op(
        u: np.ndarray, Ws: np.ndarray
    ) -> linops.LinearOperator:
        """Linear operator implementing the symmetric rank 2 covariance factor downdate
        (-= Ws u')."""

        def mv(x):
            return Ws * (u.T @ x)

        def mm(x):
            return Ws @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mm
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
        if self.solver_state is not None:
            try:
                action_obs_innerprod = self.solver_state.action_obs_innerprods[
                    self.solver_state.iteration
                ]
            except IndexError:
                action_obs_innerprod = (action.T @ observation).item()
        else:
            action_obs_innerprod = (action.T @ observation).item()
        step_size = (-action.T @ residual / action_obs_innerprod).item()

        # Update solver state
        if self.solver_state is not None:
            self.solver_state.action_obs_innerprods.append(action_obs_innerprod)
            self.solver_state.step_sizes.append(step_size)
            self.solver_state.log_rayleigh_quotients.append(
                _log_rayleigh_quotient(
                    action_obs_innerprod=action_obs_innerprod, action=action
                )
            )

        return step_size


def _log_rayleigh_quotient(action_obs_innerprod: float, action: np.ndarray) -> float:
    r"""Compute the log-Rayleigh quotient :math:`\ln R(A, s_i) = \ln(s_i^\top A
    s_i) -\ln(s_i^\top s_i)` for the current action."""
    return (np.log(action_obs_innerprod) - np.log(action.T @ action)).item()
