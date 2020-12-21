"""Belief updates for probabilistic linear solvers."""

import numpy as np

import probnum  # pylint: disable="unused-import
import probnum.linops as linops
import probnum.random_variables as rvs


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

    def __init__(self, belief_update):
        self._belief_update = belief_update

    def __call__(
        self, solver_state: "probnum.linalg.linearsolvers.LinearSolverState"
    ) -> "probnum.linalg.linearsolvers" ".LinearSolverState":
        """Update belief over quantities of interest of the linear system.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        return self._belief_update(solver_state)


class LinearGaussianBeliefUpdate(BeliefUpdate):
    """Belief update assuming (symmetric) Gaussianity and linear observations."""

    def __init__(self):
        super().__init__(belief_update=self.__call__)

    def __call__(
        self, solver_state: "probnum.linalg.linearsolvers.LinearSolverState"
    ) -> "probnum.linalg.linearsolvers" ".LinearSolverState":

        # TODO: refactor this method by splitting it up into a function for each of
        #  the individual updates

        # Setup
        x_mean = solver_state.belief[0].mean
        A_mean = solver_state.belief[1].mean
        A_covfactor = solver_state.belief[1].cov.A
        Ainv_mean = solver_state.belief[2].mean
        Ainv_covfactor = solver_state.belief[2].cov.A
        action = solver_state.actions[-1]
        observation = solver_state.observations[-1]
        residual = solver_state.residual

        # Compute step size
        sy = action.T @ observation
        step_size = -np.squeeze((action.T @ residual) / sy)

        # Step and residual update
        x_mean = x_mean + step_size * action
        solver_state.residual = residual + step_size * observation

        # (Symmetric) mean and covariance updates
        Vs = A_covfactor @ action
        delta_A = observation - A_mean @ action
        u_A = Vs / (action.T @ Vs)
        v_A = delta_A - 0.5 * (action.T @ delta_A) * u_A

        Wy = Ainv_covfactor @ observation
        delta_Ainv = action - Ainv_mean @ observation
        yWy = np.squeeze(observation.T @ Wy)
        u_Ainv = Wy / yWy
        v_Ainv = delta_Ainv - 0.5 * (observation.T @ delta_Ainv) * u_Ainv

        # Rank 2 mean updates (+= uv' + vu')
        A_mean = linops.aslinop(A_mean) + self._mean_update(u=u_A, v=v_A)
        Ainv_mean = linops.aslinop(Ainv_mean) + self._mean_update(u=u_Ainv, v=v_Ainv)

        # Rank 1 covariance Kronecker factor update (-= u_A(Vs)' and -= u_Ainv(Wy)')
        if solver_state.iteration == 0:
            _A_covfactor_update_term = self._covariance_factor_update(u=u_A, Ws=Vs)
            _Ainv_covfactor_update_term = self._covariance_factor_update(
                u=u_Ainv, Ws=Wy
            )
        else:
            _A_covfactor_update_term = (
                _A_covfactor_update_term + self._covariance_factor_update(u=u_A, Ws=Vs)
            )
            _Ainv_covfactor_update_term = (
                _Ainv_covfactor_update_term
                + self._covariance_factor_update(u=u_Ainv, Ws=Wy)
            )
        A_covfactor = linops.aslinop(self.A_covfactor0) - _A_covfactor_update_term
        Ainv_covfactor = (
            linops.aslinop(self.Ainv_covfactor0) - _Ainv_covfactor_update_term
        )

        # Update solver state with new beliefs
        solver_state.belief = (
            rvs.Constant(x_mean),
            rvs.Normal(A_mean, linops.SymmetricKronecker(A_covfactor)),
            rvs.Normal(Ainv_mean, linops.SymmetricKronecker(Ainv_covfactor)),
            solver_state.belief[3],
        )

        return solver_state

    def _update_residual(self, step_size: float, observation: np.ndarray) -> np.ndarray:
        """Update the residual :math:`r_i = Ax_i - b`."""
        raise NotImplementedError

    def _update_solution(self, step_size: float, action: np.ndarray) -> np.ndarray:
        """Update the solution :math:`x_i` to the linear system."""
        raise NotImplementedError

    def _update_matrix(self, A_mean, A_covfactor, action, observation) -> rvs.Normal:
        """Update the belief over the system matrix :math:`A`."""
        raise NotImplementedError

    def _update_inverse(
        self, Ainv_mean, Ainv_covfactor, action, observation
    ) -> rvs.Normal:
        """Update the belief over the inverse of the system matrix :math:`H=A^{-1}`."""
        raise NotImplementedError

    def _mean_update(self, u, v):
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' +
        vu')."""

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )

    def _covariance_factor_update(self, u, Ws):
        """Linear operator implementing the symmetric rank 2 covariance factor downdate
        (-= Ws u^T)."""

        def mv(x):
            return Ws @ (u.T @ x)

        return linops.LinearOperator(
            shape=(u.shape[0], u.shape[0]), matvec=mv, matmat=mv
        )
