"""Hyperparameter optimization routines for probabilistic linear solvers."""

from typing import Callable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["HyperparameterOptimization", "UncertaintyCalibration", "OptimalNoiseScale"]


class HyperparameterOptimization:
    """Optimization of hyperparameters of probabilistic linear solvers."""

    def __init__(
        self,
        hyperparam_optim: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
            Tuple[
                Tuple[np.ndarray, ...],
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
        ],
    ):
        self._hyperparam_optim = hyperparam_optim

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """Return an action based on the given problem and model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief
            Belief over the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        optimal_hyperparams
            Optimized hyperparameters.
        belief
            Updated belief over the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Updated solver state.
        """
        return self._hyperparam_optim(problem, belief, solver_state)


class UncertaintyCalibration(HyperparameterOptimization):
    """Calibrate the uncertainty of the covariance class."""

    def __init__(self):
        super().__init__(hyperparam_optim=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """"""
        raise NotImplementedError

    def _calibrate_uncertainty(
        self, n: int, log_rayleigh_quotients: np.ndarray, method: str
    ):
        r"""Calibrate uncertainty based on the Rayleigh coefficients.

        A regression model for the log-Rayleigh coefficient is built based on the
        collected observations. The degrees of freedom in the kernels of A and H are set
        according to the predicted log-Rayleigh coefficient for the remaining unexplored
        dimensions.

        Parameters
        ----------
        S : np.ndarray, shape=(n, k)
            Array of search directions
        sy : np.ndarray
            Array of inner products :math:`s_i^\top As_i``
        method : str
            Type of calibration method to use based on the Rayleigh quotient. Available
            calibration procedures are
            ====================================  ==================
             Most recent Rayleigh quotient         ``adhoc``
             Running (weighted) mean               ``weightedmean``
             GP regression for kernel matrices     ``gpkern``
            ====================================  ==================

        Returns
        -------
        phi : float
            Uncertainty scale of the null space of span(S) for the A view
        psi : float
            Uncertainty scale of the null space of span(Y) for the Ainv view
        """

        # Rayleigh quotient
        n_iterations = len(log_rayleigh_quotients)
        iters = np.arange(n_iterations + 1)

        # only calibrate if enough iterations for a regression model have been performed
        if n_iterations > 1:
            if method == "adhoc":
                logR_pred = log_rayleigh_quotients[-1]
            elif method == "weightedmean":
                deprecation_rate = 0.9
                logR_pred = log_rayleigh_quotients * np.repeat(
                    deprecation_rate, n_iterations + 1
                ) ** np.arange(n_iterations + 1)
            elif method == "gpkern":
                try:
                    import GPy  # pylint: disable=import-outside-toplevel

                    # GP mean function via Weyl's result on spectra of Gram matrices for
                    # differentiable kernels
                    # ln(sigma(n)) ~= theta_0 - theta_1 ln(n)
                    lnmap = GPy.core.Mapping(1, 1)
                    lnmap.f = lambda n: np.log(n + 10 ** -16)
                    lnmap.update_gradients = lambda a, b: None
                    mf = GPy.mappings.Additive(
                        GPy.mappings.Constant(1, 1, value=0),
                        GPy.mappings.Compound(lnmap, GPy.mappings.Linear(1, 1)),
                    )
                    k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
                    m = GPy.models.GPRegression(
                        iters[:, None] + 1,
                        log_rayleigh_quotients[:, None],
                        kernel=k,
                        mean_function=mf,
                    )
                    m.optimize(messages=False)

                    # Predict Rayleigh quotient
                    remaining_dims = np.arange(n_iterations, n)[:, None]
                    logR_pred = m.predict(remaining_dims + 1)[0].ravel()
                except ImportError as err:
                    raise ImportError(
                        "Cannot perform GP-based calibration without optional "
                        "dependency GPy. Try installing GPy via `pip install GPy`."
                    ) from err
            else:
                raise ValueError("Calibration method not recognized.")

            # Set uncertainty scale (degrees of freedom in calibration covariance class)
            Phi = (np.exp(np.mean(logR_pred))).item()
            Psi = (np.exp(-np.mean(logR_pred))).item()
        else:
            # For too few iterations take the most recent Rayleigh quotient
            Phi = np.exp(log_rayleigh_quotients[-1])
            Psi = 1 / Phi

        return Phi, Psi

    def _get_calibration_covariance_update_terms(self, phi=None, psi=None):
        """For the calibration covariance class set the calibration update terms of the
        covariance in the null spaces of span(S) and span(Y) based on the degrees of
        freedom."""
        # Search directions and observations as arrays
        S = np.hstack(self.search_dir_list)
        Y = np.hstack(self.obs_list)

        def get_null_space_map(V, unc_scale):
            """Returns a function mapping to the null space of span(V), scaling with a
            single degree of freedom and mapping back."""

            def null_space_proj(x):
                try:
                    VVinvVx = np.linalg.solve(V.T @ V, V.T @ x)
                    return x - V @ VVinvVx
                except np.linalg.LinAlgError:
                    return np.zeros_like(x)

            # For a scalar uncertainty scale projecting to the null space twice is
            # equivalent to projecting once
            return lambda y: unc_scale * null_space_proj(y)

        # Compute calibration term in the A view as a linear operator with scaling from
        # degrees of freedom
        calibration_term_A = linops.LinearOperator(
            shape=(self.n, self.n), matvec=get_null_space_map(V=S, unc_scale=phi)
        )

        # Compute calibration term in the Ainv view as a linear operator with scaling
        # from degrees of freedom
        calibration_term_Ainv = linops.LinearOperator(
            shape=(self.n, self.n), matvec=get_null_space_map(V=Y, unc_scale=psi)
        )

        return calibration_term_A, calibration_term_Ainv


class OptimalNoiseScale(HyperparameterOptimization):
    """Estimate the noise level of a noisy linear system."""

    def __init__(self):
        super().__init__(hyperparam_optim=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[np.ndarray, ...],
        "probnum.linalg.linearsolvers.LinearSystemBelief",
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """"""
        raise NotImplementedError
