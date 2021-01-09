"""Hyperparameter optimization routines for probabilistic linear solvers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy

import probnum  # pylint: disable="unused-import
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["HyperparameterOptimization", "UncertaintyCalibration", "OptimalNoiseScale"]

# pylint: disable="invalid-name"


class HyperparameterOptimization(ABC):
    """Optimization of hyperparameters of probabilistic linear solvers."""

    @abstractmethod
    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        """Optimized hyperparameters of the linear system model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        actions :
            Actions of the solver to probe the linear system with.
        observations :
            Observations of the linear system for the given actions.
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
        raise NotImplementedError


class UncertaintyCalibration(HyperparameterOptimization):
    """Calibrate uncertainty of the covariance class based on Rayleigh coefficients.

    A regression model for the log-Rayleigh coefficient is built based on the
    collected observations. The degrees of freedom in the covariance class of the
    models for :math:`A` and :math:`H` are set according to the predicted
    log-Rayleigh coefficient for the remaining unexplored dimensions.

    Parameters
    ----------
    method :
        If supplied calibrates the output via the given calibration method. Available
        procedures are

        ====================================  ================
         Most recent Rayleigh quotient        ``adhoc``
         Running (weighted) mean              ``weightedmean``
         GP regression for kernel matrices    ``gpkern``
        ====================================  ================

    Examples
    --------
    """

    def __init__(self, method: str = "gpkern"):
        self.calib_method = method
        super().__init__()

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:
        iteration = len(actions)
        log_rayleigh_quotients = None
        if solver_state is not None:
            if solver_state.log_rayleigh_quotients is not None:
                log_rayleigh_quotients = solver_state.log_rayleigh_quotients

        if log_rayleigh_quotients is None:
            S = np.hstack(actions)
            Y = np.hstack(observations)
            log_rayleigh_quotients = np.log(np.einsum("nk,nk->k", S, Y)) - np.log(
                np.einsum("nk,nk->k", S, S)
            )

        if iteration == 1:
            # For too few iterations take the most recent Rayleigh quotient
            unc_scale_A = np.exp(log_rayleigh_quotients[-1])
            unc_scale_Ainv = np.exp(-log_rayleigh_quotients[-1])
        else:
            # Select calibration method
            if self.calib_method == "adhoc":
                logR_pred = self._most_recent_log_rayleigh_quotient(
                    log_rayleigh_quotients=log_rayleigh_quotients
                )
            elif self.calib_method == "weightedmean":
                logR_pred = self._weighted_average_log_rayleigh_quotients(
                    log_rayleigh_quotients=log_rayleigh_quotients, iteration=iteration
                )
            elif self.calib_method == "gpkern":
                logR_pred = self._gp_regression_log_rayleigh_quotients(
                    log_rayleigh_quotients=log_rayleigh_quotients,
                    iteration=iteration,
                    n=problem.A.shape[0],
                )
            else:
                raise ValueError("Calibration method not recognized.")
            # Set uncertainty scale (degrees of freedom in calibration covariance class)
            unc_scale_A = (np.exp(np.mean(logR_pred))).item()
            unc_scale_Ainv = (np.exp(-np.mean(logR_pred))).item()

        return (unc_scale_A, unc_scale_Ainv), solver_state

    def _most_recent_log_rayleigh_quotient(self, log_rayleigh_quotients: List[float]):
        """Most recent log-Rayleigh quotient."""
        return log_rayleigh_quotients[-1]

    def _weighted_average_log_rayleigh_quotients(
        self, iteration: int, log_rayleigh_quotients: List[float]
    ):
        """Weighted average of log-Rayleigh quotients."""
        deprecation_rate = 0.9
        return log_rayleigh_quotients * np.repeat(
            deprecation_rate, iteration
        ) ** np.arange(iteration)

    def _gp_regression_log_rayleigh_quotients(
        self, iteration: int, n: int, log_rayleigh_quotients: List[float]
    ):
        r"""GP regression on log-Rayleigh quotients.

        Assumes the system matrix to be generated by a :math:`\nu`-times differentiable
        kernel. By Weyl's theorem [1]_ the spectra of such kernel matrices approximately
        decay as :math:`\mathcal{O}(n^{-\nu-\frac{1}{2}})`. Accordingly, the prior mean
        function of the Gaussian process is chosen as :math:`\mu(n) = \log(
        \theta_0' n^{-\theta_1}) = \theta_0 - \theta_1 \log(n)`.

        References
        ----------
        .. [1] Weyl, Hermann. Das asymptotische Verteilungsgesetz der Eigenwerte
           linearer partieller Differentialgleichungen (mit einer Anwendung auf die
           Theorie der Hohlraumstrahlung). *Mathematische Annalen*, 71(4):441–479, 1912.
        """
        try:
            import GPy  # pylint: disable=import-outside-toplevel

            iters = np.arange(iteration) + 1
            # GP mean function via Weyl's result on spectra of Gram matrices for
            # differentiable kernels
            # ln(sigma(n)) ~= theta_0 - theta_1 ln(n)
            lnmap = GPy.core.Mapping(1, 1)
            lnmap.f = lambda n: np.log(n + np.finfo(np.float64).eps)
            lnmap.update_gradients = lambda a, b: None
            mf = GPy.mappings.Additive(
                GPy.mappings.Constant(1, 1, value=0),
                GPy.mappings.Compound(lnmap, GPy.mappings.Linear(1, 1)),
            )
            k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
            m = GPy.models.GPRegression(
                iters[:, None],
                log_rayleigh_quotients[:, None],
                kernel=k,
                mean_function=mf,
            )
            m.optimize(messages=False)

            # Predict Rayleigh quotient
            remaining_dims = np.arange(iteration, n)[:, None] + 1
            return m.predict(remaining_dims)[0].ravel()
        except ImportError as err:
            raise ImportError(
                "Cannot perform GP-based calibration without optional "
                "dependency GPy. Try installing GPy via `pip install GPy`."
            ) from err


class OptimalNoiseScale(HyperparameterOptimization):
    r"""Estimate the noise level of a noisy linear system.

    Computes the optimal noise scale maximizing the log-marginal likelihood.
    """

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[
        Tuple[Union[np.ndarray, float], ...],
        Optional["probnum.linalg.linearsolvers.LinearSolverState"],
    ]:

        raise NotImplementedError

    @staticmethod
    def _optimal_noise_scale_iterative(
        previous_optimal_noise_scale: float,
        problem: LinearSystem,
        belief: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        action: np.ndarray,
        observation: np.ndarray,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _optimal_noise_scale_batch(
        problem: LinearSystem,
        prior: "probnum.linalg.linearsolvers.beliefs.LinearSystemBelief",
        actions: np.ndarray,
        observations: np.ndarray,
    ) -> float:
        # Compute intermediate quantities
        Delta0 = observations - prior.A.mean @ actions
        SW0S = actions.T @ (prior.A.cov.A @ actions)
        try:
            SW0SinvSDelta0 = scipy.linalg.solve(
                SW0S, actions.T @ Delta0, assume_a="pos"
            )  # solves k x k system k times: O(k^3)
            linop_rhs = Delta0.T @ (
                2 * prior.A.cov.A.inv() @ Delta0 - actions @ SW0SinvSDelta0
            )
            linop_tracearg = scipy.linalg.solve(
                SW0S, linop_rhs, assume_a="pos"
            )  # solves k x k system k times: O(k^3)

            # Optimal noise scale with respect to the evidence
            noise_scale_estimate = (
                linop_tracearg.trace() / (problem.A.shape[0] * actions.shape[1]) - 1
            )
        except scipy.linalg.LinAlgError as err:
            raise scipy.linalg.LinAlgError(
                "Matrix S'W_0S not invertible. Noise scale estimate may be inaccurate."
            ) from err

        return noise_scale_estimate if noise_scale_estimate > 0.0 else 0.0
