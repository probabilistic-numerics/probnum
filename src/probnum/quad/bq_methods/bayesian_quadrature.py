
from typing import Tuple, Dict


class BayesianQuadrature:
    """
    An abstract base class for Bayesian quadrature methods.

    This class is designed to be subclassed by implementations of Bayesian quadrature
    with an :meth:`integrate` method.
    """

    def integrate(self, fun, fun0, domain, nevals, **kwargs):
        """
        Integrate the function ``fun``.

        Parameters
        ----------
        fun : function
            Function to be integrated.
        fun0 : RandomProcess
            Stochastic process modelling function to be integrated.
        domain : ndarray, shape=()
            Domain to integrate over.
        nevals : int
            Number of function evaluations.
        kwargs

        Returns
        -------
        F : RandomVariable
        The integral of ``func`` from ``a`` to ``b``.
        fun0 : RandomProcess
            Stochastic process modelling the function to be integrated after ``neval``
            observations.
        info : dict
            Information on the performance of the method.
        """

        # Initialization
        F = None

        # Iteration
        # before we have an acquisition function, we choose locations at random

        for _ in range(nevals):
            # Predictive Distribution
            # fun_pred = None

            # Observation

            # Update Distribution
            self.fun0 = None

        # Information on result
        info = {"model_fit_diagnostic": None}

        return F, self.fun0, info