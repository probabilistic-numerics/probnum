"""Implementation of Bayesian Quadrature."""


class BayesianQuadrature:
    """Bayesian quadrature.

    Bayesian quadrature methods build a model for the integrand via function
    evaluations and return a belief over the value of the integral on a given
    domain with respect to the specified measure.

    Parameters
    ----------
    fun0
        Stochastic process modelling function to be integrated.
    """

    def __init__(self, fun0):
        self.fun0 = fun0

    def integrate(self, fun, domain, measure, nevals):
        """Integrate the function ``fun``.

        Parameters
        ----------
        fun :
            Function to be integrated.
        domain :
            Domain to integrate over.
        measure :
            Measure to integrate against.
        nevals :
            Number of function evaluations.

        Returns
        -------
        """
        raise NotImplementedError
