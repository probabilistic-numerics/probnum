"""Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian
quadrature, meaning a model over the integrand is constructed in order
to actively select evaluation points of the integrand to estimate the
value of the integral. Bayesian quadrature methods return a random
variable, specifying the belief about the true value of the integral.
"""

from .bq_methods import BayesianQuadrature, VanillaBayesianQuadrature


def bayesquad(fun, fun0, domain, nevals=None, measure=None, method="vanilla"):
    """Bayesian quadrature (BQ) infers integrals of the form

    .. math:: F = \\int_a^b f(x) d \\mu(x),

    of a function :math: `f:\\mathbb{R}^n \\mapsto \\mathbb{R}` integrated between bounds
    :math: `a` and :math: `b` against a measure :math: `\\mu: \\mathbb{R}^n \\mapsto \\mathbb{R}`.

    Bayesian quadrature methods return a probability distribution over the solution :math: `F` with
    uncertainty arising from finite computation (here a finite number of function evaluations).
    They start out with a random process encoding the prior belief about the function :math: `f`
    to be integrated. Conditioned on either existing or acquired function evaluations according to a
    policy, they update the belief on :math: `f`, which is translated into a posterior measure over
    the integral :math: `F`.

    Parameters
    ----------
    fun : function
        Function to be integrated.
    fun0 : RandomProcess or function, optional
        Stochastic process modelling the function to be integrated.
    domain : Tuple
        Domain of integration. Contains lower and upper bound as int or ndarray, shape=(dim,)
    measure :
        Measure to integrate against.
    nevals :
        Number of function evaluations.
    measure: IntegrationMeasure, optional
        Integration measure, defaults to the Lebesgue measure.
    method : str, optional
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WSABI                ``wsabi``
        ====================  ===========

    Returns
    -------
    integral :
        The integral of ``func`` on the domain.
    fun0 :
        Stochastic process modelling the function to be integrated after ``neval``
        observations.
    info :
        Information on the performance of the method.

    References
    ----------
    """

    # Choose Method
    bqmethod = None
    if method == "vanilla":
        bqmethod = BayesianQuadrature(fun0=fun0)
    # elif method == "wsabi":
    #     bqmethod = WarpedBayesianQuadrature(fun0=fun0)

    # Integrate
    integral, fun0, info = bqmethod.integrate(
        fun=fun, nevals=nevals, domain=domain, measure=measure
    )

    return integral, fun0, info
