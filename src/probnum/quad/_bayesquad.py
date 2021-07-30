"""Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian
quadrature, meaning a model over the integrand is constructed in order
to actively select evaluation points of the integrand to estimate the
value of the integral. Bayesian quadrature methods return a random
variable, specifying the belief about the true value of the integral.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from probnum.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import FloatArgType

from ._integration_measures import IntegrationMeasure, LebesgueMeasure
from .bq_methods import BayesianQuadrature


# pylint: disable=too-many-arguments
def bayesquad(
    fun: Callable,
    input_dim: int,
    kernel: Optional[Kernel] = None,
    domain: Optional[
        Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]]
    ] = None,
    nevals: int = None,
    measure: Optional[IntegrationMeasure] = None,
    method: str = "vanilla",
    policy: str = "bmc",
    rng: np.random.Generator = None,
) -> Tuple[Normal, Dict]:
    r"""Infer the solution of the uni- or multivariate integral :math:`\int_\Omega f(x) d \mu(x)`
    on a hyper-rectangle :math:`\Omega = [a_1, b_1] \times \cdots \times [a_D, b_D]`.

    Bayesian quadrature (BQ) infers integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x),

    of a function :math:`f:\mathbb{R}^D \mapsto \mathbb{R}` integrated on the domain
    :math:`\Omega \subset \mathbb{R}^D` against a measure :math:`\mu: \mathbb{R}^D
    \mapsto \mathbb{R}`.

    Bayesian quadrature methods return a probability distribution over the solution :math:`F` with
    uncertainty arising from finite computation (here a finite number of function evaluations).
    They start out with a random process encoding the prior belief about the function :math:`f`
    to be integrated. Conditioned on either existing or acquired function evaluations according to a
    policy, they update the belief on :math:`f`, which is translated into a posterior measure over
    the integral :math:`F`.
    See Briol et al. [1]_ for a review on Bayesian quadrature.

    Parameters
    ----------
    fun :
        Function to be integrated.
    input_dim:
        Input dimension of the integration problem
    kernel:
        the kernel used for the GP model
    domain :
        *shape=(dim,)* -- Domain of integration. Contains lower and upper bound as int or ndarray.
    measure:
        Integration measure, defaults to the Lebesgue measure.
    nevals :
        Number of function evaluations.
    method :
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WSABI                ``wsabi``
        ====================  ===========

    policy :
        Type of acquisition strategy to use. Options are

        =======================  =======
         Bayesian Monte Carlo    ``bmc``
         Uncertainty Sampling    ``us``
         Mutual Information      ``mi``
         Integral Variance       ``iv``
        =======================  =======

    rng :
        Random number generator. Required for Bayesian Monte Carlo policies.
        Optional. Default is `np.random.default_rng()`.

    Returns
    -------
    integral :
        The integral of ``func`` on the domain.
    info :
        Information on the performance of the method.

    References
    ----------
    .. [1] Briol, F.-X., et al., Probabilistic integration: A role in statistical computation?,
       *Statistical Science 34.1*, 2019, 1-22, 2019

    Examples
    --------
    >>> import numpy as np

    >>> input_dim = 1
    >>> domain = (0, 1)
    >>> def f(x):
    ...     return x
    >>> F, info = bayesquad(fun=f, input_dim=input_dim, domain=domain)
    >>> print(F.mean)
    0.5
    """
    if domain is None and measure is None:
        raise ValueError(
            "You need to either specify an integration domain or an integration "
            "measure. The Lebesgue measure can only operate on a finite domain."
        )

    # Integration measure
    if measure is None:
        measure = LebesgueMeasure(domain=domain, dim=input_dim)

    # Choose Method
    if policy == "bmc" and rng is None:
        rng = np.random.default_rng()
    bq_method = BayesianQuadrature.from_interface(
        input_dim=input_dim, kernel=kernel, method=method, policy=policy, rng=rng
    )

    if nevals is None:
        nevals = input_dim * 25

    # Integrate
    integral, info = bq_method.integrate(fun=fun, measure=measure, nevals=nevals)

    return integral, info
