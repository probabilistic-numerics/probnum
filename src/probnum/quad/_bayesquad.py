"""Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature,
meaning a model over the integrand is constructed in order to actively select evaluation
points of the integrand to estimate the value of the integral. Bayesian quadrature
methods return a random variable, specifying the belief about the true value of the
integral.
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike, IntLike

from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure
from .solvers import BayesianQuadrature


# pylint: disable=too-many-arguments, no-else-raise
def bayesquad(
    fun: Callable,
    input_dim: int,
    kernel: Optional[Kernel] = None,
    domain: Optional[
        Union[Tuple[FloatLike, FloatLike], Tuple[np.ndarray, np.ndarray]]
    ] = None,
    measure: Optional[IntegrationMeasure] = None,
    policy: Optional[str] = "bmc",
    max_evals: Optional[IntLike] = None,
    var_tol: Optional[FloatLike] = None,
    rel_tol: Optional[FloatLike] = None,
    batch_size: Optional[IntLike] = 1,
    rng: Optional[np.random.Generator] = np.random.default_rng(),
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
        Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
        ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
    input_dim :
        Input dimension of the integration problem.
    kernel :
        The kernel used for the GP model
    domain :
        *shape=(input_dim,)* -- Domain of integration. Contains lower and upper bound as
        ``int`` or ``np.ndarray``.
    measure:
        Integration measure. Defaults to the Lebesgue measure.
    policy :
        Type of acquisition strategy to use. Options are

        ==========================  =======
         Bayesian Monte Carlo [2]_  ``bmc``
        ==========================  =======

    max_evals :
        Maximum number of function evaluations.
    var_tol :
        Tolerance on the variance of the integral.
    rel_tol :
        Tolerance on consecutive updates of the integral mean.
    batch_size :
        Number of new observations at each update.
    rng :
        Random number generator. Used by Bayesian Monte Carlo other random sampling
        policies. Optional. Default is `np.random.default_rng()`.

    Returns
    -------
    integral :
        The integral of ``fun`` on the domain.
    info :
        Information on the performance of the method.

    Raises
    ------
    ValueError
        If neither a domain nor a measure are given.
    ValueError
        If a domain is given with a Gaussian measure.

    References
    ----------
    .. [1] Briol, F.-X., et al., Probabilistic integration: A role in statistical
        computation?, *Statistical Science 34.1*, 2019, 1-22, 2019
    .. [2] Rasmussen, C. E., and Z. Ghahramani, Bayesian Monte Carlo, *Advances in
        Neural Information Processing Systems*, 2003, 505-512.

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

    # Check input argument compatibility
    if domain is None and measure is None:
        raise ValueError(
            "You need to either specify an integration domain or an integration "
            "measure. The Lebesgue measure can only operate on a finite domain."
        )

    if domain is not None:
        if isinstance(measure, GaussianMeasure):
            raise ValueError("GaussianMeasure cannot be used with finite bounds.")
        elif isinstance(measure, LebesgueMeasure):
            warnings.warn(
                "Both domain and a LebesgueMeasure are specified. The domain "
                "information will be ignored."
            )

    bq_method = BayesianQuadrature.from_problem(
        input_dim=input_dim,
        kernel=kernel,
        measure=measure,
        domain=domain,
        policy=policy,
        max_evals=max_evals,
        var_tol=var_tol,
        rel_tol=rel_tol,
        batch_size=batch_size,
        rng=rng,
    )

    # Integrate
    integral_belief, bq_state = bq_method.integrate(fun=fun)

    return integral_belief, bq_state.info


def bayesquad_from_data(
    nodes: np.ndarray,
    fun_evals: np.ndarray,
    kernel: Optional[Kernel] = None,
    domain: Optional[
        Tuple[Union[np.ndarray, FloatLike], Union[np.ndarray, FloatLike]]
    ] = None,
    measure: Optional[IntegrationMeasure] = None,
) -> Tuple[Normal, Dict]:
    r"""Infer the value of an integral from a given set of nodes and function
    evaluations.

    Parameters
    ----------
    nodes :
        *shape=(n_eval, input_dim)* -- Locations at which the function evaluations are
        available as ``fun_evals``.
    fun_evals :
        *shape=(n_eval,)* -- Function evaluations at ``nodes``.
    kernel :
        The kernel used for the GP model.
    domain :
        *shape=(input_dim,)* -- Domain of integration. Contains lower and upper bound as
        int or ndarray.
    measure:
        Integration measure. Defaults to the Lebesgue measure.

    Returns
    -------
    integral :
        The integral of ``fun`` on the domain.
    info :
        Information on the performance of the method.

    Raises
    ------
    ValueError
        If neither a domain nor a measure are given.
    ValueError
        If a domain is given with a Gaussian measure.

    Examples
    --------
    >>> import numpy as np
    >>> domain = (0, 1)
    >>> nodes = np.linspace(0, 1, 15)[:, None]
    >>> fun_evals = 3*nodes**2
    >>> F, info = bayesquad_from_data(nodes=nodes, fun_evals=fun_evals, domain=domain)
    >>> print(F.mean)
    1.0001
    """

    # Check input argument compatibility
    if domain is None and measure is None:
        raise ValueError(
            "You need to either specify an integration domain or an integration "
            "measure. The Lebesgue measure can only operate on a finite domain."
        )

    if domain is not None:
        if isinstance(measure, GaussianMeasure):
            raise ValueError("GaussianMeasure cannot be used with finite bounds.")
        elif isinstance(measure, LebesgueMeasure):
            warnings.warn(
                "Both domain and a LebesgueMeasure are specified. The domain "
                "information will be ignored."
            )

    if nodes.ndim != 2:
        raise ValueError(
            "The nodes must be given a in an array with shape=(n_eval, input_dim)"
        )
    n_eval, input_dim = nodes.shape

    bq_method = BayesianQuadrature.from_problem(
        input_dim=input_dim,
        kernel=kernel,
        measure=measure,
        domain=domain,
        max_evals=n_eval,
        batch_size=n_eval,
        policy="fixed",
    )

    # Integrate
    integral_belief, bq_state = bq_method.integrate(nodes=nodes, fun_evals=fun_evals)

    return integral_belief, bq_state.info
