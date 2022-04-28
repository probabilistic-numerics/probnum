"""Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature,
meaning a model over the integrand is constructed in order to actively select evaluation
points of the integrand to estimate the value of the integral. Bayesian quadrature
methods return a random variable, specifying the belief about the true value of the
integral.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple
import warnings

import numpy as np

from probnum.quad.solvers.bq_state import BQIterInfo
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike, IntLike

from ._integration_measures import IntegrationMeasure, LebesgueMeasure
from ._quad_typing import DomainLike, DomainType
from .solvers import BayesianQuadrature


def bayesquad(
    fun: Callable,
    input_dim: int,
    kernel: Optional[Kernel] = None,
    domain: Optional[DomainLike] = None,
    measure: Optional[IntegrationMeasure] = None,
    policy: Optional[str] = "bmc",
    max_evals: Optional[IntLike] = None,
    var_tol: Optional[FloatLike] = None,
    rel_tol: Optional[FloatLike] = None,
    batch_size: Optional[IntLike] = 1,
    rng: Optional[np.random.Generator] = np.random.default_rng(),
) -> Tuple[Normal, BQIterInfo]:
    r"""Infer the solution of the uni- or multivariate integral
    :math:`\int_\Omega f(x) d \mu(x)`
    on a hyper-rectangle :math:`\Omega = [a_1, b_1] \times \cdots \times [a_D, b_D]`
    or :math:`\Omega = \mathbb{R}^D`.

    Bayesian quadrature (BQ) infers integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x),

    of a function :math:`f:\mathbb{R}^D \mapsto \mathbb{R}` integrated on the domain
    :math:`\Omega \subset \mathbb{R}^D` against a measure
    :math:`\mu` on :math:`\mathbb{R}^D`.

    Bayesian quadrature methods return a probability distribution over the solution
    :math:`F` with uncertainty arising from finite computation (here a finite number
    of function evaluations). They start out with a random process encoding the prior
    belief about the function :math:`f` to be integrated. Conditioned on either existing
    or acquired function evaluations according to a policy, they update the belief on
    :math:`f`, which is translated into a posterior measure over the integral :math:`F`.
    See Briol et al. [1]_ for a review on Bayesian quadrature.

    Parameters
    ----------
    fun
        Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
        ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
    input_dim
        Input dimension of the integration problem.
    kernel
        The kernel used for the GP model. Defaults to the ``ExpQuad`` kernel.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    measure
        The integration measure. Defaults to the Lebesgue measure on ``domain``.
    policy
        Type of acquisition strategy to use. Defaults to 'bmc'. Options are

        ==========================  =======
         Bayesian Monte Carlo [2]_  ``bmc``
        ==========================  =======

    max_evals
        Maximum number of function evaluations.
    var_tol
        Tolerance on the variance of the integral.
    rel_tol
        Tolerance on consecutive updates of the integral mean.
    batch_size
        Number of new observations at each update.
    rng
        Random number generator. Used by Bayesian Monte Carlo other random sampling
        policies. Optional. Default is `np.random.default_rng()`.

    Returns
    -------
    integral :
        The integral belief of :math:`F` subject to the provided measure or domain.
    info :
        Information on the performance of the method.

    Raises
    ------
    ValueError
        If neither a domain nor a measure are given.

    Warns
    -----
    When ``domain`` is given but not used.

    Notes
    -----
        If multiple stopping conditions are provided, the method stops once one of
        them is satisfied. If no stopping condition is provided, the default values are
        ``max_evals = 25 * input_dim`` and ``var_tol = 1e-6``.

    See Also
    --------
    bayesquad_from_data : Computes the integral :math:`F` using a given dataset of
                          nodes and function evaluations.

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
    ...     return x.reshape(-1, )
    >>> F, info = bayesquad(fun=f, input_dim=input_dim, domain=domain)
    >>> print(F.mean)
    0.5
    """

    input_dim, domain, measure = _check_domain_measure_compatibility(
        input_dim=input_dim, domain=domain, measure=measure
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
    integral_belief, _, info = bq_method.integrate(fun=fun, nodes=None, fun_evals=None)

    return integral_belief, info


def bayesquad_from_data(
    nodes: np.ndarray,
    fun_evals: np.ndarray,
    kernel: Optional[Kernel] = None,
    domain: Optional[DomainLike] = None,
    measure: Optional[IntegrationMeasure] = None,
) -> Tuple[Normal, BQIterInfo]:
    r"""Infer the value of an integral from a given set of nodes and function
    evaluations.

    Parameters
    ----------
    nodes
        *shape=(n_eval, input_dim)* -- Locations at which the function evaluations are
        available as ``fun_evals``.
    fun_evals
        *shape=(n_eval,)* -- Function evaluations at ``nodes``.
    kernel
        The kernel used for the GP model. Defaults to the ``ExpQuad`` kernel.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    measure
        The integration measure. Defaults to the Lebesgue measure.

    Returns
    -------
    integral :
        The integral belief subject to the provided measure or domain.
    info :
        Information on the performance of the method.

    Raises
    ------
    ValueError
        If neither a domain nor a measure are given.

    Warns
    -----
    When ``domain`` is given but not used.

    See Also
    --------
    bayesquad : Computes the integral using an acquisition policy.

    Examples
    --------
    >>> import numpy as np
    >>> domain = (0, 1)
    >>> nodes = np.linspace(0, 1, 15)[:, None]
    >>> fun_evals = nodes.reshape(-1, )
    >>> F, info = bayesquad_from_data(nodes=nodes, fun_evals=fun_evals, domain=domain)
    >>> print(F.mean)
    0.5
    """

    if nodes.ndim != 2:
        raise ValueError(
            "The nodes must be given a in an array with shape=(n_eval, input_dim)"
        )

    input_dim, domain, measure = _check_domain_measure_compatibility(
        input_dim=nodes.shape[1], domain=domain, measure=measure
    )

    bq_method = BayesianQuadrature.from_problem(
        input_dim=input_dim,
        kernel=kernel,
        measure=measure,
        domain=domain,
        policy=None,
    )

    # Integrate
    integral_belief, _, info = bq_method.integrate(
        fun=None, nodes=nodes, fun_evals=fun_evals
    )

    return integral_belief, info


def _check_domain_measure_compatibility(
    input_dim: IntLike,
    domain: Optional[DomainLike],
    measure: Optional[IntegrationMeasure],
) -> Tuple[int, Optional[DomainType], IntegrationMeasure]:

    # Neither domain nor measure given
    if domain is None and measure is None:
        raise ValueError(
            "You need to either specify an integration domain or an integration "
            "measure. The Lebesgue measure can only operate on a finite domain."
        )

    # Ignore domain if measure is given
    if domain is not None and measure is not None:
        warnings.warn(
            "Both 'domain' and a 'measure' are specified. 'domain' will be ignored."
        )
        domain = None

    # Set measure if only domain is given
    if measure is None:
        measure = LebesgueMeasure(domain=domain, input_dim=input_dim)
        domain = measure.domain  # domain has been converted to correct type

    return input_dim, domain, measure
