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

from probnum.quad.integration_measures import IntegrationMeasure, LebesgueMeasure
from probnum.quad.solvers import BayesianQuadrature, BQIterInfo
from probnum.quad.typing import DomainLike, DomainType
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import IntLike


def bayesquad(
    fun: Callable,
    input_dim: IntLike,
    kernel: Optional[Kernel] = None,
    measure: Optional[IntegrationMeasure] = None,
    domain: Optional[DomainLike] = None,
    policy: Optional[str] = "bmc",
    initial_design: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    options: Optional[dict] = None,
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
    measure
        The integration measure. Defaults to the Lebesgue measure on ``domain``.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    policy
        Type of acquisition strategy to use. Defaults to 'bmc'. Options are

        ==================================================== ===========
         Bayesian Monte Carlo [2]_                            ``bmc``
         Van Der Corput points                                ``vdc``
         Uncertainty Sampling with random candidates          ``us_rand``
         Uncertainty Sampling with optimizer                  ``us``
         Mutual information with random candidates            ``mi_rand``
         Mutual information with optimizer                    ``mi``
         Integral variance reduction with random candidates   ``ivr_rand``
         Integral variance reduction with optimizer           ``ivr``
        ==================================================== ===========

    initial_design
        The type of initial design to use. If ``None`` is given, no initial design is
        used. Options are

        ==========================  =========
         Samples from measure       ``mc``
         Latin hypercube [3]_       ``latin``
        ==========================  =========

    rng
        The random number generator used for random methods.

    options
        A dictionary with the following optional solver settings

            scale_estimation : Optional[str]
                Estimation method to use to compute the scale parameter. Defaults
                to 'mle'. Options are

                ==============================  =======
                 Maximum likelihood estimation  ``mle``
                ==============================  =======

            max_evals : Optional[IntLike]
                Maximum number of function evaluations.
            var_tol : Optional[FloatLike]
                Tolerance on the variance of the integral.
            rel_tol : Optional[FloatLike]
                Tolerance on consecutive updates of the integral mean.
            jitter : Optional[FloatLike]
                Non-negative jitter to numerically stabilise kernel matrix
                inversion. Defaults to 1e-8.
            batch_size : Optional[IntLike]
                Number of new observations at each update. Defaults to 1.
            n_initial_design_nodes : Optional[IntLike]
                The number of nodes created by the initial design. Defaults to
                ``input_dim * 5`` if an initial design is given.
            n_candidates : Optional[IntLike]
                The number of candidate nodes used by the policies that maximize an
                acquisition function by drawing random candidates. Defaults to 1e2.
                Applicable to policies 'us_rand', 'mi_rand' and 'ivr_rand'.
            n_restarts : Optional[IntLike]
                The number of restarts that the acquisition optimizer performs in
                order to find the maximizer. Defaults to 10. Applicable to policies
                'us', 'mi' and 'ivr'.

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
    UserWarning
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

    Warnings
    --------
    Currently the method does not support tuning of the kernel parameters
    other than the global kernel scale. Hence, the method may perform poorly unless the
    kernel parameters are set to appropriate values by the user.

    References
    ----------
    .. [1] Briol, F.-X., et al., Probabilistic integration: A role in statistical
        computation?, *Statistical Science 34.1*, 2019, 1-22.
    .. [2] Rasmussen, C. E., and Z. Ghahramani, Bayesian Monte Carlo, *Advances in
        Neural Information Processing Systems*, 2003, 505-512.
    .. [3] Mckay et al., A Comparison of Three Methods for Selecting Values of Input
        Variables in the Analysis of Output from a Computer Code, *Technometrics*, 1979.

    Examples
    --------
    >>> import numpy as np
    >>> input_dim = 1
    >>> domain = (0, 1)
    >>> def fun(x):
    ...     return x.reshape(-1, )
    >>> F, info = bayesquad(fun, input_dim, domain=domain, rng=np.random.default_rng(0))
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
        initial_design=initial_design,
        options=options,
    )

    # Integrate
    integral_belief, _, info = bq_method.integrate(
        fun=fun, nodes=None, fun_evals=None, rng=rng
    )

    return integral_belief, info


def bayesquad_from_data(
    nodes: np.ndarray,
    fun_evals: np.ndarray,
    kernel: Optional[Kernel] = None,
    measure: Optional[IntegrationMeasure] = None,
    domain: Optional[DomainLike] = None,
    options: Optional[dict] = None,
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
    measure
        The integration measure. Defaults to the Lebesgue measure.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    options
        A dictionary with the following optional solver settings

            scale_estimation : Optional[str]
                Estimation method to use to compute the scale parameter. Defaults
                to 'mle'. Options are

                ==============================  =======
                 Maximum likelihood estimation  ``mle``
                ==============================  =======

            jitter : Optional[FloatLike]
                Non-negative jitter to numerically stabilise kernel matrix
                inversion. Defaults to 1e-8.

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
    UserWarning
        When ``domain`` is given but not used.

    See Also
    --------
    bayesquad : Computes the integral using an acquisition policy.

    Warnings
    --------
    Currently the method does not support tuning of the kernel parameters
    other than the global kernel scale. Hence, the method may perform poorly unless the
    kernel parameters are set to appropriate values by the user.

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
        initial_design=None,
        options=options,
    )

    # Integrate
    integral_belief, _, info = bq_method.integrate(
        fun=None, nodes=nodes, fun_evals=fun_evals, rng=None
    )

    return integral_belief, info


def multilevel_bayesquad_from_data(
    nodes: Tuple[np.ndarray, ...],
    fun_diff_evals: Tuple[np.ndarray, ...],
    kernels: Optional[Tuple[Kernel, ...]] = None,
    measure: Optional[IntegrationMeasure] = None,
    domain: Optional[DomainLike] = None,
    options: Optional[dict] = None,
) -> Tuple[Normal, Tuple[BQIterInfo, ...]]:
    r"""Infer the value of an integral from given sets of nodes and function
    evaluations using a multilevel method.

    In multilevel Bayesian quadrature, the integral :math:`\int_\Omega f(x) d \mu(x)`
    is (approximately) decomposed as a telescoping sum over :math:`L+1` levels:

    .. math:: \int_\Omega f(x) d \mu(x) \approx \int_\Omega f_0(x) d
        \mu(x) + \sum_{l=1}^L \int_\Omega [f_l(x) - f_{l-1}(x)] d \mu(x),

    where :math:`f_l` is an increasingly accurate but also increasingly expensive
    approximation to :math:`f`. It is not necessary that the highest level approximation
    :math:`f_L` be equal to :math:`f`.

    Bayesian quadrature is subsequently applied to independently infer each of the
    :math:`L+1` integrals and the outputs are summed to infer
    :math:`\int_\Omega f(x) d \mu(x)`. [1]_

    Parameters
    ----------
    nodes
        Tuple of length :math:`L+1` containing the locations for each level at which
        the functionn evaluations are available as ``fun_diff_evals``. Each element
        must be a shape=(n_eval, input_dim) ``np.ndarray``. If a tuple containing only
        one element is provided, it is inferred that the same nodes ``nodes[0]`` are
        used on every level.
    fun_diff_evals
        Tuple of length :math:`L+1` containing the evaluations of :math:`f_l - f_{l-1}`
        for each level at the nodes provided in ``nodes``. Each element must be a
        shape=(n_eval,) ``np.ndarray``. The zeroth element contains the evaluations of
        :math:`f_0`.
    kernels
        Tuple of length :math:`L+1` containing the kernels used for the GP model at each
        level. See **Notes** for further details. Defaults to the ``ExpQuad`` kernel for
        each level.
    measure
        The integration measure. Defaults to the Lebesgue measure.
    domain
        The integration domain. Contains lower and upper bound as scalar or
        ``np.ndarray``. Obsolete if ``measure`` is given.
    options
        A dictionary with the following optional solver settings

            scale_estimation : Optional[str]
                Estimation method to use to compute the scale parameter. Used
                independently on each level. Defaults to 'mle'. Options are

                ==============================  =======
                 Maximum likelihood estimation  ``mle``
                ==============================  =======

            jitter : Optional[FloatLike]
                Non-negative jitter to numerically stabilise kernel matrix
                inversion. Same jitter is used on each level. Defaults to 1e-8.

    Returns
    -------
    integral :
        The integral belief subject to the provided measure or domain.
    infos :
        Information on the performance of the method for each level.

    Raises
    ------
    ValueError
        If ``nodes``, ``fun_diff_evals`` or ``kernels`` have different lengths.

    Warns
    -----
    UserWarning
        When ``domain`` is given but not used.

    Notes
    -----
    The tuple of kernels provided by the ``kernels`` parameter must contain distinct
    kernel instances, i.e., ``kernels[i] is kernel[j]`` must return ``False`` for any
    :math:`i\neq j`.

    Warnings
    --------
    Currently the method does not support tuning of the kernel parameters
    other than the global kernel scale. Hence, the method may perform poorly unless the
    kernel parameters are set to appropriate values by the user.

    References
    ----------
    .. [1] Li, K., et al., Multilevel Bayesian quadrature, AISTATS, 2023.

    Examples
    --------
    >>> import numpy as np
    >>> input_dim = 1
    >>> domain = (0, 1)
    >>> n_level = 6
    >>> def fun(x, l):
    ...     return x.reshape(-1, ) / (l + 1.0)
    >>> nodes = ()
    >>> fun_diff_evals = ()
    >>> for l in range(n_level):
    ...     n_l = 2*l + 1
    ...     nodes += (np.reshape(np.linspace(0, 1, n_l), (n_l, input_dim)),)
    ...     fun_diff_evals += (np.reshape(fun(nodes[l], l), (n_l,)),)
    >>> F, infos = multilevel_bayesquad_from_data(nodes=nodes,
    ...                                           fun_diff_evals=fun_diff_evals,
    ...                                           domain=domain)
    >>> print(np.round(F.mean, 4))
    0.7252
    """

    n_level = len(fun_diff_evals)
    if kernels is None:
        kernels = n_level * (None,)
    if len(nodes) == 1:
        nodes = n_level * (nodes[0],)
    if not len(nodes) == len(fun_diff_evals) == len(kernels):
        raise ValueError(
            f"You must provide an equal number of kernels ({(len(kernels))}), "
            f"vectors of function evaluations ({len(fun_diff_evals)}) "
            f"and sets of nodes ({len(nodes)})."
        )

    integer_belief = Normal(mean=0.0, cov=0.0)
    infos = ()
    for level in range(n_level):
        integer_belief_l, info_l = bayesquad_from_data(
            nodes=nodes[level],
            fun_evals=fun_diff_evals[level],
            kernel=kernels[level],
            measure=measure,
            domain=domain,
            options=options,
        )
        integer_belief += integer_belief_l
        infos += (info_l,)

    return integer_belief, infos


def _check_domain_measure_compatibility(
    input_dim: IntLike,
    domain: Optional[DomainLike],
    measure: Optional[IntegrationMeasure],
) -> Tuple[int, Optional[DomainType], IntegrationMeasure]:
    input_dim = int(input_dim)

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
