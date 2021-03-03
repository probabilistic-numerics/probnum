"""Convenience functions for Gaussian filtering and smoothing.

We support the following methods:
    - ekf0: Extended Kalman filtering based on a zero-th order Taylor
        approximation [1]_, [2]_, [3]_. Also known as "PFOS".
    - ekf1: Extended Kalman filtering [3]_.
    - ukf: Unscented Kalman filtering [3]_.
    - eks0: Extended Kalman smoothing based on a zero-th order Taylor
        approximation [4]_.
    - eks1: Extended Kalman smoothing [4]_.
    - uks: Unscented Kalman smoothing.

References
----------
.. [1] https://arxiv.org/pdf/1610.05261.pdf
.. [2] https://arxiv.org/abs/1807.09737
.. [3] https://arxiv.org/abs/1810.03440
.. [4] https://arxiv.org/pdf/2004.00623.pdf
"""

import numpy as np

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from probnum.diffeq import steprule
from probnum.diffeq.ode import IVP
from probnum.diffeq.odefiltsmooth import ivp2filter
from probnum.diffeq.odefiltsmooth.ivpfiltsmooth import GaussianIVPFilter


def probsolve_ivp(
    f,
    t0,
    tmax,
    y0,
    df=None,
    method="ek0",
    dense_output=True,
    which_prior="ibm1",
    atol=1e-4,
    rtol=1e-2,
    step=None,
    driftspeed=1.0,
    lengthscale=1.0,
):
    r"""Solve initial value problem with Gaussian filtering and smoothing.

    Numerically computes a Gauss-Markov process which solves numerically
    the initial value problem (IVP) based on a system of first order
    ordinary differential equations (ODEs)

    .. math:: \\dot x(t) = f(t, x(t)), \\quad x(t_0) = x_0,
        \\quad t \\in [t_0, T]

    by regarding it as a (nonlinear) Gaussian filtering (and smoothing)
    problem [3]_. For some configurations it recovers certain multistep
    methods [1]_.
    Convergence rates of filtering [2]_ and smoothing [4]_ are
    comparable to those of methods of Runge-Kutta type.


    This function turns a prior-string into an :class:`ODEPrior`, a
    method-string into a filter/smoother of class :class:`GaussFiltSmooth`, creates a
    :class:`GaussianIVPFilter` object and calls the :meth:`solve()` method. For
    advanced usage we recommend to do this process manually which
    enables advanced methods of tuning the algorithm.

    This function supports the methods:
    extended Kalman filtering based on a zero-th order Taylor
    approximation (EKF0),
    extended Kalman filtering (EKF1),
    unscented Kalman filtering (UKF),
    extended Kalman smoothing based on a zero-th order Taylor
    approximation (EKS0),
    extended Kalman smoothing (EKS1), and
    unscented Kalman smoothing (UKS).

    For adaptive step-size selection of ODE filters, we implement the
    scheme proposed by Schober et al. (2019), and further examined by Bosch et al (2021),
    where the local error estimate is derived from the local, calibrated
    uncertainty estimate.

    Arguments
    ---------
    f :
        ODE vector field.
    t0 :
        Initial time point.
    tmax :
        Final time point.
    y0 :
        Initial value.
    df :
        Jacobian of the ODE vector field.
    atol : float
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    rtol : float
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    step :
        Step size. If atol and rtol are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen as :math:`0.01 \cdot |y_0|/|f(t_0, y_0)|`.
    which_prior : str, optional
        Which prior is to be used. Default is an IBM(1), further support
        for IBM(:math:`q`), IOUP(:math:`q`), Matern(:math:`q+1/2`),
        :math:`q\\in\\{1, 2, 3, 4\\}` is provided. The available
        options are

        ======================  ========================================
         IBM(:math:`q`)         ``'ibm1'``, ``'ibm2'``, ``'ibm3'``,
                                ``'ibm4'``
         IOUP(:math:`q`)        ``'ioup1'``, ``'ioup2'``, ``'ioup3'``,
                                ``'ioup4'``
         Matern(:math:`q+0.5`)  ``'matern32'``, ``'matern52'``,
                                ``'matern72'``, ``'matern92'``
        ======================  ========================================

        The type of prior relates to prior assumptions about the
        derivative of the solution. The IBM(:math:`q`) prior leads to a
        :math:`q`-th order method that is recommended if little to no
        prior information about the solution is available. On the other
        hand, if the :math:`q`-th derivative is expected to regress to
        zero, an IOUP(:math:`q`) prior might be suitable.
    method : str, optional
        Which method is to be used. Default is ``ek0`` which is the
        method proposed by Schober et al.. The available
        options are

        ================================================  ==============
         Extended Kalman filtering/smoothing (0th order)  ``'ek0'``
         Extended Kalman filtering/smoothing (1st order)  ``'ek1'``
         Unscented Kalman filtering/smoothing             ``'uk'``
        ================================================  ==============

        First order extended Kalman filtering and smoothing methods
        require Jacobians of the RHS-vector field of the IVP. That is,
        the argument ``df`` needs to be specified. The unscented Kalman filter is supported,
        but since its square-root implementation is not available yet, it will be less stable
        than the extended Kalman filter variations.
    dense_output : bool
        Whether we want dense output. Optional. Default is ``True``. For the ODE filter,
        dense output requires smoothing, so if ``dense_output`` is False, no smoothing is performed;
        but when it is ``True``, the filter solution is smoothed.
    driftspeed : float
        Drift speed of the IOUP process. Only used there, i.e. IBM and Matern remain unaffected by this argument.
        Optional. Default is 1.0.
    lengthscale : float
        Length scale of the Matern process. Only used there, i.e. IBM and IOUP remain unaffected by this argument.
        Optional. Default is 1.0.

    Returns
    -------
    solution : KalmanODESolution
        Solution of the ODE problem.

        Can be evaluated at and sampled from at arbitrary grid points.
        Further, it contains fields:

        t : :obj:`np.ndarray`, shape=(N,)
            Mesh used by the solver to compute the solution.
            It includes the initial time :math:`t_0` but not necessarily the
            final time :math:`T`.
        y : :obj:`list` of :obj:`RandomVariable`, length=N
            Discrete-time solution at times :math:`t_1, ..., t_N`,
            as a list of random variables.
            The means and covariances can be accessed with ``solution.y.mean``
            and ``solution.y.cov``.

    See Also
    --------
    GaussianIVPFilter : Solve IVPs with Gaussian filtering and smoothing
    KalmanODESolution : Solution of ODE problems based on Gaussian filtering and smoothing.

    References
    ----------
    .. [1] Schober, M., Särkkä, S. and Hennig, P..
        A probabilistic model for the numerical solution of initial
        value problems.
        Statistics and Computing, 2019.
    .. [2] Tronarp, F., Kersting, H., Särkkä, S., and Hennig, P..
        Probabilistic solutions to ordinary differential equations as
        non-linear Bayesian filtering: a new perspective.
        Statistics and Computing, 2019.
    .. [3] Bosch, N., Hennig, P., and Tronarp, F..
        Calibrated adaptive probabilistic ODE solvers.
        AISTATS 2021.


    Examples
    --------
    >>> from probnum.diffeq import logistic, probsolve_ivp
    >>> from probnum import random_variables as rvs
    >>> import numpy as np
    >>> initrv = rvs.Constant(0.15)
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="ekf0", step=0.1)
    >>> print(np.round(solution.y.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.36]
     [0.46]
     [0.56]
     [0.65]
     [0.74]
     [0.81]
     [0.86]
     [0.9 ]
     [0.93]
     [0.95]
     [0.97]
     [0.98]
     [0.98]]

    >>> initrv = rvs.Constant(0.15)
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="eks1", which_prior="ioup3", step=0.1)
    >>> print(np.round(solution.y.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.37]
     [0.47]
     [0.57]
     [0.66]
     [0.74]
     [0.81]
     [0.87]
     [0.91]
     [0.93]
     [0.96]
     [0.97]
     [0.98]
     [0.99]]
    """
    ivp = IVP(timespan=(t0, tmax), initrv=pnrv.Constant(np.asarray(y0)), rhs=f, jac=df)
    stprl = _create_steprule(atol, rtol, step, ivp)
    prior = _string2prior(ivp, which_prior, driftspeed, lengthscale)
    gfilt = _create_filter(ivp, prior, method)
    solver = GaussianIVPFilter(ivp, gfilt, with_smoothing=dense_output)
    solution = solver.solve(steprule=stprl)
    return solution


def _create_filter(ivp, prior, method):
    """Create the solver object that is used."""
    if method not in ["ek0", "ek1", "uk"]:
        raise ValueError("Method not supported.")
    gfilt = _string2filter(ivp, prior, method)
    return gfilt


def _create_steprule(atol, rtol, step, ivp):
    if atol is None and rtol is None and step is None:
        errormsg = (
            "Please specify either absolute and relative tolerances or a step size."
        )
        raise ValueError(errormsg)

    if atol is None and rtol is None:
        stprl = steprule.ConstantSteps(step)
    else:
        if step is None:
            # lazy version of Hairer, Wanner, Norsett, p. 169
            norm_y0 = np.linalg.norm(ivp.initrv.mean)
            norm_dy0 = np.linalg.norm(ivp(ivp.t0, ivp.initrv.mean))
            firststep = 0.01 * norm_y0 / norm_dy0
        else:
            firststep = step
        stprl = steprule.AdaptiveSteps(firststep=firststep, atol=atol, rtol=rtol)
    return stprl


def _string2prior(ivp, which_prior, driftspeed, lengthscale):

    ibm_family = ["ibm1", "ibm2", "ibm3", "ibm4"]
    ioup_family = ["ioup1", "ioup2", "ioup3", "ioup4"]
    matern_family = ["matern32", "matern52", "matern72", "matern92"]
    if which_prior in ibm_family:
        return _string2ibm(ivp, which_prior)
    elif which_prior in ioup_family:
        return _string2ioup(ivp, which_prior, driftspeed=driftspeed)
    elif which_prior in matern_family:
        return _string2matern(ivp, which_prior, lengthscale=lengthscale)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2ibm(ivp, which_prior):

    if which_prior == "ibm1":
        return pnfs.statespace.IBM(
            1,
            ivp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ibm2":
        return pnfs.statespace.IBM(
            2,
            ivp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ibm3":
        return pnfs.statespace.IBM(
            3,
            ivp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ibm4":
        return pnfs.statespace.IBM(
            4,
            ivp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2ioup(ivp, which_prior, driftspeed):

    if which_prior == "ioup1":
        return pnfs.statespace.IOUP(
            1,
            ivp.dimension,
            driftspeed,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ioup2":
        return pnfs.statespace.IOUP(
            2,
            ivp.dimension,
            driftspeed,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ioup3":
        return pnfs.statespace.IOUP(
            3,
            ivp.dimension,
            driftspeed,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "ioup4":
        return pnfs.statespace.IOUP(
            4,
            ivp.dimension,
            driftspeed,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2matern(ivp, which_prior, lengthscale):

    if which_prior == "matern32":
        return pnfs.statespace.Matern(
            1,
            ivp.dimension,
            lengthscale,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "matern52":
        return pnfs.statespace.Matern(
            2,
            ivp.dimension,
            lengthscale,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "matern72":
        return pnfs.statespace.Matern(
            3,
            ivp.dimension,
            lengthscale,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    elif which_prior == "matern92":
        return pnfs.statespace.Matern(
            4,
            ivp.dimension,
            lengthscale,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2filter(_ivp, _prior, _method):

    evlvar = 0.0
    if _method == "ek0":
        return ivp2filter.ivp2ekf0(_ivp, _prior, evlvar)
    if _method == "ek1":
        return ivp2filter.ivp2ekf1(_ivp, _prior, evlvar)
    if _method == "uk":
        return ivp2filter.ivp2ukf(_ivp, _prior, evlvar)
    raise ValueError("Type of filter not supported.")
