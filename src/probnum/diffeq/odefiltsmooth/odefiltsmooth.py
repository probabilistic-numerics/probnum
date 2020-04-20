"""
Convenience functions for Gaussian filtering and smoothing.

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

from probnum.diffeq import steprule
from probnum.diffeq.odefiltsmooth import prior, ivptofilter
from probnum.diffeq.odefiltsmooth import GaussianIVPFilter, GaussianIVPSmoother


def probsolve_ivp(ivp, method="ekf0", which_prior="ibm1", tol=None,
                  step=None, firststep=None, **kwargs):
    """
    Solve initial value problem with Gaussian filtering and smoothing.

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


    This function turns a prior-string into an ``ODEPrior``, a
    method-string into a ``GaussianSmoother``, creates a
    ``GaussianIVPSmoother`` object and calls the ``solve()`` method. For
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

    Arguments
    ---------
    ivp : IVP
        Initial value problem to be solved.
    step : float
        Step size :math:`h` of the solver. This defines the
        discretisation mesh as each proposed step is equal to :math:`h`
        and all proposed steps are accepted.
        Only one of out of ``step`` and ``tol`` is set.
    tol : float
        Tolerance :math:`\\varepsilon` of the adaptive step scheme.
        We implement the scheme proposed by Schober et al., accepting a
        step if the absolute as well as the relative error estimate are
        smaller than the tolerance,
        :math:`\\max\\{e, e / |y|\\} \\leq \\varepsilon`.
        Only one of out of ``step`` and ``tol`` is set.
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
        Which method is to be used. Default is ``ekf0`` which is the
        method proposed by Schober et al.. The available
        options are

        ================================================  ==============
         Extended Kalman filtering/smoothing (0th order)  ``'ekf0'``,
                                                          ``'eks0'``
         Extended Kalman filtering/smoothing (1st order)  ``'ekf1'``,
                                                          ``'eks1'``
         Unscented Kalman filtering/smoothing             ``'ukf'``,
                                                          ``'uks'``
        ================================================  ==============

        First order extended Kalman filtering and smoothing methods
        require Jacobians of the RHS-vector field of the IVP. The
        uncertainty estimates as returned by EKF1/S1 and UKF/S appear to
        be more reliable than those of EKF0/S0. The latter is more
        stable when it comes to very small steps.

    firststep : float, optional
        First suggested step :math:`h_0` for adaptive step size scheme.
        Default is None which lets the solver start with the suggestion
        :math:`h_0 = T - t_0`. For low accuracy it might be more
        efficient to start out with smaller :math:`h_0` so that the
        first acceptance occurs earlier.

    Returns
    -------
    means : np.ndarray, shape=(N, d*(q+1))
        Mean vector of the solution at times :math:`t_1, ..., t_N`.
        The elements are ordered as
        ``(m_1, m_1', m_1'', ..., m_2, m_2', ...)``
        where ``m_1`` is the estimate of the first coordinate of the
        solution, ``m_1'`` is the estimate of the derivative of the
        first coordinate, and so on.
    covs : np.ndarray, shape=(N, d*(q+1), d*(q+1))
        Covariance matrices of the solution at times
        :math:`t_1, ..., t_N`. The ordering reflects the ordering of
        the mean vector.
    times : np.ndarray, shape=(N,)
        Mesh used by the solver to compute the solution.
        It includes the initial time :math:`t_0` but not necessarily the
        final time :math:`T`.

    See Also
    --------
    GaussianIVPFilter : Solve IVPs with Gaussian filtering.
    GaussianIVPSmoother : Solve IVPs with Gaussian smoothing.

    References
    ----------
    .. [1] Schober, M., Särkkä, S. and Hennig, P..
        A probabilistic model for the numerical solution of initial
        value problems.
        Statistics and Computing, 2019.
    .. [2] Kersting, H., Sullivan, T.J., and Hennig, P..
        Convergence rates of Gaussian ODE filters.
        2019.
    .. [3] Tronarp, F., Kersting, H., Särkkä, S., and Hennig, P..
        Probabilistic solutions to ordinary differential equations as
        non-linear Bayesian filtering: a new perspective.
        Statistics and Computing, 2019.
    .. [4] Tronarp, F., Särkkä, S., and Hennig, P..
        Bayesian ODE solvers: the maximum a posteriori estimate.
        2019.


    Examples
    --------
    >>> from probnum.diffeq import logistic, probsolve_ivp
    >>> from probnum.prob import RandomVariable, Dirac, Normal
    >>> initrv = RandomVariable(distribution=Dirac(0.15))
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> means, covs, times = probsolve_ivp(ivp, method="ekf0", step=0.1)
    >>> print(means)
    [[0.15       0.51      ]
     [0.2076198  0.642396  ]
     [0.27932997 0.79180747]
     [0.3649165  0.91992313]
     [0.46054129 0.9925726 ]
     [0.55945475 0.98569653]
     [0.65374523 0.90011316]
     [0.73686744 0.76233098]
     [0.8053776  0.60787222]
     [0.85895587 0.4636933 ]
     [0.89928283 0.34284592]
     [0.92882899 0.24807715]
     [0.95007559 0.17685497]
     [0.96515825 0.12479825]
     [0.97577054 0.08744746]
     [0.9831919  0.06097975]]

    >>> initrv = RandomVariable(distribution=Dirac(0.15))
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> means, covs, times = probsolve_ivp(ivp, method="eks1", which_prior="ioup3", step=0.1)
    >>> print(means)
    [[  0.15         0.51         1.428       -0.57058847]
     [  0.20791497   0.65874554   2.03752238  -7.07137359]
     [  0.28229381   0.81041611   1.08305597   2.71961497]
     [  0.36925352   0.93162419   1.16158644 -10.36628566]
     [  0.46646306   0.99550295   0.17033983  -6.20762551]
     [  0.56587493   0.9826431   -0.45689221  -8.50556098]
     [  0.66048248   0.89698152  -1.1848091   -3.68614765]
     [  0.74369637   0.76244935  -1.47038678  -1.49777207]
     [  0.8123721    0.60969654  -1.53817311   1.57340589]
     [  0.86592643   0.46439253  -1.35515267   2.37437168]
     [  0.90598187   0.34071527  -1.11173213   2.83861203]
     [  0.93495652   0.24325135  -0.84433513   2.40538222]
     [  0.95544694   0.17027234  -0.62178048   2.0057747 ]
     [  0.96968716   0.11757592  -0.44062203   1.46760485]
     [  0.97947605   0.08041089  -0.3090062    1.07105312]
     [  0.98614523   0.05465129  -0.20824359   0.94816533]]
    """
    _check_step_tol(step, tol)
    _check_method(method)
    _prior = _string_to_prior(ivp, which_prior, **kwargs)
    if tol is not None:
        stprl = _step_to_adaptive_steprule(tol, _prior)
        if firststep is None:
            firststep = ivp.tmax - ivp.t0
    else:
        stprl = _step_to_steprule(step)
        firststep = step
    gfilt = _string_to_filter(ivp, _prior, method, **kwargs)
    if method in ["ekf0", "ekf1", "ukf"]:
        solver = GaussianIVPFilter(ivp, gfilt, stprl)
    else:
        solver = GaussianIVPSmoother(ivp, gfilt, stprl)
    return solver.solve(firststep=firststep, **kwargs)


def _check_step_tol(step, tol):
    """ """
    both_none = tol is None and step is None
    both_not_none = tol is not None and step is not None
    if both_none or both_not_none:
        errormsg = "Please specify either a tolerance or a step size."
        raise TypeError(errormsg)


def _check_method(method):
    """ """
    if method not in ["ekf0", "ekf1", "ukf", "eks0", "eks1", "uks"]:
        raise TypeError("Method not supported.")


def _string_to_prior(ivp, which_prior, **kwargs):
    """
    """
    ibm_family = ["ibm1", "ibm2", "ibm3", "ibm4"]
    ioup_family = ["ioup1", "ioup2", "ioup3", "ioup4"]
    matern_family = ["matern32", "matern52", "matern72", "matern92"]
    if which_prior in ibm_family:
        return _string_to_prior_ibm(ivp, which_prior, **kwargs)
    elif which_prior in ioup_family:
        return _string_to_prior_ioup(ivp, which_prior, **kwargs)
    elif which_prior in matern_family:
        return _string_to_prior_matern(ivp, which_prior, **kwargs)
    else:
        raise RuntimeError("It should have been impossible to "
                           "reach this point.")


def _string_to_prior_ibm(ivp, which_prior, **kwargs):
    """
    """
    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if which_prior == "ibm1":
        return prior.IBM(1, ivp.ndim, diffconst)
    elif which_prior == "ibm2":
        return prior.IBM(2, ivp.ndim, diffconst)
    elif which_prior == "ibm3":
        return prior.IBM(3, ivp.ndim, diffconst)
    elif which_prior == "ibm4":
        return prior.IBM(4, ivp.ndim, diffconst)
    else:
        raise RuntimeError("It should have been impossible to "
                           "reach this point.")


def _string_to_prior_ioup(_ivp, _which_prior, **kwargs):
    """
    """
    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if "driftspeed" in kwargs.keys():
        driftspeed = kwargs["driftspeed"]
    else:
        driftspeed = 1.0
    if _which_prior == "ioup1":
        return prior.IOUP(1, _ivp.ndim, driftspeed, diffconst)
    elif _which_prior == "ioup2":
        return prior.IOUP(2, _ivp.ndim, driftspeed, diffconst)
    elif _which_prior == "ioup3":
        return prior.IOUP(3, _ivp.ndim, driftspeed, diffconst)
    elif _which_prior == "ioup4":
        return prior.IOUP(4, _ivp.ndim, driftspeed, diffconst)
    else:
        raise RuntimeError("It should have been impossible to "
                           "reach this point.")


def _string_to_prior_matern(ivp, which_prior, **kwargs):
    """
    """
    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if "lengthscale" in kwargs.keys():
        lengthscale = kwargs["lengthscale"]
    else:
        lengthscale = 1.0
    if which_prior == "matern32":
        return prior.Matern(1, ivp.ndim, lengthscale, diffconst)
    elif which_prior == "matern52":
        return prior.Matern(2, ivp.ndim, lengthscale, diffconst)
    elif which_prior == "matern72":
        return prior.Matern(3, ivp.ndim, lengthscale, diffconst)
    elif which_prior == "matern92":
        return prior.Matern(4, ivp.ndim, lengthscale, diffconst)
    else:
        raise RuntimeError("It should have been impossible to "
                           "reach this point.")


def _string_to_filter(_ivp, _prior, _method, **kwargs):
    """
    """
    if "evlvar" in kwargs.keys():
        evlvar = kwargs["evlvar"]
    else:
        evlvar = 0.0
    if _method == "ekf0" or _method == "eks0":
        return ivptofilter.ivp_to_ekf0(_ivp, _prior, evlvar)
    elif _method == "ekf1" or _method == "eks1":
        return ivptofilter.ivp_to_ekf1(_ivp, _prior, evlvar)
    elif _method == "ukf" or _method == "uks":
        return ivptofilter.ivp_to_ukf(_ivp, _prior, evlvar)
    else:
        raise TypeError("Type of filter not supported.")


def _step_to_steprule(stp):
    """
    """
    return steprule.ConstantSteps(stp)


def _step_to_adaptive_steprule(_tol, _prior, **kwargs):
    """
    """
    convrate = _prior.ordint + 1
    return steprule.AdaptiveSteps(_tol, convrate, **kwargs)
