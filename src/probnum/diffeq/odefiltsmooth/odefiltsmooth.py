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
from probnum.diffeq.odefiltsmooth import prior, ivp2filter
from probnum.diffeq.odefiltsmooth.ivpfiltsmooth import GaussianIVPFilter


def probsolve_ivp(
    ivp,
    method="ekf0",
    which_prior="ibm1",
    tol=None,
    step=None,
    firststep=None,
    precond_step=None,
    **kwargs
):
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

    precond_step : float, optional
        Expected average step size, used for preconditioning.
        See :class:`ODEPrior` for details.
        Default is ``precond_step=1.0`` which amounts to no
        preconditioning. If constant step size, precond_step is
        overwritten with the actual step size to provide optimal
        preconditioning.

    Returns
    -------
    solution : ODESolution
        Solution of the ODE problem.

        Contains fields:

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
    ODESolution : Solution of ODE problems

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
    >>> from probnum import random_variables as rvs
    >>> initrv = rvs.Dirac(0.15)
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="ekf0", step=0.1)
    >>> print(solution.y.mean())
    [[0.15      ]
     [0.2076198 ]
     [0.27932997]
     [0.3649165 ]
     [0.46054129]
     [0.55945475]
     [0.65374523]
     [0.73686744]
     [0.8053776 ]
     [0.85895587]
     [0.89928283]
     [0.92882899]
     [0.95007559]
     [0.96515825]
     [0.97577054]
     [0.9831919 ]]

    >>> initrv = rvs.Dirac(0.15)
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="eks1", which_prior="ioup3", step=0.1)
    >>> print(solution.y.mean())
    [[0.15      ]
     [0.20795795]
     [0.28228416]
     [0.36926   ]
     [0.46646494]
     [0.5658788 ]
     [0.66048505]
     [0.74369892]
     [0.81237394]
     [0.86592791]
     [0.90598293]
     [0.9349573 ]
     [0.95544749]
     [0.96968754]
     [0.97947631]
     [0.98614541]]
    """
    gfilt, firststep, stprl = _create_solver_inputs(
        ivp, method, which_prior, tol, step, firststep, precond_step, **kwargs
    )
    with_smoothing = method[-2] == "s"
    solver = GaussianIVPFilter(ivp, gfilt, with_smoothing=with_smoothing)
    solution = solver.solve(firststep=firststep, steprule=stprl, **kwargs)
    return solution


def _create_solver_inputs(
    ivp, method, which_prior, tol, step, firststep, precond_step, **kwargs
):
    """Create the solver object that is used."""
    _check_step_tol(step, tol)
    _check_method(method)
    if precond_step is None:
        precond_step = step or 1.0
    _prior = _string2prior(ivp, which_prior, precond_step, **kwargs)
    if tol is not None:
        stprl = steprule.AdaptiveSteps(tol, _prior.ordint + 1, **kwargs)
        if firststep is None:
            firststep = ivp.tmax - ivp.t0
    else:
        stprl = steprule.ConstantSteps(step)
        firststep = step
    gfilt = _string2filter(ivp, _prior, method, **kwargs)
    return gfilt, firststep, stprl


def _check_step_tol(step, tol):

    both_none = tol is None and step is None
    both_not_none = tol is not None and step is not None
    if both_none or both_not_none:
        errormsg = "Please specify either a tolerance or a step size."
        raise ValueError(errormsg)


def _check_method(method):

    if method not in ["ekf0", "ekf1", "ukf", "eks0", "eks1", "uks"]:
        raise ValueError("Method not supported.")


def _string2prior(ivp, which_prior, precond_step, **kwargs):

    ibm_family = ["ibm1", "ibm2", "ibm3", "ibm4"]
    ioup_family = ["ioup1", "ioup2", "ioup3", "ioup4"]
    matern_family = ["matern32", "matern52", "matern72", "matern92"]
    if which_prior in ibm_family:
        return _string2ibm(ivp, which_prior, precond_step, **kwargs)
    elif which_prior in ioup_family:
        return _string2ioup(ivp, which_prior, precond_step, **kwargs)
    elif which_prior in matern_family:
        return _string2matern(ivp, which_prior, precond_step, **kwargs)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2ibm(ivp, which_prior, precond_step, **kwargs):

    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if which_prior == "ibm1":
        return prior.IBM(1, ivp.ndim, diffconst, precond_step)
    elif which_prior == "ibm2":
        return prior.IBM(2, ivp.ndim, diffconst, precond_step)
    elif which_prior == "ibm3":
        return prior.IBM(3, ivp.ndim, diffconst, precond_step)
    elif which_prior == "ibm4":
        return prior.IBM(4, ivp.ndim, diffconst, precond_step)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2ioup(ivp, which_prior, precond_step, **kwargs):

    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if "driftspeed" in kwargs.keys():
        driftspeed = kwargs["driftspeed"]
    else:
        driftspeed = 1.0
    if which_prior == "ioup1":
        return prior.IOUP(1, ivp.ndim, driftspeed, diffconst, precond_step)
    elif which_prior == "ioup2":
        return prior.IOUP(2, ivp.ndim, driftspeed, diffconst, precond_step)
    elif which_prior == "ioup3":
        return prior.IOUP(3, ivp.ndim, driftspeed, diffconst, precond_step)
    elif which_prior == "ioup4":
        return prior.IOUP(4, ivp.ndim, driftspeed, diffconst, precond_step)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2matern(ivp, which_prior, precond_step, **kwargs):

    if "diffconst" in kwargs.keys():
        diffconst = kwargs["diffconst"]
    else:
        diffconst = 1.0
    if "lengthscale" in kwargs.keys():
        lengthscale = kwargs["lengthscale"]
    else:
        lengthscale = 1.0
    if which_prior == "matern32":
        return prior.Matern(1, ivp.ndim, lengthscale, diffconst, precond_step)
    elif which_prior == "matern52":
        return prior.Matern(2, ivp.ndim, lengthscale, diffconst, precond_step)
    elif which_prior == "matern72":
        return prior.Matern(3, ivp.ndim, lengthscale, diffconst, precond_step)
    elif which_prior == "matern92":
        return prior.Matern(4, ivp.ndim, lengthscale, diffconst, precond_step)
    else:
        raise RuntimeError("It should have been impossible to reach this point.")


def _string2filter(_ivp, _prior, _method, **kwargs):

    if "evlvar" in kwargs.keys():
        evlvar = kwargs["evlvar"]
    else:
        evlvar = 0.0
    if _method in ("ekf0", "eks0"):
        return ivp2filter.ivp2ekf0(_ivp, _prior, evlvar)
    elif _method in ("ekf1", "eks1"):
        return ivp2filter.ivp2ekf1(_ivp, _prior, evlvar)
    elif _method in ("ukf", "uks"):
        return ivp2filter.ivp2ukf(_ivp, _prior, evlvar)
    else:
        raise ValueError("Type of filter not supported.")
