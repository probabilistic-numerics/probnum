"""Perturbation-based probabilistic ODE solver."""


__all__ = ["perturbsolve_ivp"]


def perturbsolve_ivp(
    f,
    t0,
    tmax,
    y0,
    rng,
    method="RK45",
    perturb="step",
    noise_scale=1.0,
    dense_output=True,
    adaptive=True,
    atol=1e-2,
    rtol=1e-2,
    step=None,
):
    r"""Solve an initial value problem with a perturbation-based probabilistic ODE solver.

    Parameters
    ----------
    f :
        ODE vector field.
    t0 :
        Initial time point.
    tmax :
        Final time point.
    y0 :
        Initial value.
    rng :
        Random number generator.
    method :
        Integration method to use.
        The following are available (docs adapted from `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_):
            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [2]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [3]_. Can be applied in
              the complex domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [4]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [5]_.
              Python implementation of the "DOP853" algorithm originally
              written in Fortran [6]_. A 7-th order interpolation polynomial
              accurate to 7-th order is used for the dense output.
              Can be applied in the complex domain.
        Other integrators are not supported currently.
    perturb
        Which perturbation style to use. Currently, only one method is supported:
            * `step`: Perturbed-step (aka random time-step numerical integration) method [1]_.
    noise_scale
        Scale of the perturbation. Optional. Default is 1.0. The size of this parameter
        significantly impacts the width of the posterior.
    dense_output : bool
        Whether we want dense output. Optional. Default is ``True``. For the ODE filter,
        dense output requires smoothing, so if ``dense_output`` is False, no smoothing is performed;
        but when it is ``True``, the filter solution is smoothed.
    adaptive :
        Whether to use adaptive steps or not. Default is `True`.
    atol :
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    rtol :
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    step :
        Step size. If atol and rtol are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen as :math:`0.01 \cdot |y_0|/|f(t_0, y_0)|`.

    References
    ----------
    .. [1] Abdulle, A. and Garegnani, G.
        Random time step probabilistic methods for uncertainty quantification in chaotic and geometric numerical integration.
        Statistics and Computing. 2020.
    .. [2] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [3] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [4] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [5] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [6] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """

    raise NotImplementedError
