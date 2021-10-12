"""Perturbation-based probabilistic ODE solver."""
import numpy as np
import scipy.integrate

from probnum import problems
from probnum.diffeq import perturbed, stepsize

__all__ = ["perturbsolve_ivp"]


METHODS = {
    "RK45": scipy.integrate.RK45,
    "RK23": scipy.integrate.RK23,
}
"""Implemented Scipy solvers."""

PERTURBS = {
    "step-lognormal": perturbed.step.PerturbedStepSolver.construct_with_lognormal_perturbation,
    "step-uniform": perturbed.step.PerturbedStepSolver.construct_with_uniform_perturbation,
}
"""Implemented perturbation-styles."""


def perturbsolve_ivp(
    f,
    t0,
    tmax,
    y0,
    rng,
    method="RK45",
    perturb="step-lognormal",
    noise_scale=10.0,
    adaptive=True,
    atol=1e-6,
    rtol=1e-3,
    step=None,
    time_stops=None,
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

            * `RK45` (default): Explicit Runge-Kutta method of order 5(4) [2]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [3]_. Can be applied in
              the complex domain.
            * `RK23`: Explicit Runge-Kutta method of order 3(2) [4]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.

        Other integrators are not supported currently.
    perturb :
        Which perturbation style to use.
        Currently, the following methods are supported:

            * `step-lognormal`: Perturbed-step (aka random time-step numerical integration) method
              with lognormally distributed perturbation of the step-size [1]_.
            * `step-uniform`: Perturbed-step (aka random time-step numerical integration) method
              with uniformly distributed perturbation of the step-size [1]_.

        Other integrators are not supported currently.
    noise_scale :
        Scale of the perturbation. Optional. Default is 10.0. The magnitude of this parameter
        significantly impacts the width of the posterior.
    adaptive :
        Whether to use adaptive steps or not. Default is `True`.
    atol :
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-6``.
    rtol :
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-3``.
    step :
        Step size. If atol and rtol are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen as prescribed by :meth:`propose_firststep`.
    time_stops: np.ndarray
        Time-points through which the solver must step. Optional. Default is None.

    Examples
    --------
    >>> from probnum.diffeq import perturbsolve_ivp
    >>> import numpy as np

    Solve a simple logistic ODE with fixed steps.
    Per default, `perturbsolve_ivp` uses a perturbed-step solver with lognormal perturbation.

    >>> rng = np.random.default_rng(seed=2)
    >>>
    >>> def f(t, x):
    ...     return 4*x*(1-x)
    >>>
    >>> y0 = np.array([0.15])
    >>> t0, tmax = 0., 1.5
    >>> solution = perturbsolve_ivp(f, t0, tmax, y0, rng=rng, step=0.25, method="RK23", adaptive=False)
    >>> print(np.round(solution.states.mean, 3))
    [[0.15 ]
     [0.325]
     [0.56 ]
     [0.772]
     [0.893]
     [0.964]
     [0.989]]

    Each solution is the result of a randomly-perturbed call of an underlying Runge-Kutta solver.
    Therefore, if you call it again, the output will be different:

    >>> other_solution = perturbsolve_ivp(f, t0, tmax, y0, rng=rng, step=0.25, method="RK23", adaptive=False)
    >>> print(np.round(other_solution.states.mean, 3))
    [[0.15 ]
     [0.319]
     [0.57 ]
     [0.785]
     [0.908]
     [0.968]
     [0.989]]


    Other methods, such as `RK45` (as well as other perturbation styles) are easily accessible.
    Let us solve the same equation, with an adaptive RK45 solver that uses uniformly perturbed steps.

    >>> solution = perturbsolve_ivp(f, t0, tmax, y0, rng=rng, atol=1e-5, rtol=1e-6, method="RK45", perturb="step-uniform", adaptive=True)
    >>> print(np.round(solution.states.mean, 3))
    [[0.15 ]
     [0.152]
     [0.167]
     [0.26 ]
     [0.431]
     [0.646]
     [0.849]
     [0.883]
     [0.915]
     [0.953]
     [0.976]
     [0.986]]

    References
    ----------
    .. [1] Abdulle, A. and Garegnani, G..
        Random time step probabilistic methods for uncertainty quantification in chaotic and geometric numerical integration.
        Statistics and Computing. 2020.
    .. [2] J. R. Dormand, P. J. Prince..
        A family of embedded Runge-Kutta formulae.
        Journal of Computational and Applied Mathematics, Vol. 6, No. 1, pp. 19-26, 1980.
    .. [3] L. W. Shampine.
        Some Practical Runge-Kutta Formulas.
        Mathematics of Computation, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [4] P. Bogacki, L.F. Shampine.
        A 3(2) Pair of Runge-Kutta Formulas.
        Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """

    ivp = problems.InitialValueProblem(t0=t0, tmax=tmax, y0=np.asarray(y0), f=f)

    # Create steprule
    if adaptive is True:
        if atol is None or rtol is None:
            raise ValueError(
                "Please provide absolute and relative tolerance for adaptive steps."
            )
        firststep = step if step is not None else stepsize.propose_firststep(ivp)
        steprule = stepsize.AdaptiveSteps(firststep=firststep, atol=atol, rtol=rtol)
    else:
        steprule = stepsize.ConstantSteps(step)

    if method not in METHODS.keys():
        msg1 = f"Parameter method='{method}' is not supported. "
        msg2 = f"Possible values are {list(METHODS.keys())}."
        errormsg = msg1 + msg2
        raise ValueError(errormsg)
    scipy_solver = METHODS[method]
    wrapped_scipy_solver = perturbed.scipy_wrapper.WrappedScipyRungeKutta(
        scipy_solver, steprule=steprule
    )

    if perturb not in PERTURBS.keys():
        msg1 = f"Parameter perturb='{perturb}' is not supported. "
        msg2 = f"Possible values are {list(PERTURBS.keys())}."
        errormsg = msg1 + msg2
        raise ValueError(errormsg)

    perturbed_solver = PERTURBS[perturb](
        rng=rng,
        solver=wrapped_scipy_solver,
        noise_scale=noise_scale,
    )

    return perturbed_solver.solve(ivp=ivp, stop_at=time_stops)
