import numpy as np
import pytest

from probnum.diffeq import ode
from probnum.diffeq.odefiltsmooth import KalmanODESolution, probsolve_ivp
from probnum.randvars import Constant


@pytest.fixture
def ivp():
    initrv = Constant(20.0 * np.ones(2))
    return ode.lotkavolterra([0.0, 0.25], initrv)


@pytest.mark.parametrize("method", ["EK0", "EK1"])
@pytest.mark.parametrize(
    "algo_order",
    [1, 2, 5],
)
@pytest.mark.parametrize("dense_output", [True, False])
@pytest.mark.parametrize("step", [0.01, None])
def test_adaptive_solver_successful(ivp, method, algo_order, dense_output, step):
    """The solver terminates successfully for all sorts of parametrizations."""
    f = ivp.rhs
    df = ivp.jacobian
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    sol = probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        df=df,
        adaptive=True,
        atol=1e-1,
        rtol=1e-1,
        algo_order=algo_order,
        method=method,
        dense_output=dense_output,
        step=step,
    )
    # Successful return value as documented
    assert isinstance(sol, KalmanODESolution)

    # Adaptive steps are not evenly distributed
    step_diff = np.diff(sol.t)
    step_ratio = np.amin(step_diff) / np.amax(step_diff)
    assert step_ratio < 0.5


#
# def test_wrong_prior_raises_error(ivp):
#     """Priors that are not in the list raise errors."""
#     f = ivp.rhs
#     t0, tmax = ivp.timespan
#     y0 = ivp.initrv.mean
#
#     # Anything that is no {IBM, IOUP, MAT} + Number is wrong
#     # (the Matern number must end on a 2).
#     for which_prior in ["IBM_5", "IOUPX5", "MAT112Y", "MAT33"]:
#         with pytest.raises(ValueError):
#             probsolve_ivp(
#                 f,
#                 t0,
#                 tmax,
#                 y0,
#                 which_prior=which_prior,
#             )


def test_wrong_method_raises_error(ivp):
    """Methods that are not in the list raise errors."""
    f = ivp.rhs
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    # UK1 does not exist anymore
    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, method="UK")


def test_no_step_or_tol_info_raises_error(ivp):
    """Providing neither a step-size nor a tolerance raises an error."""
    f = ivp.rhs
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, step=None, adaptive=True, atol=None, rtol=None)
