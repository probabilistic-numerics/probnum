import numpy as np
import pytest

from probnum.diffeq import ode
from probnum.diffeq.odefiltsmooth import KalmanODESolution, probsolve_ivp
from probnum.random_variables import Constant


@pytest.fixture
def ivp():
    initrv = Constant(20.0 * np.ones(2))
    return ode.lotkavolterra([0.0, 0.25], initrv)


@pytest.mark.parametrize("method", ["EK0", "EK1"])
@pytest.mark.parametrize(
    "which_prior",
    [
        "IBM1",
        "IBM2",
        "IBM3",
        "IBM4",
        "IOUP1",
        "IOUP2",
        "IOUP3",
        "IOUP4",
        "MAT32",
        "MAT52",
        "MAT72",
        "MAT92",
    ],
)
@pytest.mark.parametrize("dense_output", [True, False])
@pytest.mark.parametrize("step", [0.01, None])
def test_adaptive_solver_successful(ivp, method, which_prior, dense_output, step):
    """The solver terminates successfully for all sorts of parametrizations."""
    f = ivp.rhs
    df = ivp.jacobian
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    lengthscale = 10.0
    driftspeed = 1.0

    sol = probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        df=df,
        atol=1e-1,
        rtol=1e-1,
        which_prior=which_prior,
        method=method,
        dense_output=dense_output,
        step=step,
        driftspeed=driftspeed,
        lengthscale=lengthscale,
    )
    # Successful return value as documented
    assert isinstance(sol, KalmanODESolution)

    # Adaptive steps are not evenly distributed
    step_diff = np.diff(sol.t)
    step_ratio = np.amin(step_diff) / np.amax(step_diff)
    assert step_ratio < 0.5


def test_wrong_prior_raises_error(ivp):
    """Priors that are not in the list raise errors."""
    f = ivp.rhs
    df = ivp.jacobian
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    for which_prior in ["IBM5", "IOUP5", "MAT112"]:
        with pytest.raises(ValueError):
            probsolve_ivp(
                f,
                t0,
                tmax,
                y0,
                df=df,
                which_prior=which_prior,
            )


def test_wrong_filter_raises_error(ivp):
    """Priors that are not in the list raise errors."""
    f = ivp.rhs
    df = ivp.jacobian
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean

    # UK1 does not exist
    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, df=df, method="UK1")
