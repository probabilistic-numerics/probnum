import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum.diffeq import probsolve_ivp
from probnum.diffeq.odefiltsmooth import KalmanODESolution


@pytest.fixture
def ivp():
    y0 = 20.0 * np.ones(2)
    return diffeq_zoo.lotkavolterra(t0=0.0, tmax=0.25, y0=y0)


@pytest.mark.parametrize("method", ["EK0", "EK1"])
@pytest.mark.parametrize(
    "algo_order",
    [1, 2, 5],
)
@pytest.mark.parametrize("dense_output", [True, False])
@pytest.mark.parametrize("step", [0.01, None])
@pytest.mark.parametrize("diffusion_model", ["constant", "dynamic"])
@pytest.mark.parametrize("tolerance", [0.1, np.array([0.09, 0.10])])
@pytest.mark.parametrize("time_stops", [[0.15, 0.16], None])
def test_adaptive_solver_successful(
    ivp,
    method,
    algo_order,
    dense_output,
    step,
    diffusion_model,
    tolerance,
    time_stops,
):
    """The solver terminates successfully for all sorts of parametrizations."""
    f = ivp.f
    df = ivp.df
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    sol = probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        df=df,
        adaptive=True,
        atol=tolerance,
        rtol=tolerance,
        algo_order=algo_order,
        method=method,
        dense_output=dense_output,
        step=step,
        time_stops=time_stops,
    )
    # Successful return value as documented
    assert isinstance(sol, KalmanODESolution)

    # Adaptive steps are not evenly distributed
    step_diff = np.diff(sol.locations)
    step_ratio = np.amin(step_diff) / np.amax(step_diff)
    assert step_ratio < 0.5

    if time_stops is not None:
        for t in time_stops:
            assert t in sol.locations


def test_wrong_method_raises_error(ivp):
    """Methods that are not in the list raise errors."""
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    # UK1 does not exist anymore
    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, method="UK")


def test_no_step_or_tol_info_raises_error(ivp):
    """Providing neither a step-size nor a tolerance raises an error."""
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, step=None, adaptive=True, atol=None, rtol=None)


def test_wrong_diffusion_raises_error(ivp):
    """Methods that are not in the list raise errors."""
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    # UK1 does not exist anymore
    with pytest.raises(ValueError):
        probsolve_ivp(f, t0, tmax, y0, diffusion_model="something_wrong")
