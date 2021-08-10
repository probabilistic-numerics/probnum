import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq


@pytest.fixture
def rng():
    return np.random.default_rng(seed=123)


@pytest.fixture
def ivp():
    y0 = 20.0 * np.ones(2)
    return diffeq_zoo.lotkavolterra(t0=0.0, tmax=0.25, y0=y0)


@pytest.mark.parametrize("method", ["RK23", "RK45"])
@pytest.mark.parametrize("perturb", ["step-lognormal", "step-uniform"])
@pytest.mark.parametrize("noise_scale", [0.01, 100])
@pytest.mark.parametrize("step", [0.01, None])
@pytest.mark.parametrize("tolerance", [0.1, np.array([0.09, 0.10])])
@pytest.mark.parametrize("time_stops", [[0.15, 0.16], None])
def test_adaptive_solver_successful(
    rng, ivp, method, perturb, noise_scale, step, tolerance, time_stops
):
    """The solver terminates successfully for all sorts of parametrizations."""
    sol = diffeq.perturbsolve_ivp(
        f=ivp.f,
        t0=ivp.t0,
        tmax=ivp.tmax,
        y0=ivp.y0,
        rng=rng,
        noise_scale=noise_scale,
        perturb=perturb,
        adaptive=True,
        atol=tolerance,
        rtol=tolerance,
        method=method,
        step=step,
        time_stops=time_stops,
    )
    # Successful return value as documented
    assert isinstance(sol, diffeq.ODESolution)

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

    wrong_methods = ["DOP853", "Radau", "LSODA", "BDF"]
    for wrong_method in wrong_methods:
        with pytest.raises(ValueError):
            diffeq.perturbsolve_ivp(f, t0, tmax, y0, rng=rng, method=wrong_method)


def test_wrong_perturb_raises_error(ivp):
    """Perturbation-styles that are not in the list raise errors."""
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    wrong_perturbs = ["step-something", "state"]
    for wrong_perturb in wrong_perturbs:
        with pytest.raises(ValueError):
            diffeq.perturbsolve_ivp(f, t0, tmax, y0, rng=rng, perturb=wrong_perturb)


def test_no_step_or_tol_info_raises_error(ivp, rng):
    """Providing neither a step-size nor a tolerance raises an error."""
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0

    with pytest.raises(ValueError):
        diffeq.perturbsolve_ivp(
            f, t0, tmax, y0, rng, step=None, adaptive=True, atol=None, rtol=None
        )
