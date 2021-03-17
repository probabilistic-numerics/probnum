"""Test the ODE filter on a more granular level.

The tests in here go through probsolve_ivp(), but do not directly test
the interface, but the implementation. Therefore this test module is
named w.r.t. ivpfiltsmooth.py.
"""
import numpy as np
import pytest

from probnum.diffeq import ode
from probnum.diffeq.odefiltsmooth import probsolve_ivp
from probnum.randvars import Constant


@pytest.fixture
def ivp():
    initrv = Constant(0.1 * np.ones(1))
    return ode.logistic([0.0, 1.5], initrv)


@pytest.fixture
def step():
    return 0.2


@pytest.fixture
def sol(ivp, step):
    f = ivp.rhs
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean
    return probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        method="ek0",
        algo_order=1,
        adaptive=False,
        step=step,
    )


def test_first_iteration(ivp, sol):
    """Test whether first few means and covariances coincide with Proposition 1 in
    Schober et al., 2019."""
    state_rvs = sol.kalman_posterior.filtering_posterior.state_rvs
    ms, cs = state_rvs.mean, state_rvs.cov

    exp_mean = np.array([ivp.initrv.mean, ivp.rhs(0, ivp.initrv.mean)])
    np.testing.assert_allclose(ms[0], exp_mean[:, 0], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(cs[0], np.zeros((2, 2)), atol=1e-14, rtol=1e-14)


def test_second_iteration(ivp, sol, step):
    """Test whether first few means and covariances coincide with Prop.

    1 in Schober et al., 2019.
    """

    state_rvs = sol.kalman_posterior.filtering_posterior.state_rvs
    ms, cs = state_rvs.mean, state_rvs.cov

    y0 = ivp.initrv.mean
    z0 = ivp.rhs(0, y0)
    z1 = ivp.rhs(0, y0 + step * z0)
    exp_mean = np.array([y0 + 0.5 * step * (z0 + z1), z1])
    np.testing.assert_allclose(ms[1], exp_mean[:, 0], rtol=1e-14)


@pytest.mark.parametrize("algo_order", [1, 2, 3])
def test_convergence_error(ivp, algo_order):
    """Assert that by halfing the step-size, the error of the small step is roughly the
    error of the large step multiplied with (small / large)**(nu)"""

    # Set up two different step-sizes
    step_large = 0.2
    step_small = 0.5 * step_large
    expected_decay = (step_small / step_large) ** algo_order

    # Solve IVP with both step-sizes
    f = ivp.rhs
    t0, tmax = ivp.timespan
    y0 = ivp.initrv.mean
    sol_small_step = probsolve_ivp(
        f, t0, tmax, y0, step=step_small, algo_order=algo_order, adaptive=False
    )
    sol_large_step = probsolve_ivp(
        f, t0, tmax, y0, step=step_large, algo_order=algo_order, adaptive=False
    )

    # Check that the final point is identical (sanity check)
    np.testing.assert_allclose(sol_small_step.t[-1], sol_large_step.t[-1])

    # Compute both errors
    ref_sol = ivp.solution(sol_small_step.t[-1])
    err_small_step = np.linalg.norm(ref_sol - sol_small_step.y[-1].mean)
    err_large_step = np.linalg.norm(ref_sol - sol_large_step.y[-1].mean)

    # Non-strict rtol, bc this test is flaky by construction
    # As long as rtol < 1., this test seems meaningful.
    np.testing.assert_allclose(
        err_small_step, expected_decay * err_large_step, rtol=0.9
    )
