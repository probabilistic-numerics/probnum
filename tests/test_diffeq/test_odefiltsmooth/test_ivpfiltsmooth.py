"""Test the ODE filter on a more granular level.

The tests in here go through probsolve_ivp(), but do not directly test the interface,
but the implementation. Therefore this test module is named w.r.t. ivpfiltsmooth.py.
"""
import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randprocs


@pytest.fixture
def ivp():
    y0 = np.array([0.1])
    return diffeq_zoo.logistic(t0=0.0, tmax=1.5, y0=y0)


@pytest.fixture
def step():
    return 0.2


@pytest.fixture
def sol(ivp, step):
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0
    return diffeq.probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        method="ek0",
        algo_order=1,
        adaptive=False,
        step=step,
        diffusion_model="constant",
        dense_output=False,
    )


def test_first_iteration(ivp, sol):
    """Test whether first few means and covariances coincide with Proposition 1 in
    Schober et al., 2019."""
    state_rvs = sol.kalman_posterior.states
    ms, cs = state_rvs.mean, state_rvs.cov

    exp_mean = np.array([ivp.y0, ivp.f(0, ivp.y0)])
    np.testing.assert_allclose(ms[0], exp_mean[:, 0], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(cs[0], np.zeros((2, 2)), atol=1e-14, rtol=1e-14)


def test_second_iteration(ivp, sol, step):
    """Test whether first few means and covariances coincide with Prop.

    1 in Schober et al., 2019.
    """

    state_rvs = sol.kalman_posterior.states

    ms, cs = state_rvs.mean, state_rvs.cov

    y0 = ivp.y0
    z0 = ivp.f(0, y0)
    z1 = ivp.f(0, y0 + step * z0)
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
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    y0 = ivp.y0
    sol_small_step = diffeq.probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        step=step_small,
        algo_order=algo_order,
        adaptive=False,
        diffusion_model="dynamic",
    )
    sol_large_step = diffeq.probsolve_ivp(
        f,
        t0,
        tmax,
        y0,
        step=step_large,
        algo_order=algo_order,
        adaptive=False,
        diffusion_model="dynamic",
    )

    # Check that the final point is identical (sanity check)
    np.testing.assert_allclose(
        sol_small_step.locations[-1], sol_large_step.locations[-1]
    )

    # Compute both errors
    ref_sol = ivp.solution(sol_small_step.locations[-1])
    err_small_step = np.linalg.norm(ref_sol - sol_small_step.states[-1].mean)
    err_large_step = np.linalg.norm(ref_sol - sol_large_step.states[-1].mean)

    # Non-strict rtol, bc this test is flaky by construction
    # As long as rtol < 1., this test seems meaningful.
    np.testing.assert_allclose(
        err_small_step, expected_decay * err_large_step, rtol=0.95
    )


def test_callback(ivp, step):
    d = ivp.dimension
    nu = 1

    steprule = diffeq.stepsize.ConstantSteps(step)
    prior_process = randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0, num_derivatives=nu, wiener_process_dimension=d
    )
    info_op = diffeq.odefiltsmooth.information_operators.ODEResidual(
        num_prior_derivatives=nu, ode_dimension=d
    )
    approx = diffeq.odefiltsmooth.approx_strategies.EK0()
    with_smoothing = True
    init_strat = diffeq.odefiltsmooth.initialization_routines.RungeKuttaInitialization()
    solver = diffeq.odefiltsmooth.GaussianIVPFilter(
        steprule=steprule,
        prior_process=prior_process,
        information_operator=info_op,
        approx_strategy=approx,
        with_smoothing=with_smoothing,
        initialization_routine=init_strat,
    )

    t0, tmax = ivp.t0, ivp.tmax
    t = 0.5 * (t0 + tmax)
    replace = lambda x: x
    condition = lambda state: state.t == t
    callback = diffeq.callbacks.DiscreteCallback(replace=replace, condition=condition)

    solver.solve(ivp, stop_at=[t], callbacks=callback)
