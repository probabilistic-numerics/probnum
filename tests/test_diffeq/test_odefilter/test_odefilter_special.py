"""Test the ODE filter on a more granular level.

The tests in here go through probsolve_ivp(), but do not directly test the interface,
but the implementation. Therefore this test module is named w.r.t. ivpfiltsmooth.py.
"""
import numpy as np
import pytest

from probnum import diffeq, randprocs
import probnum.problems.zoo.diffeq as diffeq_zoo


@pytest.fixture(name="ivp")
def fixture_ivp():
    y0 = np.array([0.1])
    return diffeq_zoo.logistic(t0=0.0, tmax=1.5, y0=y0)


@pytest.fixture(name="step_large")
def fixture_step_large():
    return 0.2


@pytest.fixture(name="steprule_large")
def fixture_steprule_large(step_large):
    """Constant step-sizes."""
    return diffeq.stepsize.ConstantSteps(step_large)


@pytest.fixture(name="step_small")
def fixture_step_small(step_large):
    return 0.5 * step_large


@pytest.fixture(name="steprule_small")
def fixture_steprule_small(step_small):
    """Constant step-sizes."""
    return diffeq.stepsize.ConstantSteps(step_small)


@pytest.fixture(name="prior_iwp")
def fixture_prior_iwp(ivp, algo_order):
    return randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=algo_order,
        wiener_process_dimension=ivp.dimension,
        diffuse=True,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


@pytest.fixture(name="diffusion_model")
def fixture_diffusion_model():
    return randprocs.markov.continuous.ConstantDiffusion()


@pytest.fixture(name="odefilter_large_step")
def fixture_odefilter_large_step(ivp, steprule_large, prior_iwp, diffusion_model):
    return diffeq.odefilter.ODEFilter(
        steprule=steprule_large,
        prior_process=prior_iwp,
        diffusion_model=diffusion_model,
        init_routine=diffeq.odefilter.init_routines.Stack(),
        with_smoothing=False,
    )


@pytest.fixture(name="odefilter_small_step")
def fixture_odefilter_small_step(ivp, steprule_small, prior_iwp, diffusion_model):
    return diffeq.odefilter.ODEFilter(
        steprule=steprule_small,
        prior_process=prior_iwp,
        diffusion_model=diffusion_model,
        init_routine=diffeq.odefilter.init_routines.Stack(),
        with_smoothing=False,
    )


@pytest.fixture(name="solution_large_step")
def fixture_solution_large_step(ivp, odefilter_large_step):
    return odefilter_large_step.solve(ivp)


@pytest.fixture(name="solution_small_step")
def fixture_solution_small_step(ivp, odefilter_small_step):
    return odefilter_small_step.solve(ivp)


@pytest.mark.parametrize("algo_order", [1])
def test_first_iteration(ivp, solution_large_step):
    """Compare the first iteration to a closed-form solution.

    First mean and cov coincides with Proposition 1 in Schober et al., 2019.
    """
    state_rvs = solution_large_step.kalman_posterior.states
    ms, cs = state_rvs.mean, state_rvs.cov

    exp_mean = np.array([ivp.y0, ivp.f(0, ivp.y0)])
    np.testing.assert_allclose(ms[0], exp_mean[:, 0], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(cs[0], np.zeros((2, 2)), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("algo_order", [1])
def test_second_iteration(ivp, solution_large_step, step_large):
    """Compare the first iteration to a closed-form solution.

    Second mean and cov coincides with Prop. 1 in Schober et al., 2019.
    """

    state_rvs = solution_large_step.kalman_posterior.states
    ms, cs = state_rvs.mean, state_rvs.cov

    y0 = ivp.y0
    z0 = ivp.f(0, y0)
    z1 = ivp.f(0, y0 + step_large * z0)
    exp_mean = np.array([y0 + 0.5 * step_large * (z0 + z1), z1])
    np.testing.assert_allclose(ms[1], exp_mean[:, 0], rtol=1e-14)


@pytest.mark.parametrize("algo_order", [1, 2])
def test_convergence_error(
    ivp, step_small, step_large, solution_large_step, solution_small_step, algo_order
):
    """Assert that by halfing the step-size, the error of the small step is roughly the
    error of the large step multiplied with (small / large)**(nu)"""

    # Check that the final point is identical (sanity check)
    np.testing.assert_allclose(
        solution_small_step.locations[-1], solution_large_step.locations[-1]
    )

    # Compute both errors
    ref_sol = ivp.solution(solution_small_step.locations[-1])
    err_small_step = np.linalg.norm(ref_sol - solution_small_step.states[-1].mean)
    err_large_step = np.linalg.norm(ref_sol - solution_large_step.states[-1].mean)

    # Non-strict rtol, bc this test is flaky by construction
    # As long as rtol < 1., this test seems meaningful.
    error_decay_expected = (step_small / step_large) ** algo_order
    error_decay_received = err_small_step / err_large_step
    np.testing.assert_allclose(error_decay_received, error_decay_expected, rtol=0.95)


@pytest.fixture(name="t_span_midpoint")
def fixture_t_span_midpoint(ivp):
    t0, tmax = ivp.t0, ivp.tmax
    t = 0.5 * (t0 + tmax)
    return t


@pytest.fixture(name="callback")
def fixture_callback(ivp, t_span_midpoint):
    replace = lambda x: x
    condition = lambda state: state.t == t_span_midpoint
    callback = diffeq.callbacks.DiscreteCallback(replace=replace, condition=condition)
    return callback


@pytest.mark.parametrize("algo_order", [1])
def test_callback(ivp, callback, t_span_midpoint, odefilter_large_step):
    odefilter_large_step.solve(ivp, stop_at=[t_span_midpoint], callbacks=callback)
