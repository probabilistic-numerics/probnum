import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import base, rk
from scipy.integrate._ivp.common import OdeSolution

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randvars


@pytest_cases.fixture
@pytest_cases.parametrize_with_cases(
    "testsolver, scipysolver, ode", cases=".test_wrapped_scipy_cases"
)
def solvers(testsolver, scipysolver, ode):
    return testsolver, scipysolver, ode


@pytest.fixture
def y():
    return np.array([0.1])


@pytest.fixture
def times():
    return [0.0, 1.0]


@pytest.fixture
def start_point():
    return 0.5


@pytest.fixture
def stop_point():
    return 0.6


@pytest.fixture
def dense_output():
    return [base.DenseOutput(0, 1)]


@pytest.fixture
def list_of_randvars():
    return list([randvars.Constant(1)])


@pytest.fixture
def doprisolver():
    y0 = np.array([0.1])
    ode = diffeq_zoo.logistic(t0=0.0, tmax=1.0, y0=y0)
    return rk.DOP853(ode.f, ode.t0, y0, ode.tmax)


def test_init(doprisolver):
    with pytest.raises(TypeError):
        steprule = diffeq.stepsize.ConstantSteps(0.2)  # irrelevant value
        diffeq.perturbed.scipy_wrapper.WrappedScipyRungeKutta(
            rk.DOP853, steprule=steprule
        )


def test_initialise(solvers):
    testsolver, scipysolver, ode = solvers
    state = testsolver.initialize(ode)
    time_scipy = scipysolver.t
    state_scipy = scipysolver.y
    np.testing.assert_allclose(state.t, time_scipy, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(state.rv.mean[0], state_scipy[0], atol=1e-13, rtol=1e-13)


def test_step_execution(solvers):
    testsolver, scipysolver, ode = solvers
    scipysolver.step()

    # perform step of the same size
    teststate = diffeq.ODESolverState(
        ivp=ode,
        rv=randvars.Constant(scipysolver.y_old),
        t=scipysolver.t_old,
        error_estimate=None,
        reference_state=None,
    )
    testsolver.initialize(ode)
    dt = scipysolver.t - scipysolver.t_old
    new_state = testsolver.attempt_step(teststate, dt)
    np.testing.assert_allclose(scipysolver.y, new_state.rv.mean)


def test_step_variables(solvers, y, start_point, stop_point):
    testsolver, scipysolver, ode = solvers

    teststate = diffeq.ODESolverState(
        ivp=ode,
        rv=randvars.Constant(y),
        t=start_point,
        error_estimate=None,
        reference_state=None,
    )
    testsolver.initialize(ode)
    solver_y_new = testsolver.attempt_step(teststate, dt=stop_point - start_point)
    y_new, f_new = rk.rk_step(
        scipysolver.fun,
        start_point,
        y,
        scipysolver.f,
        stop_point - start_point,
        scipysolver.A,
        scipysolver.B,
        scipysolver.C,
        scipysolver.K,
    )

    # error estimation is correct
    scipy_error_estimation = scipysolver._estimate_error(
        scipysolver.K, stop_point - start_point
    )
    np.testing.assert_allclose(
        solver_y_new.error_estimate, scipy_error_estimation, atol=1e-13, rtol=1e-13
    )

    # locations are correct
    np.testing.assert_allclose(
        testsolver.solver.t_old, start_point, atol=1e-13, rtol=1e-13
    )
    np.testing.assert_allclose(testsolver.solver.t, stop_point, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        testsolver.solver.h_previous,
        stop_point - start_point,
        atol=1e-13,
        rtol=1e-13,
    )

    # evaluations are correct
    np.testing.assert_allclose(testsolver.solver.y_old, y, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(testsolver.solver.y, y_new, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        testsolver.solver.h_abs, stop_point - start_point, atol=1e-13, rtol=1e-13
    )
    np.testing.assert_allclose(testsolver.solver.f, f_new, atol=1e-13, rtol=1e-13)


def test_dense_output(solvers):
    testsolver, scipysolver, ode = solvers

    # perform steps of the same size
    testsolver.initialize(ode)
    scipysolver.step()
    teststate = diffeq.ODESolverState(
        ivp=ode,
        rv=randvars.Constant(scipysolver.y_old),
        t=scipysolver.t_old,
        error_estimate=None,
        reference_state=None,
    )
    state = testsolver.attempt_step(
        state=teststate, dt=scipysolver.t - scipysolver.t_old
    )

    # sanity check: the steps are the same
    # (this is contained in a different test already, but if this one
    # does not work, the dense output test below is meaningless)
    np.testing.assert_allclose(scipysolver.y, state.rv.mean)

    testsolver_dense = testsolver.dense_output()
    scipy_dense = scipysolver._dense_output_impl()

    t_old = scipysolver.t_old
    t = scipysolver.t
    t_mid = (t_old + t) / 2.0

    for time in [t_old, t, t_mid]:
        test_dense = testsolver_dense(time)
        ref_dense = scipy_dense(time)
        np.testing.assert_allclose(
            test_dense,
            ref_dense,
            atol=1e-13,
            rtol=1e-13,
        )


def test_rvlist_to_odesol(times, dense_output, list_of_randvars):
    scipy_sol = OdeSolution(times, dense_output)
    probnum_solution = diffeq.perturbed.scipy_wrapper.WrappedScipyODESolution(
        scipy_sol, list_of_randvars
    )
    assert isinstance(
        probnum_solution, diffeq.perturbed.scipy_wrapper.WrappedScipyODESolution
    )
