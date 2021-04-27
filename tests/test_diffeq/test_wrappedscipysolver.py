import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import base, rk
from scipy.integrate._ivp.common import OdeSolution

from probnum import randvars
from probnum.diffeq import odesolution, wrappedscipyodesolution


@pytest_cases.fixture
@pytest_cases.parametrize_with_cases(
    "testsolver, scipysolver", cases=".test_wrappedscipy_cases"
)
def solver_case(testsolver, scipysolver):
    return testsolver, scipysolver


# to test the dense output, new solvers have to be set up
@pytest_cases.fixture
@pytest_cases.parametrize_with_cases(
    "testsolver, scipysolver", cases=".test_wrappedscipy_cases"
)
def dense_case(testsolver, scipysolver):
    return testsolver, scipysolver


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


def test_initialise(solver_case):
    testsolver, scipysolver = solver_case
    time, state = testsolver.initialise()
    time_scipy = scipysolver.t
    state_scipy = scipysolver.y
    np.testing.assert_allclose(time, time_scipy, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(state.mean[0], state_scipy[0], atol=1e-13, rtol=1e-13)


def test_step_execution(solver_case):
    testsolver, scipysolver = solver_case
    scipysolver.step()
    # perform step of the same size
    testsolver.step(
        scipysolver.t_old,
        scipysolver.t,
        randvars.Constant(scipysolver.y_old),
    )
    np.testing.assert_allclose(scipysolver.y, testsolver.solver.y)


def test_step_variables(solver_case, y, start_point, stop_point):
    testsolver, scipysolver = solver_case
    solver_y_new, solver_error_estimation = testsolver.step(
        start_point, stop_point, randvars.Constant(y)
    )
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
        solver_error_estimation, scipy_error_estimation, atol=1e-14, rtol=1e-14
    )
    # locations are correct
    np.testing.assert_allclose(
        testsolver.solver.t_old, start_point, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(testsolver.solver.t, stop_point, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        testsolver.solver.h_previous,
        stop_point - start_point,
        atol=1e-14,
        rtol=1e-14,
    )
    # evaluations are correct
    np.testing.assert_allclose(testsolver.solver.y_old.mean, y, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(testsolver.solver.y, y_new, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        testsolver.solver.h_abs, stop_point - start_point, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(testsolver.solver.f, f_new, atol=1e-14, rtol=1e-14)


def test_dense_output(dense_case, y, start_point, stop_point):
    testsolver, scipysolver = dense_case
    scipysolver.step()
    # perform step of the same size
    testsolver.step(
        scipysolver.t_old,
        scipysolver.t,
        randvars.Constant(scipysolver.y_old),
    )
    # step has to be performed before dense-output can be computed
    testsolver_dense = testsolver.dense_output()
    scipy_dense = scipysolver._dense_output_impl()
    np.testing.assert_allclose(
        testsolver_dense(scipysolver.t_old),
        scipy_dense(scipysolver.t_old),
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        testsolver_dense(scipysolver.t),
        scipy_dense(scipysolver.t),
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        testsolver_dense((scipysolver.t_old + scipysolver.t) / 2),
        scipy_dense((scipysolver.t_old + scipysolver.t) / 2),
        atol=1e-14,
        rtol=1e-14,
    )


def test_rvlist_to_odesol(times, dense_output, list_of_randvars):
    scipy_sol = OdeSolution(times, dense_output)
    probnum_solution = wrappedscipyodesolution.WrappedScipyODESolution(
        scipy_sol, list_of_randvars
    )
    assert issubclass(
        wrappedscipyodesolution.WrappedScipyODESolution, odesolution.ODESolution
    )
    assert isinstance(probnum_solution, wrappedscipyodesolution.WrappedScipyODESolution)
