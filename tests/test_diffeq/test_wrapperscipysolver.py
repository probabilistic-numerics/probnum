import pathlib

import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import base, rk
from scipy.integrate._ivp.common import OdeSolution

from probnum import diffeq, randvars
from probnum.diffeq import odesolution, wrapperscipyodesolution, wrapperscipysolver

case_modules = [
    ".test_wrapperscipysolver_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_wrapperscipysolver_cases").glob(
        "*_cases.py"
    )
]


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
def lst():
    return list([randvars.Constant(1)])


@pytest_cases.parametrize_with_cases("testsolver,scipysolver", cases=case_modules)
def test_initialise(testsolver, scipysolver):
    time, state = testsolver.initialise()
    time_scipy = scipysolver.t
    state_scipy = scipysolver.y
    np.testing.assert_allclose(time, time_scipy, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(state.mean[0], state_scipy[0], atol=1e-14, rtol=1e-14)


@pytest_cases.parametrize_with_cases("testsolver,scipysolver", cases=case_modules)
def test_step_execution(testsolver, scipysolver, start_point, stop_point, y):
    scipy_y_new, f_new = rk.rk_step(
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
    scipy_error_estimation = scipysolver._estimate_error(
        scipysolver.K, stop_point - start_point
    )
    solver_y_new, solver_error_estimation = testsolver.step(
        start_point, stop_point, randvars.Constant(0.1)
    )
    np.testing.assert_allclose(solver_y_new.mean, scipy_y_new, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        solver_error_estimation, scipy_error_estimation, atol=1e-14, rtol=1e-14
    )


@pytest_cases.parametrize_with_cases("testsolver,scipysolver", cases=case_modules)
def test_step_variables(testsolver, scipysolver, y, start_point, stop_point):
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


@pytest_cases.parametrize_with_cases("testsolver,scipysolver", cases=case_modules)
def test_dense_output(testsolver, scipysolver, y, start_point, stop_point):
    # step has to be performed before dense-output can be computed
    scipysolver.step()
    # perform step of the same size
    testsolver.step(
        scipysolver.t_old,
        scipysolver.t,
        randvars.Constant(scipysolver.y_old),
    )
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


def test_rvlist_to_odesol(times, dense_output, lst):
    scipy_sol = OdeSolution(times, dense_output)
    probnum_solution = wrapperscipyodesolution.WrapperScipyODESolution(
        scipy_sol, times, lst
    )
    assert issubclass(
        wrapperscipyodesolution.WrapperScipyODESolution, odesolution.ODESolution
    )
    assert isinstance(probnum_solution, wrapperscipyodesolution.WrapperScipyODESolution)
