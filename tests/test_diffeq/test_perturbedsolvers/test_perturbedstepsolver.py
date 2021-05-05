import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import base, rk

from probnum import diffeq, randvars
from probnum.diffeq import odesolution, wrappedscipysolver
from probnum.diffeq.perturbedsolvers import (
    _perturbation_functions,
    perturbedstepsolution,
    perturbedstepsolver,
)


@pytest_cases.fixture
@pytest_cases.parametrize_with_cases("solvers", cases="test_perturbed_cases")
# Workaround: usually the input of this would be "testsolver, perturbedsolver" instead of "solvers"
# see issue https://github.com/smarie/python-pytest-cases/issues/202
def solvers(solvers):
    testsolver, perturbedsolver = solvers
    return testsolver, perturbedsolver


@pytest.fixture
def start_point():
    return 0.0


@pytest.fixture
def stop_point():
    return 0.1


@pytest.fixture
def y():
    return randvars.Constant(0.1)


@pytest.fixture
def dense_output():
    return [base.DenseOutput(0, 1)]


@pytest.fixture
def times():
    return [0, 1]


@pytest.fixture
def steprule():
    return diffeq.ConstantSteps(0.1)


@pytest.fixture
def list_of_randvars():
    return list([randvars.Constant(1)])


@pytest.fixture
def deterministicsolver():
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    )
    return perturbedstepsolver.PerturbedStepSolver(
        testsolver,
        noise_scale=1,
        perturb_function=_perturbation_functions.perturb_uniform,
        random_state=123,
    )


def test_initialise(solvers):
    testsolver, perturbedsolver = solvers
    time, state = perturbedsolver.initialise()
    time_scipy = testsolver.solver.t
    state_scipy = testsolver.solver.y
    np.testing.assert_allclose(time, time_scipy, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(state.mean[0], state_scipy[0], atol=1e-14, rtol=1e-14)


def test_step(solvers, start_point, stop_point, y):

    # When performing two small similar steps, their output should be similar.
    # For the first step no error estimation is available, the first step is
    # therefore deterministic and to check for non-determinism, two steps have to be
    # performed.
    testsolver, perturbedsolver = solvers
    first_step, first_error = perturbedsolver.step(start_point, stop_point, y)
    perturbed_y_1, perturbed_error_estimation_1 = perturbedsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    perturbedsolver.initialise()
    first_step, first_error = perturbedsolver.step(start_point, stop_point, y)
    perturbed_y_2, perturbed_error_estimation_2 = perturbedsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    np.testing.assert_allclose(
        perturbed_y_1.mean, perturbed_y_2.mean, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        perturbed_error_estimation_1,
        perturbed_error_estimation_2,
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.all(np.not_equal(perturbed_y_1.mean, perturbed_y_2.mean))


def test_solve(solvers, steprule):
    testsolver, perturbedstepsolver = solvers
    solution = perturbedstepsolver.solve(steprule)
    assert isinstance(solution, diffeq.ODESolution)


def test_step_fixed_seed(deterministicsolver, start_point, stop_point, y):

    # This is the same test as test_step() but with fixed random_state and therefore
    # deterministic output.
    first_step, first_error = deterministicsolver.step(start_point, stop_point, y)
    perturbed_y_1, perturbed_error_estimation_1 = deterministicsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    deterministicsolver.initialise()
    first_step, first_error = deterministicsolver.step(start_point, stop_point, y)
    perturbed_y_2, perturbed_error_estimation_2 = deterministicsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    np.testing.assert_allclose(
        perturbed_y_1.mean, perturbed_y_2.mean, atol=1e-14, rtol=1e-14
    )


def test_method_callback(solvers, start_point, stop_point, y):
    testsolver, perturbedstepsolver = solvers
    perturbedstepsolver.initialise()
    perturbedstepsolver.step(start_point, stop_point, y)
    np.testing.assert_allclose(len(perturbedstepsolver.scales), 0)
    perturbedstepsolver.method_callback(start_point, y, 0)
    np.testing.assert_allclose(len(perturbedstepsolver.scales), 1)


def test_rvlist_to_odesol(solvers, times, list_of_randvars, dense_output):
    testsolver, perturbedstepsolver = solvers
    perturbedstepsolver.interpolants = dense_output
    perturbedstepsolver.scales = [1]
    probnum_solution = perturbedstepsolver.rvlist_to_odesol(times, list_of_randvars)
    assert issubclass(
        perturbedstepsolution.PerturbedStepSolution, odesolution.ODESolution
    )
    assert isinstance(probnum_solution, perturbedstepsolution.PerturbedStepSolution)
