import numpy as np
import pytest
import pytest_cases
from pn_ode_benchmarks import scipy_solver
from scipy.integrate._ivp import base
from scipy.integrate._ivp.common import OdeSolution

from probnum import diffeq, randvars
from probnum.diffeq import odesolution
from probnum.diffeq.perturbedsolvers import perturbedstepsolution, perturbedstepsolver


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
def list_of_randvars():
    return list([randvars.Constant(1)])


def test_initialise(solvers):
    testsolver, perturbedsolver = solvers
    time, state = perturbedsolver.initialise()
    time_scipy = testsolver.solver.t
    state_scipy = testsolver.solver.y
    np.testing.assert_allclose(time, time_scipy, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(state.mean[0], state_scipy[0], atol=1e-14, rtol=1e-14)


"""
def test_step(solvers, start_point, stop_point, y):

    # Convergence of the perturbation functions is tested in the corresponding test file.
    # Here it is checked that the perturbed steps are close to the original steps.
    testsolver, perturbedsolver = solvers
    noisy_step = perturbedsolver.perturb_step(
        stop_point - start_point, random_state=123
    )
    y_new, y_error_estimation = testsolver.step(
        start_point, start_point + noisy_step, y
    )
    perturbed_y_new, perturbed_error_estimation = perturbedsolver.step(
        start_point, stop_point, y
    )
    np.testing.assert_allclose(perturbed_y_new.mean, y_new.mean, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        perturbed_error_estimation, y_error_estimation, atol=1e-14, rtol=1e-14
    )
"""


def test_step(solvers, start_point, stop_point, y):

    # When performing two small similar steps, their output should be similar
    testsolver, perturbedsolver = solvers

    # The first step is deterministic.
    first_step, first_error = perturbedsolver.step(start_point, stop_point, y)
    perturbed_y_1, perturbed_error_estimation_1 = perturbedsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    # Reset noise_scales and dense_output.
    perturbedsolver.initialise()
    first_step, first_error = perturbedsolver.step(start_point, stop_point, y)
    perturbed_y_2, perturbed_error_estimation_2 = perturbedsolver.step(
        stop_point, stop_point + start_point, y + first_step
    )
    np.testing.assert_allclose(
        perturbed_y_1.mean, perturbed_y_2.mean, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        perturbed_error_estimation_1,
        perturbed_error_estimation_2,
        atol=1e-14,
        rtol=1e-14,
    )
    # np.testing.assert_(perturbed_y_1.mean.any() != perturbed_y_2.mean.any())


# seed fixen, determinismus
# solve aufrufen


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
