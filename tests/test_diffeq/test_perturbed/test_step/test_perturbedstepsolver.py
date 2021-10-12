import numpy as np
import pytest
import pytest_cases
from scipy.integrate._ivp import base

from probnum import diffeq, randvars


@pytest_cases.fixture
@pytest_cases.parametrize_with_cases(
    "testsolver, perturbedsolver, ode", cases=".test_perturbed_cases"
)
def solvers(testsolver, perturbedsolver, ode):
    return testsolver, perturbedsolver, ode


@pytest.fixture
def start_point():
    return 0.1


@pytest.fixture
def stop_point():
    return 0.2


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
    return list(randvars.Constant(1))


def test_initialise(solvers):
    testsolver, perturbedsolver, ode = solvers
    state_scipy = testsolver.initialize(ode)
    state_perturbed = perturbedsolver.initialize(ode)

    np.testing.assert_allclose(state_perturbed.t, state_scipy.t, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        state_perturbed.rv.mean[0], state_scipy.rv.mean[0], atol=1e-14, rtol=1e-14
    )


def test_step(solvers, start_point, stop_point, y):
    """When performing two small similar steps, their output should be similar.

    For the first step no error estimation is available, the first step is therefore
    deterministic and to check for non-determinism, two steps have to be performed.
    """

    _, perturbedsolver, ode = solvers
    perturbedsolver.initialize(ode)

    test_state = diffeq.ODESolverState(
        ivp=ode, rv=y, t=start_point, error_estimate=None, reference_state=None
    )
    step_after_first_step = perturbedsolver.attempt_step(
        test_state, dt=stop_point - start_point
    )
    perturbed_y_1 = perturbedsolver.attempt_step(
        step_after_first_step, dt=stop_point - start_point
    )

    perturbedsolver.initialize(ode)
    test_state = diffeq.ODESolverState(
        ivp=ode, rv=y, t=start_point, error_estimate=None, reference_state=None
    )

    step_after_first_step = perturbedsolver.attempt_step(
        test_state, dt=stop_point - start_point
    )
    perturbed_y_2 = perturbedsolver.attempt_step(
        step_after_first_step, dt=stop_point - start_point
    )

    np.testing.assert_allclose(
        perturbed_y_1.rv.mean, perturbed_y_2.rv.mean, atol=1e-4, rtol=1e-4
    )

    np.testing.assert_allclose(
        perturbed_y_1.error_estimate,
        perturbed_y_2.error_estimate,
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.all(np.not_equal(perturbed_y_1.rv.mean, perturbed_y_2.rv.mean))


def test_solve(solvers):
    _, perturbedstepsolver, ode = solvers
    solution = perturbedstepsolver.solve(ode)
    assert isinstance(solution, diffeq.ODESolution)


def test_rvlist_to_odesol(solvers, times, list_of_randvars, dense_output):
    _, perturbedstepsolver, ode = solvers
    perturbedstepsolver.interpolants = dense_output
    perturbedstepsolver.scales = [1]
    probnum_solution = perturbedstepsolver.rvlist_to_odesol(times, list_of_randvars)
    assert isinstance(probnum_solution, diffeq.ODESolution)
    assert isinstance(probnum_solution, diffeq.perturbed.step.PerturbedStepSolution)


def test_postprocess(solvers):
    testsolver, perturbedstepsolver, ode = solvers
    odesol = perturbedstepsolver.solve(ode)
    post_process = perturbedstepsolver.postprocess(odesol)
    assert isinstance(post_process, diffeq.ODESolution)
