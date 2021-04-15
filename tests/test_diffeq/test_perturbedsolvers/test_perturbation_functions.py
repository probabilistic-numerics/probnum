import numpy as np
import pytest

from probnum.diffeq.perturbedsolvers import _perturbation_functions

random_state = np.random.mtrand.RandomState(seed=1)


@pytest.fixture
def step():
    return 0.2


@pytest.fixture
def solver_order():
    return 4


@pytest.fixture
def noise_scale():
    return 1


@pytest.fixture
def num_samples():
    return 100


def test_perturb_uniform_mean(step, solver_order, noise_scale, num_samples):
    suggested_steps = _perturbation_functions.perturb_uniform(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    mean_suggested_step = np.sum(suggested_steps) / num_samples
    np.testing.assert_allclose(mean_suggested_step, step, atol=1e-4, rtol=1e-4)


def test_perturb_lognormal_mean(step, solver_order, noise_scale, num_samples):
    suggested_steps = _perturbation_functions.perturb_lognormal(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    mean_suggested_step = np.sum(suggested_steps) / num_samples
    np.testing.assert_allclose(mean_suggested_step, step, atol=1e-6, rtol=1e-6)


def test_perturb_uniform_var(step, solver_order, noise_scale, num_samples):
    expected_var = (1 / 3) * step ** (2 * solver_order + 1)
    suggested_steps = _perturbation_functions.perturb_uniform(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    var = ((suggested_steps - step) ** 2) / num_samples
    np.testing.assert_allclose(expected_var, var, atol=1e-4, rtol=1e-4)


def test_perturb_lognormal_var(step, solver_order, noise_scale, num_samples):
    expected_var = (1 / 3) * step ** (2 * solver_order + 1)
    suggested_steps = _perturbation_functions.perturb_uniform(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    var = ((suggested_steps - step) ** 2) / num_samples
    np.testing.assert_allclose(expected_var, var, atol=1e-6, rtol=1e-6)
