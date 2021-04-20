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


@pytest.mark.parametrize(
    "perturb_fct",
    [
        _perturbation_functions.perturb_uniform,
        _perturbation_functions.perturb_lognormal,
    ],
)
def test_mean(perturb_fct, step, solver_order, noise_scale, num_samples):
    suggested_steps = perturb_fct(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    mean_suggested_step = np.sum(suggested_steps) / num_samples
    np.testing.assert_allclose(mean_suggested_step, step, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "perturb_fct",
    [
        _perturbation_functions.perturb_uniform,
        _perturbation_functions.perturb_lognormal,
    ],
)
def test_var(perturb_fct, step, solver_order, noise_scale, num_samples):
    expected_var = step ** (2 * solver_order + 1)
    suggested_steps = perturb_fct(
        step, solver_order, noise_scale, random_state=1, size=num_samples
    )
    var = ((suggested_steps - step) ** 2) / num_samples
    np.testing.assert_allclose(expected_var, var, atol=1e-4, rtol=1e-4)
