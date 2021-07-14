import numpy as np
import pytest

from probnum import diffeq


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1)


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
        diffeq.perturbed.step.perturb_uniform,
        diffeq.perturbed.step.perturb_lognormal,
    ],
)
def test_mean(perturb_fct, step, solver_order, noise_scale, num_samples, rng):
    suggested_steps = perturb_fct(
        rng, step, solver_order, noise_scale, size=num_samples
    )
    mean_suggested_step = np.sum(suggested_steps) / num_samples
    np.testing.assert_allclose(mean_suggested_step, step, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "perturb_fct",
    [
        diffeq.perturbed.step.perturb_uniform,
        diffeq.perturbed.step.perturb_lognormal,
    ],
)
def test_var(perturb_fct, step, solver_order, noise_scale, num_samples, rng):
    expected_var = step ** (2 * solver_order + 1)
    suggested_steps = perturb_fct(
        rng, step, solver_order, noise_scale, size=num_samples
    )
    var = ((suggested_steps - step) ** 2) / num_samples
    np.testing.assert_allclose(expected_var, var, atol=1e-4, rtol=1e-4)
