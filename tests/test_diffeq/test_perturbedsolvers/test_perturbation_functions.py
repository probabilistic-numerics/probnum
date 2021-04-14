import numpy as np

from probnum.diffeq.perturbedsolvers import _perturbation_functions

random_state = np.random.mtrand.RandomState(seed=1234)


def test_perturb_uniform():
    num_samples = 100
    all_steps = 0
    unperturbed_step = 0.2
    for _ in range(num_samples):
        stp = _perturbation_functions.perturb_uniform(unperturbed_step, 4, 1)
        all_steps += stp
    mean_suggested_step = all_steps / num_samples
    np.testing.assert_allclose(
        mean_suggested_step, unperturbed_step, atol=1e-4, rtol=1e-4
    )


def test_perturb_lognormal():
    num_samples = 100
    all_steps = 0
    unperturbed_step = 0.2
    for _ in range(num_samples):
        stp = _perturbation_functions.perturb_lognormal(unperturbed_step, 4, 1)
        all_steps += stp
    mean_suggested_step = all_steps / num_samples
    np.testing.assert_allclose(
        mean_suggested_step, unperturbed_step, atol=1e-4, rtol=1e-4
    )
