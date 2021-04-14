import numpy as np

from probnum.diffeq.perturbedsolvers import _perturbation_functions

random_state = np.random.mtrand.RandomState(seed=1234)


def test_perturb_uniform():
    num_samples = 100
    all_steps = 0
    for _ in range(num_samples):
        stp = _perturbation_functions.perturb_uniform(0.2, 4, 1)
        all_steps += stp
    mean_suggested_step = all_steps / num_samples
    np.testing.assert_allclose(0.2, mean_suggested_step, atol=1e-4, rtol=1e-4)


def test_perturb_lognormal():
    num_samples = 100
    all_steps = 0
    for _ in range(num_samples):
        stp = _perturbation_functions.perturb_lognormal(0.2, 4, 1)
        all_steps += stp
    mean_suggested_step = all_steps / num_samples
    np.testing.assert_allclose(mean_suggested_step, 0.2, atol=1e-4, rtol=1e-4)
