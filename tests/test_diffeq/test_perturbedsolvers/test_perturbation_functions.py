import numpy as np

from probnum.diffeq.perturbedsolvers import _perturbation_functions

random_state = np.random.mtrand.RandomState(seed=1)


def test_perturb_uniform():
    random_state = np.random.mtrand.RandomState(seed=1)
    step = 0.2
    noise_scale = 1
    order = 4
    noisy_step = random_state.uniform(
        step - noise_scale * step ** (order + 0.5),
        step + noise_scale * step ** (order + 0.5),
    )
    proposed_step = _perturbation_functions.perturb_uniform(step, order, noise_scale)
    np.testing.assert_allclose(noisy_step, proposed_step, atol=1e-14, rtol=1e-14)


def test_perturb_lognormal():
    random_state = np.random.mtrand.RandomState(seed=1)
    step = 0.2
    noise_scale = 1
    order = 4
    mean = np.log(step) - np.log(np.sqrt(1 + noise_scale * (step ** (2 * order))))
    cov = np.log(1 + noise_scale * (step ** (2 * order)))
    noisy_step = np.exp(random_state.normal(mean, cov))
    proposed_step = _perturbation_functions.perturb_lognormal(step, order, noise_scale)
    np.testing.assert_allclose(noisy_step, proposed_step, atol=1e-14, rtol=1e-14)


def test_perturb_uniform_avrg():
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


def test_perturb_lognormal_avrg():
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
