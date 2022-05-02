"""Convenience function(s) for state space models."""

import numpy as np

from probnum import backend
from probnum.randprocs.markov import _markov, _transition


def generate_artificial_measurements(
    rng: np.random.Generator,
    prior_process: _markov.MarkovProcess,
    measmod: _transition.Transition,
    times: np.ndarray,
):
    """Samples true states and observations at pre-determined timesteps "times" for a
    state space model.

    Parameters
    ----------
    rng
        Random number generator.
    prior_process
        Markov process to sample from, defining dynamics and initial conditions.
    measmod
        Transition model describing the measurement model.
    times
        Timesteps on which the states are to be sampled.

    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times), measmod.dimension)
        Observations according to measurement model.
    """
    obs = np.zeros((len(times), measmod.output_dim))

    rng_state = backend.random.rng_state(
        int(rng.bit_generator._seed_seq.generate_state(1, dtype=np.uint64)[0] // 2)
    )
    latent_states_rng_state, rng_state = backend.random.split(rng_state, num=2)
    latent_states = prior_process.sample(rng_state=latent_states_rng_state, args=times)

    for idx, (state, t) in enumerate(zip(latent_states, times)):
        measured_rv, _ = measmod.forward_realization(state, t=t)
        sample_rng_state, rng_state = backend.random.split(rng_state, num=2)
        obs[idx] = measured_rv.sample(seed=sample_rng_state)
    return latent_states, obs
