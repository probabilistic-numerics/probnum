"""Convenience function(s) for state space models."""

import numpy as np

from .transition import Transition


def generate_samples(
    rng: np.random.Generator,
    process,
    measmod: Transition,
    times: np.ndarray,
):
    """Samples true states and observations at pre-determined timesteps "times" for a
    state space model.

    Parameters
    ----------
    rng
        Random number generator.
    process
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

    latent_states = process.sample(rng, args=times)

    for idx, (state, t) in enumerate(zip(latent_states, times)):
        measured_rv, _ = measmod.forward_realization(state, t=t)
        obs[idx] = measured_rv.sample(rng=rng)
    return latent_states, obs
