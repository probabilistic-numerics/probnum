"""Convenience function(s) for state space models."""

import numpy as np

from probnum import backend
from probnum.randprocs.markov import _markov_process, _transition


def generate_artificial_measurements(
    rng: np.random.Generator,
    prior_process: _markov_process.MarkovProcess,
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

    seed = backend.random.seed(
        int(rng.bit_generator._seed_seq.generate_state(1, dtype=np.uint64)[0] // 2)
    )
    latent_states_seed, seed = backend.random.split(seed, num=2)
    latent_states = prior_process.sample(seed=latent_states_seed, args=times)

    for idx, (state, t) in enumerate(zip(latent_states, times)):
        measured_rv, _ = measmod.forward_realization(state, t=t)
        sample_seed, seed = backend.random.split(seed, num=2)
        obs[idx] = measured_rv.sample(seed=sample_seed)
    return latent_states, obs
