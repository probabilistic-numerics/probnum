"""Convenience function(s) for state space models."""

import numpy as np
import scipy.stats

from probnum import randvars

from .transition import Transition


def generate_samples(
    rng: np.random.Generator,
    dynmod: Transition,
    measmod: Transition,
    initrv: randvars.RandomVariable,
    times: np.ndarray,
):
    """Samples true states and observations at pre-determined timesteps "times" for a
    state space model.

    Parameters
    ----------
    rng
        Random number generator.
    dynmod
        Transition model describing the prior dynamics.
    measmod
        Transition model describing the measurement model.
    initrv
        Random variable according to initial distribution
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

    base_measure_realizations_latent_state = scipy.stats.norm.rvs(
        size=(times.shape + (measmod.input_dim,)), random_state=rng
    )
    latent_states = np.array(
        dynmod.jointly_transform_base_measure_realization_list_forward(
            base_measure_realizations=base_measure_realizations_latent_state,
            t=times,
            initrv=initrv,
            _diffusion_list=np.ones_like(times[:-1]),
        )
    )

    for idx, (state, t) in enumerate(zip(latent_states, times)):
        measured_rv, _ = measmod.forward_realization(state, t=t)
        obs[idx] = measured_rv.sample(rng=rng)
    return latent_states, obs
