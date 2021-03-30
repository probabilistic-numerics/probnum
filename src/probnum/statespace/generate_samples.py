"""Convenience function(s) for state space models."""

import numpy as np
import scipy.stats


def generate_samples(dynmod, measmod, initrv, times, random_state=None):
    """Samples true states and observations at pre-determined timesteps "times" for a
    state space model.

    Parameters
    ----------
    dynmod : statespace.Transition
        Transition model describing the prior dynamics.
    measmod : statespace.Transition
        Transition model describing the measurement model.
    initrv : probnum.RandomVariable object
        Random variable according to initial distribution
    times : np.ndarray, shape (n,)
        Timesteps on which the states are to be sampled.
    random_state :
        Random state that is used to generate the samples from the latent state.
        The measurement samples are not affected by this.

    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times), measmod.dimension)
        Observations according to measurement model.
    """
    obs = np.zeros((len(times), measmod.output_dim))

    base_measure_realizations_latent_state = scipy.stats.norm.rvs(
        size=(times.shape + (measmod.input_dim,)), random_state=random_state
    )
    base_measure_realizations_observation = scipy.stats.norm.rvs(
        size=(times.shape + (measmod.output_dim,)), random_state=random_state
    )

    # One diffusion less than timepoints, because they correspond to intervals.
    squared_diffusions = np.ones_like(times[:-1])
    latent_states = np.array(
        dynmod.jointly_transform_base_measure_realization_list_forward(
            base_measure_realizations=base_measure_realizations_latent_state,
            t=times,
            initrv=initrv,
            _diffusion_list=squared_diffusions,
        )
    )

    for idx, (state, t, base) in enumerate(
        zip(latent_states, times, base_measure_realizations_observation)
    ):
        measured_rv, _ = measmod.forward_realization(state, t=t)
        obs[idx] = measured_rv.mean + measured_rv.cov_cholesky @ base
    return latent_states, obs
