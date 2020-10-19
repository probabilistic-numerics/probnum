"""
Utility functions for state space models.

- generate_*(): generate samples according to continuous-discrete (_cd)
or discrete-discrete (_dd) models
"""

import numpy as np


def generate_cd(dynmod, measmod, initrv, times, _nsteps=5):
    """
    Samples true states and observations at pre-determined
    timesteps "times" for a continuous-discrete model.
    Parameters
    ----------
    dynmod : continuous.ContinuousModel instance
        Continuous dynamic model.
    measmod : discrete.DiscreteModel instance
        Discrete measurement model.
    initrv : probnum.RandomVariable object
        Random variable according to initial distribution
    times : np.ndarray, shape (n,)
        Timesteps on which the states are to be sampled.
    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times)-1, measmod.dimension)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), dynmod.dimension))
    obs = np.zeros((len(times) - 1, measmod.dimension))
    states[0] = initrv.sample()
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        step = (stop - start) / _nsteps
        next_state_rv, _ = dynmod.transition_realization(
            real=states[idx - 1], start=start, stop=stop, step=step
        )
        states[idx] = next_state_rv.sample()
        next_obs_rv, _ = measmod.transition_realization(real=states[idx], start=stop)
        obs[idx - 1] = next_obs_rv.sample()
    return states, obs


def generate_dd(dynmod, measmod, initrv, times):
    """
    Samples true states and observations at pre-determined
    timesteps "times" for a continuous-discrete model.
    Parameters
    ----------
    dynmod : discrete.DiscreteModel instance
        Discrete dynamic model.
    measmod : discrete.DiscreteModel instance
        Discrete measurement model.
    initrv : probnum.RandomVariable object
        Random variable according to initial distribution
    times : np.ndarray, shape (n,)
        Timesteps on which the states are to be sampled.
    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times)-1, measmod.dimension)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), dynmod.dimension))
    obs = np.zeros((len(times) - 1, len(measmod.diffusionmatrix(0.))))
    states[0] = initrv.sample()
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        next_state_rv, _ = dynmod.transition_realization(
            real=states[idx - 1], start=start, stop=stop
        )
        states[idx] = next_state_rv.sample()
        next_obs_rv, _ = measmod.transition_realization(real=states[idx], start=stop)
        obs[idx - 1] = next_obs_rv.sample()
    return states, obs
