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
    states : np.ndarray; shape (len(times), dynmod.ndim)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times)-1, measmod.ndim)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), dynmod.ndim))
    obs = np.zeros((len(times) - 1, measmod.ndim))
    states[0] = initrv.sample()
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        step = (stop - start) / _nsteps
        states[idx] = dynmod.sample(start, stop, step, states[idx - 1])
        obs[idx - 1] = measmod.sample(stop, states[idx])
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
    states : np.ndarray; shape (len(times), dynmod.ndim)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times)-1, measmod.ndim)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), dynmod.ndim))
    obs = np.zeros((len(times) - 1, measmod.ndim))
    states[0] = initrv.sample()
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        states[idx] = dynmod.sample(start, states[idx - 1])
        obs[idx - 1] = measmod.sample(stop, states[idx])
    return states, obs
