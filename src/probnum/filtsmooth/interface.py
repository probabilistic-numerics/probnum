"""
Convenience functions for state space models.
"""
import numpy as np


def generate_cd(dynmod, measmod, initdist, times, _nsteps=5):
    """
    Samples from continuous-discrete state space model.

    Generates true states, observations at "times".

    "cd" = continuous-discrete.

    Example
    """
    states = np.zeros((len(times), dynmod.ndim))
    obs = np.zeros((len(times) - 1, measmod.ndim))
    states[0] = initdist.sample(1)[:, 0]
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        step = (stop - start) / _nsteps
        states[idx] = dynmod.sample(start, stop, step, states[idx - 1])
        obs[idx - 1] = measmod.sample(stop, states[idx])
    return states, obs


def generate_dd(dynmod, measmod, initdist, times):
    """
    Samples from discrete-discrete state space model.

    Generates true states, observations at "times".

    "dd" = discrete-discrete.

    Example
    """
    states = np.zeros((len(times), dynmod.ndim))
    obs = np.zeros((len(times) - 1, measmod.ndim))
    states[0] = initdist.sample(1)[:, 0]
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        states[idx] = dynmod.sample(start, states[idx - 1])
        obs[idx - 1] = measmod.sample(stop, states[idx])

    return states, obs
