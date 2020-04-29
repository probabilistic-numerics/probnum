"""
Utility functions for state space models.

- generate_*(): generate samples according to continuous-discrete (_cd)
or discrete-discrete (_dd) models
"""

from abc import ABC, abstractmethod
import numpy as np

from probnum.prob import RandomVariable, Distribution

__all__ = ["ConditionalDistribution", "generate_cd", "generate_dd"]


class ConditionalDistribution(ABC):
    """
    """
    def conditional(self, rv_or_val, start, stop, **kwargs):
        """
        Returns conditional probability distribution.

        Condition a distribution either on a value or on another
        distribution.
        """
        if isinstance(rv_or_val, np.ndarray) or np.isscalar(rv_or_val):
            return self.conditional_value(value=rv_or_val, start=start,
                                          stop=stop, **kwargs)
        elif isinstance(rv_or_val, RandomVariable):
            return self.conditional_randvar(randvar=rv_or_val, start=start,
                                            stop=stop, **kwargs)
        else:
            errormsg = ("Conditional distribution not implemented for "
                        + "input %s" % str(rv_or_val))
            raise NotImplementedError(errormsg)

    @abstractmethod
    def conditional_value(self, value, start, stop, **kwargs):
        """ """
        raise NotImplementedError

    @abstractmethod
    def conditional_randvar(self, randvar, start, stop, **kwargs):
        """ """
        raise NotImplementedError


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
    initrv : prob.RandomVariable object
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
    initrv : prob.RandomVariable object
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
