"""Markov transition rules: continuous and discrete."""

import abc
from typing import Dict, Union

import numpy as np

from probnum.random_variables import RandomVariable

__all__ = ["Transition", "generate"]


class Transition(abc.ABC):
    """Markov transition rules in discrete or continuous time.

    In continuous time, this is a Markov process and described by a
    stochastic differential equation (SDE)

    .. math:: d x_t = f(t, x_t) d t + d w_t

    driven by a Wiener process :math:`w`. In discrete time, it is defined by
    a transformation

    .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t).

    Sometimes, these can be equivalent. For example: mild solutions to
    linear, time-invariant SDEs have an equivalent, discretised form that can
    be written as a transformation.

    See Also
    --------
    :class:`ContinuousModel`
        Continuously indexed transitions (SDEs)
    :class:`DiscreteModel`
        Discretely indexed transitions (transformations)
    """

    def __call__(
        self,
        arr_or_rv: Union[np.ndarray, "RandomVariable"],
        start: float = None,
        stop: float = None,
        **kwargs
    ) -> ("RandomVariable", Dict):
        """Transition a random variable or a realization of one.

        The input is either interpreted as a random variable or as a
        realization. Accordingly, the respective methods are called:
        :meth:`transition_realization` or :meth:`transition_rv`.
        """
        if isinstance(arr_or_rv, RandomVariable):
            return self.transition_rv(rv=arr_or_rv, start=start, stop=stop, **kwargs)
        return self.transition_realization(
            real=arr_or_rv, start=start, stop=stop, **kwargs
        )

    @abc.abstractmethod
    def transition_realization(
        self, real: np.ndarray, start: float, stop: float = None, **kwargs
    ) -> ("RandomVariable", Dict):
        """Transition a realization of a random variable from time :math:`t` to time
        :math:`t+\\Delta t`.

        For random variable :math:`x_t`, it returns the random variable defined by

        .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t = r) .

        This is different to :meth:`transition_rv` which computes the parametrization
        of :math:`x_{t + \\Delta t}` based on the parametrization of :math:`x_t`.

        Nb: Think of transition as a verb, i.e. this method "transitions" a realization of a random variable.

        Parameters
        ----------
        real :
            Realization of the random variable.
        start :
            Starting point :math:`t`.
        stop :
            End point :math:`t + \\Delta t`.

        Returns
        -------
        RandomVariable
            Random variable, describing the state at time :math:`t + \\Delta t`
            based on realization at time :math:`t`.
        dict
            Additional information in form of a dictionary,
            for instance the cross-covariance in the
            prediction step, access to which is useful in smoothing.

        See Also
        --------
        :meth:`transition_rv`
            Apply transition to a random variable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rv(
        self, rv: "RandomVariable", start: float, stop: float = None, **kwargs
    ) -> ("RandomVariable", Dict):
        """Transition a random variable from time :math:`t` to time
        :math:`t+\\Delta t`.

        For random variable :math:`x_t`, it returns the random variable defined by

        .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t) .

        This returns a random variable where the parametrization depends on the paramtrization of :math:`x_t`.
        This is different to :meth:`transition_rv` which computes the parametrization
        of :math:`x_{t + \\Delta t}` based on a realization of :math:`x_t`.

        Nb: Think of transition as a verb, i.e. this method "transitions" a random variable.


        Parameters
        ----------
        rv :
            Realization of the random variable.
        start :
            Starting point :math:`t`.
        stop :
            End point :math:`t + \\Delta t`.

        Returns
        -------
        RandomVariable
            Random variable, describing the state at time :math:`t + \\Delta t`
            based on realization at time :math:`t`.
        dict
            Additional information in form of a dictionary,
            for instance the cross-covariance in the
            prediction step, access to which is useful in smoothing.

        See Also
        --------
        :meth:`transition_realization`
            Apply transition to a realization of a random variable.
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Dimension of the transition model.

        Not all transition models have a unique dimension. Some turn a
        state (x, y) into a scalar z and it is not clear whether the
        dimension should be 2 or 1.
        """
        raise NotImplementedError


def generate(dynmod, measmod, initrv, times, num_steps=5):
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
    num_steps : int
        Number of steps to be taken for numerical integration
        of the continuous prior model. Optional. Default is 5.
        Irrelevant for LTI or discrete models.

    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times)-1, measmod.dimension)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), _read_dimension(dynmod, initrv)))
    obs = np.zeros((len(times) - 1, _read_dimension(measmod, initrv)))
    states[0] = initrv.sample()
    for idx in range(1, len(times)):
        start, stop = times[idx - 1], times[idx]
        step = (stop - start) / num_steps
        next_state_rv, _ = dynmod.transition_realization(
            real=states[idx - 1], start=start, stop=stop, step=step
        )
        states[idx] = next_state_rv.sample()
        next_obs_rv, _ = measmod.transition_realization(real=states[idx], start=stop)
        obs[idx - 1] = next_obs_rv.sample()
    return states, obs


def _read_dimension(transition, initrv):
    """Extracts dimension of a transition without calling .dimension(), which is not
    implemented everywhere."""
    return len(transition.transition_realization(initrv.mean, 0.0, 1.0)[0].sample())
