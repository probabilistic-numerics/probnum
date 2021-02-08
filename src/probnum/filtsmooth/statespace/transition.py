"""Markov transition rules: continuous and discrete."""

import abc
from typing import Dict, Optional

import numpy as np

from probnum.random_variables import RandomVariable

__all__ = ["Transition", "generate"]


class Transition(abc.ABC):
    """Markov transition kernel implementations in discrete and continuous time.

    In continuous time, this is a Markov process and described by the solution of a
    stochastic differential equation (SDE)

    .. math:: d x_t = f(t, x_t) d t + d w_t

    driven by a Wiener process :math:`w`. In discrete time, it is a Markov chain and
    described by a transformation

    .. math:: x_{t + \\Delta t}  | x_t \\sim p(x_{t + \\Delta t}  | x_t).

    Sometimes, these can be equivalent. For example, linear, time-invariant SDEs
    have a mild solution that can be written as a discrete transition.


    See Also
    --------
    :class:`ContinuousModel`
        Continuously indexed transitions (SDEs)
    :class:`DiscreteModel`
        Discretely indexed transitions (transformations)
    """

    def __init__(self, input_dim=None, output_dim=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def forward_rv(
        self, rv, t, dt=None, _compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        pass

    @abc.abstractmethod
    def forward_realization(
        self, real, t, dt=None, _compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        pass

    @abc.abstractmethod
    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        pass

    @abc.abstractmethod
    def backward_realization(
        self,
        real_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        pass

    # Utility functions that are used surprisingly often

    def _backward_realization_as_rv(self, real, *args, **kwargs):
        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)
        return self.backward_rv(self, real_as_rv, *args, **kwargs)

    def _forward_realization_as_rv(self, real, *args, **kwargs):
        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)
        return self.forward_rv(self, real_as_rv, *args, **kwargs)

    #
    # def forward_realization(
    #     self,
    #     real: np.ndarray,
    #     start: float,
    #     stop: Optional[float] = None,
    #     step: Optional[float] = None,
    #     _diffusion: Optional[float] = 1.0,
    #     _linearise_at: Optional[RandomVariable] = None,
    # ) -> (RandomVariable, Dict):
    #     """Transition a realization of a random variable from time :math:`t` to time
    #     :math:`t+\\Delta t`.
    #
    #     For random variable :math:`x_t`, it returns the random variable defined by
    #
    #     .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t = r) .
    #
    #     This is different to :meth:`forward_rv` which computes the parametrization
    #     of :math:`x_{t + \\Delta t}` based on the parametrization of :math:`x_t`.
    #
    #     Nb: Think of transition as a verb, i.e. this method "transitions" a realization of a random variable.
    #
    #     Parameters
    #     ----------
    #     real :
    #         Realization of the random variable.
    #     start :
    #         Starting point :math:`t`.
    #     stop :
    #         End point :math:`t + \\Delta t`.
    #     step :
    #         Intermediate step-size. Optional, default is None.
    #     _diffusion :
    #         Optional diffusion parameter for this transition. This field is usually used
    #         to update diffusions with calibrated versions thereof.
    #     _linearise_at :
    #         For approximate transitions , for instance ContinuousEKFComponent,
    #         this argument overloads the state at which the Jacobian is computed.
    #
    #     Returns
    #     -------
    #     RandomVariable
    #         Random variable, describing the state at time :math:`t + \\Delta t`
    #         based on realization at time :math:`t`.
    #     dict
    #         Additional information in form of a dictionary,
    #         for instance the cross-covariance in the
    #         prediction step, access to which is useful in smoothing.
    #
    #     See Also
    #     --------
    #     :meth:`forward_rv`
    #         Apply transition to a random variable.
    #     """
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def forward_rv(
    #     self,
    #     rv: "RandomVariable",
    #     start: float,
    #     stop: Optional[float] = None,
    #     step: Optional[float] = None,
    #     with_gain: Optional[bool] = False,
    #     _diffusion: Optional[float] = 1.0,
    #     _linearise_at: Optional[RandomVariable] = None,
    # ) -> (RandomVariable, Dict):
    #     """Transition a random variable from time :math:`t` to time
    #     :math:`t+\\Delta t`.
    #
    #     For random variable :math:`x_t`, it returns the random variable defined by
    #
    #     .. math:: x_{t + \\Delta t} \\sim p(x_{t + \\Delta t}  | x_t) .
    #
    #     This returns a random variable where the parametrization depends on the paramtrization of :math:`x_t`.
    #     This is different to :meth:`forward_rv` which computes the parametrization
    #     of :math:`x_{t + \\Delta t}` based on a realization of :math:`x_t`.
    #
    #     Nb: Think of transition as a verb, i.e. this method "transitions" a random variable.
    #
    #
    #     Parameters
    #     ----------
    #     rv :
    #         Realization of the random variable.
    #     start :
    #         Starting point :math:`t`.
    #     stop :
    #         End point :math:`t + \\Delta t`.
    #     step :
    #         Intermediate step-size. Optional, default is None.
    #     _diffusion :
    #         Optional diffusion parameter for this transition. This field is usually used
    #         to update diffusions with calibrated versions thereof.
    #     _linearise_at :
    #         For approximate transitions , for instance ContinuousEKFComponent,
    #         this argument overloads the state at which the Jacobian is computed.
    #
    #     Returns
    #     -------
    #     RandomVariable
    #         Random variable, describing the state at time :math:`t + \\Delta t`
    #         based on realization at time :math:`t`.
    #     dict
    #         Additional information in form of a dictionary,
    #         for instance the cross-covariance in the
    #         prediction step, access to which is useful in smoothing.
    #
    #     See Also
    #     --------
    #     :meth:`forward_realization`
    #         Apply transition to a realization of a random variable.
    #     """
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def backward_realization(
    #     self,
    #     realization: np.ndarray,
    #     rv_futu: "RandomVariable",
    #     rv_past: "RandomVariable",
    #     start: float,
    #     stop: Optional[float] = None,
    #     step: Optional[float] = None,
    #     _diffusion: Optional[float] = 1.0,
    #     _linearise_at: Optional[RandomVariable] = None,
    # ):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def backward_rv(
    #     self,
    #     rv_attained: "RandomVariable",
    #     rv_futu: "RandomVariable",
    #     rv_past: "RandomVariable",
    #     start: float,
    #     stop: Optional[float] = None,
    #     step: Optional[float] = None,
    #     _diffusion: Optional[float] = 1.0,
    #     _linearise_at: Optional[RandomVariable] = None,
    # ):
    #     raise NotImplementedError
    #
    # @property
    # def dimension(self) -> int:
    #     """Dimension of the transition model.
    #
    #     Not all transition models have a unique dimension. Some turn a
    #     state (x, y) into a scalar z and it is not clear whether the
    #     dimension should be 2 or 1.
    #     """
    #     raise NotImplementedError


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
        Irrelevant for time-invariant or discrete models.

    Returns
    -------
    states : np.ndarray; shape (len(times), dynmod.dimension)
        True states according to dynamic model.
    obs : np.ndarray; shape (len(times), measmod.dimension)
        Observations according to measurement model.
    """
    states = np.zeros((len(times), measmod.input_dim))
    obs = np.zeros((len(times), measmod.output_dim))

    # initial observation point
    states[0] = initrv.sample()
    next_obs_rv, _ = measmod.forward_realization(real=states[0], t=times[0])
    obs[0] = next_obs_rv.sample()

    # all future points
    for idx in range(1, len(times)):
        t, dt = times[idx - 1], times[idx] - times[idx - 1]
        next_state_rv, _ = dynmod.forward_realization(real=states[idx - 1], t=t, dt=dt)
        states[idx] = next_state_rv.sample()
        next_obs_rv, _ = measmod.forward_realization(real=states[idx], t=t)
        obs[idx] = next_obs_rv.sample()
    return states, obs


def _read_dimension(transition, initrv):
    """Extracts dimension of a transition without calling .dimension(), which is not
    implemented everywhere."""
    # relies on evaluating at zero, which is a dangerous endeavour and therefore,
    # this method is not used in Transition.dimension
    transitioned, _ = transition.forward_realization(
        real=initrv.mean, start=0.0, stop=1.0, step=1.0
    )
    return len(transitioned.sample())
