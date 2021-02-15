"""Markov transition rules: continuous and discrete."""

import abc
from typing import Dict, Optional

import numpy as np

import probnum.random_variables as pnrv
from probnum._randomvariablelist import _RandomVariableList


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

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def forward_realization(
        self, real, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    # Smoothing and sampling implementations

    def smooth_list(self, rv_list, locations, _previous_posterior=None):
        """Apply smoothing to a list of RVs.

        Parameters
        ----------
        rv_list : _RandomVariableList or array_like
            List of random variables to be smoothed.
        locations : array_like
            Locations of the random variables in rv_list.
        _previous_posterior
            For posterior linearisation. Used iterated smoothing.

        Returns
        -------
        _RandomVariableList
            List of smoothed random variables.
        """

        final_rv = rv_list[-1]
        curr_rv = final_rv
        out_rvs = [curr_rv]
        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = rv_list[idx - 1]

            _linearise_smooth_step_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(locations[idx - 1])
            )

            # Actual smoothing step
            curr_rv, _ = self.backward_rv(
                curr_rv,
                unsmoothed_rv,
                t=locations[idx - 1],
                dt=locations[idx] - locations[idx - 1],
                _linearise_at=_linearise_smooth_step_at,
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return _RandomVariableList(out_rvs)

    def sample_list(self, rv_list, locations, _previous_posterior=None):
        """Jointly sample from a transition, conditioned on a list of RVs.

        Parameters
        ----------
        rv_list : _RandomVariableList or array_like
            List of random variables to be smoothed.
        locations : array_like
            Locations of the random variables in rv_list.
        _previous_posterior
            For posterior linearisation. Used iterated smoothing.

        Returns
        -------
        _RandomVariableList
            List of smoothed random variables.
        """

        num_dim = len(rv_list[-1].sample())

        curr_sample = rv_list[-1].sample()
        out_samples = [curr_sample]
        zero_mat = np.zeros((num_dim, num_dim))
        curr_rv = pnrv.Normal(curr_sample, zero_mat, cov_cholesky=zero_mat)

        for idx in reversed(range(1, len(locations))):
            unsmoothed_rv = rv_list[idx - 1]

            _linearise_smooth_step_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(locations[idx - 1])
            )

            # Actual smoothing step
            curr_rv, _ = self.backward_rv(
                curr_rv,
                unsmoothed_rv,
                t=locations[idx - 1],
                dt=locations[idx] - locations[idx - 1],
                _linearise_at=_linearise_smooth_step_at,
            )

            curr_sample = curr_rv.sample()
            out_samples.append(curr_sample)
            curr_rv = pnrv.Normal(curr_sample, zero_mat, cov_cholesky=zero_mat)

        out_samples.reverse()
        return out_samples

    # Utility functions that are used surprisingly often:
    #
    # Call forward/backward transitions of realisations by
    # turning it into a Normal RV with zero covariance and by
    # referring to the forward/backward transition of RVs.

    def _backward_realization_via_backward_rv(self, real, *args, **kwargs):
        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)
        return self.backward_rv(real_as_rv, *args, **kwargs)

    def _forward_realization_via_forward_rv(self, real, *args, **kwargs):
        zero_cov = np.zeros((len(real), len(real)))
        real_as_rv = pnrv.Normal(mean=real, cov=zero_cov, cov_cholesky=zero_cov)
        return self.forward_rv(real_as_rv, *args, **kwargs)

    #
    #  The below is still here, because it contains docstrings that need to be copied soon!
    #
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
