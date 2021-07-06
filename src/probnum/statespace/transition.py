"""Markov transition rules: continuous and discrete."""

import abc

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.typing import FloatArgType, IntArgType


class Transition(abc.ABC):
    r"""Interface for Markov transitions in discrete and continuous time.

    This framework describes transition probabilities

    .. math:: p(\mathcal{G}_t[x(t)] \,|\,  x(t))

    for some operator :math:`\mathcal{G}: \mathbb{R}^d \rightarrow \mathbb{R}^m`, which are used to describe the evolution of Markov processes

    .. math:: p(x(t+\Delta t) \,|\, x(t))

    both in discrete time (Markov chains) and in continuous time (Markov processes).
    In continuous time, Markov processes are modelled as the solution of a
    stochastic differential equation (SDE)

    .. math:: d x(t) = f(t, x(t)) d t + d w(t)

    driven by a Wiener process :math:`w`. In discrete time, Markov chain are
    described by a transformation

    .. math:: x({t + \Delta t})  \,|\, x(t) \sim p(x({t + \Delta t})  \,|\, x(t)).

    Sometimes, these can be equivalent. For example, linear, time-invariant SDEs
    have a mild solution that can be written as a discrete transition.
    In ProbNum, we also use discrete-time transition objects to describe observation models,

    .. math:: z_k \,|\, x(t_k) \sim p(z_k \,|\, x(t_k))

    for some :math:`k=0,...,K`. All three building blocks are used heavily in filtering and smoothing, as well as solving ODEs.

    See Also
    --------
    :class:`SDE`
        Markov-processes in continuous time.
    :class:`DiscreteGaussian`
        Markov-chains and general discrete-time transitions (likelihoods).
    """

    def __init__(self, input_dim: IntArgType, output_dim: IntArgType):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim})"

    @abc.abstractmethod
    def forward_rv(
        self, rv, t, dt=None, compute_gain=False, _diffusion=1.0, _linearise_at=None
    ):
        r"""Forward-pass of a state, according to the transition. In other words, return a description of

        .. math:: p(\mathcal{G}_t[x(t)] \,|\, x(t)),

        or, if we take a message passing perspective,

        .. math:: p(\mathcal{G}_t[x(t)] \,|\, x(t), z_{\leq t}),

        for past observations :math:`z_{\leq t}`. (This perspective will be more interesting in light of :meth:`backward_rv`).


        Parameters
        ----------
        rv
            Random variable that describes the current state.
        t
            Current time point.
        dt
            Increment :math:`\Delta t`. Ignored for discrete-time transitions.
        compute_gain
            Flag that indicates whether the expected gain of the forward transition shall be computed. This is important if the forward-pass
            is computed as part of a forward-backward pass, as it is for instance the case in a Kalman update.
        _diffusion
            Special diffusion of the driving stochastic process, which is used internally.
        _linearise_at
            Specific point of linearisation for approximate forward passes (think: extended Kalman filtering). Used internally for iterated filtering and smoothing.

        Returns
        -------
        RandomVariable
            New state, after applying the forward-pass.
        Dict
            Information about the forward pass. Can for instance contain a `gain` key, if `compute_gain` was set to `True` (and if the transition supports this functionality).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_realization(
        self,
        realization,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        r"""Forward-pass of a realization of a state, according to the transition. In other words, return a description of

        .. math:: p(\mathcal{G}_t[x(t)] \,|\, x(t)=\xi),

        for some realization :math:`\xi`.

        Parameters
        ----------
        realization
            Realization :math:`\xi` of the random variable :math:`x(t)` that describes the current state.
        t
            Current time point.
        dt
            Increment :math:`\Delta t`. Ignored for discrete-time transitions.
        compute_gain
            Flag that indicates whether the expected gain of the forward transition shall be computed. This is important if the forward-pass
            is computed as part of a forward-backward pass, as it is for instance the case in a Kalman update.
        _diffusion
            Special diffusion of the driving stochastic process, which is used internally.
        _linearise_at
            Specific point of linearisation for approximate forward passes (think: extended Kalman filtering). Used internally for iterated filtering and smoothing.

        Returns
        -------
        RandomVariable
            New state, after applying the forward-pass.
        Dict
            Information about the forward pass. Can for instance contain a `gain` key, if `compute_gain` was set to `True` (and if the transition supports this functionality).
        """
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
        r"""Backward-pass of a state, according to the transition. In other words, return a description of

        .. math::
            p(x(t) \,|\, z_{\mathcal{G}_t})
            = \int p(x(t) \,|\, z_{\mathcal{G}_t}, \mathcal{G}_t(x(t)))
            p(\mathcal{G}_t(x(t)) \,|\, z_{\mathcal{G}_t})) d \mathcal{G}_t(x(t)),

        for observations :math:`z_{\mathcal{G}_t}` of :math:`{\mathcal{G}_t}(x(t))`.
        For example, this function is called in a Rauch-Tung-Striebel smoothing step, which computes a Gaussian distribution

        .. math::
            p(x(t) \,|\, z_{\leq t+\Delta t})
            = \int p(x(t) \,|\, z_{\leq t+\Delta t}, x(t+\Delta t))
            p(x(t+\Delta t) \,|\, z_{\leq t+\Delta t})) d x(t+\Delta t),

        from filtering distribution :math:`p(x(t) \,|\, z_{\leq t})` and smoothing distribution :math:`p(x(t+\Delta t) \,|\, z_{\leq t+\Delta t})`,
        where :math:`z_{\leq t + \Delta t}` contains both :math:`z_{\leq t}` and :math:`z_{t + \Delta t}`.

        Parameters
        ----------
        rv_obtained
            "Incoming" distribution (think: :math:`p(x(t+\Delta t) \,|\, z_{\leq t+\Delta t})`) as a RandomVariable.
        rv
            "Current" distribution (think: :math:`p(x(t) \,|\, z_{\leq t})`) as a RandomVariable.
        rv_forwarded
            "Forwarded" distribution (think: :math:`p(x(t+\Delta t) \,|\, z_{\leq t})`) as a RandomVariable. Optional. If provided (in conjunction with `gain`), computation might be more efficient,
            because most backward passes require the solution of a forward pass. If `rv_forwarded` is not provided, :meth:`forward_rv` might be called internally (depending on the object)
            which is skipped if `rv_forwarded` has been provided
        gain
            Expected gain from "observing states at time :math:`t+\Delta t` from time :math:`t`). Optional. If provided (in conjunction with `rv_forwarded`), some additional computations may be avoided (depending on the object).
        t
            Current time point.
        dt
            Increment :math:`\Delta t`. Ignored for discrete-time transitions.
        _diffusion
            Special diffusion of the driving stochastic process, which is used internally.
        _linearise_at
            Specific point of linearisation for approximate forward passes (think: extended Kalman filtering). Used internally for iterated filtering and smoothing.

        Returns
        -------
        RandomVariable
            New state, after applying the backward-pass.
        Dict
            Information about the backward-pass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward_realization(
        self,
        realization_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        r"""Backward-pass of a realisation of a state, according to the transition. In other words, return a description of

        .. math::
            p(x(t) \,|\, {\mathcal{G}_t(x(t)) = \xi})

        for an observed realization :math:`\xi` of  :math:`{\mathcal{G}_t}(x(t))`.
        For example, this function is called in a Kalman update step.

        Parameters
        ----------
        realization_obtained
            Observed realization :math:`\xi` as an array.
        rv
            "Current" distribution :math:`p(x(t))` as a RandomVariable.
        rv_forwarded
            "Forwarded" distribution (think: :math:`p(\mathcal{G}_t(x(t)) \,|\, x(t))`) as a RandomVariable. Optional. If provided (in conjunction with `gain`), computation might be more efficient,
            because most backward passes require the solution of a forward pass. If `rv_forwarded` is not provided, :meth:`forward_rv` might be called internally (depending on the object)
            which is skipped if `rv_forwarded` has been provided
        gain
            Expected gain. Optional. If provided (in conjunction with `rv_forwarded`), some additional computations may be avoided (depending on the object).
        t
            Current time point.
        dt
            Increment :math:`\Delta t`. Ignored for discrete-time transitions.
        _diffusion
            Special diffusion of the driving stochastic process, which is used internally.
        _linearise_at
            Specific point of linearisation for approximate forward passes (think: extended Kalman filtering). Used internally for iterated filtering and smoothing.

        Returns
        -------
        RandomVariable
            New state, after applying the backward-pass.
        Dict
            Information about the backward-pass.
        """
        raise NotImplementedError

    # Smoothing and sampling implementations

    def smooth_list(
        self, rv_list, locations, _diffusion_list, _previous_posterior=None
    ):
        """Apply smoothing to a list of random variables, according to the present
        transition.

        Parameters
        ----------
        rv_list : _randomvariablelist._RandomVariableList
            List of random variables to be smoothed.
        locations :
            Locations :math:`t` of the random variables in the time-domain. Used for continuous-time transitions.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Specify a previous posterior to improve linearisation in approximate backward passes.
            Used in iterated smoothing based on posterior linearisation.

        Returns
        -------
        _randomvariablelist._RandomVariableList
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
            squared_diffusion = _diffusion_list[idx - 1]

            # Actual smoothing step
            curr_rv, _ = self.backward_rv(
                curr_rv,
                unsmoothed_rv,
                t=locations[idx - 1],
                dt=locations[idx] - locations[idx - 1],
                _diffusion=squared_diffusion,
                _linearise_at=_linearise_smooth_step_at,
            )
            out_rvs.append(curr_rv)
        out_rvs.reverse()
        return _randomvariablelist._RandomVariableList(out_rvs)

    def jointly_transform_base_measure_realization_list_backward(
        self,
        base_measure_realizations: np.ndarray,
        t: FloatArgType,
        rv_list: _randomvariablelist._RandomVariableList,
        _diffusion_list: np.ndarray,
        _previous_posterior=None,
    ) -> np.ndarray:
        """Transform samples from a base measure into joint backward samples from a list
        of random variables.

        Parameters
        ----------
        base_measure_realizations :
            Base measure realizations (usually samples from a standard Normal distribution).
            These are transformed into joint realizations of the random variable list.
        rv_list :
            List of random variables to be jointly sampled from.
        t :
            Locations of the random variables in the list. Assumed to be sorted.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Previous posterior. Used for iterative posterior linearisation.

        Returns
        -------
        np.ndarray
            Jointly transformed realizations.
        """
        curr_rv = rv_list[-1]

        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            -1
        ].reshape((-1,))
        out_samples = [curr_sample]

        for idx in reversed(range(1, len(t))):
            unsmoothed_rv = rv_list[idx - 1]
            _linearise_smooth_step_at = (
                None if _previous_posterior is None else _previous_posterior(t[idx - 1])
            )

            # Condition on the 'future' realization and sample
            squared_diffusion = _diffusion_list[idx - 1]
            dt = t[idx] - t[idx - 1]
            curr_rv, _ = self.backward_realization(
                curr_sample,
                unsmoothed_rv,
                t=t[idx - 1],
                dt=dt,
                _linearise_at=_linearise_smooth_step_at,
                _diffusion=squared_diffusion,
            )
            curr_sample = (
                curr_rv.mean
                + curr_rv.cov_cholesky
                @ base_measure_realizations[idx - 1].reshape(
                    -1,
                )
            )
            out_samples.append(curr_sample)

        out_samples.reverse()
        return out_samples

    def jointly_transform_base_measure_realization_list_forward(
        self,
        base_measure_realizations: np.ndarray,
        t: FloatArgType,
        initrv: randvars.RandomVariable,
        _diffusion_list: np.ndarray,
        _previous_posterior=None,
    ) -> np.ndarray:
        """Transform samples from a base measure into joint backward samples from a list
        of random variables.

        Parameters
        ----------
        base_measure_realizations :
            Base measure realizations (usually samples from a standard Normal distribution).
            These are transformed into joint realizations of the random variable list.
        initrv :
            Initial random variable.
        t :
            Locations of the random variables in the list. Assumed to be sorted.
        _diffusion_list :
            List of diffusions that correspond to the intervals in the locations.
            If `locations=(t0, ..., tN)`, then `_diffusion_list=(d1, ..., dN)`, i.e. it contains one element less.
        _previous_posterior :
            Previous posterior. Used for iterative posterior linearisation.

        Returns
        -------
        np.ndarray
            Jointly transformed realizations.
        """
        curr_rv = initrv

        curr_sample = curr_rv.mean + curr_rv.cov_cholesky @ base_measure_realizations[
            0
        ].reshape((-1,))
        out_samples = [curr_sample]

        for idx in range(1, len(t)):

            _linearise_prediction_step_at = (
                None if _previous_posterior is None else _previous_posterior(t[idx - 1])
            )

            squared_diffusion = _diffusion_list[idx - 1]
            dt = t[idx] - t[idx - 1]
            curr_rv, _ = self.forward_realization(
                curr_sample,
                t=t[idx - 1],
                dt=dt,
                _linearise_at=_linearise_prediction_step_at,
                _diffusion=squared_diffusion,
            )
            curr_sample = (
                curr_rv.mean
                + curr_rv.cov_cholesky
                @ base_measure_realizations[idx - 1].reshape((-1,))
            )
            out_samples.append(curr_sample)
        return out_samples

    # Utility functions that are used surprisingly often:
    #
    # Call forward/backward transitions of realisations by
    # turning it into a Normal RV with zero covariance and by
    # referring to the forward/backward transition of RVs.

    def _backward_realization_via_backward_rv(self, realization, *args, **kwargs):

        real_as_rv = randvars.Constant(support=realization)
        return self.backward_rv(real_as_rv, *args, **kwargs)

    def _forward_realization_via_forward_rv(self, realization, *args, **kwargs):
        real_as_rv = randvars.Constant(support=realization)
        return self.forward_rv(real_as_rv, *args, **kwargs)
