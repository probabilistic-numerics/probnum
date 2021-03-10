"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable, and collects the time-grid as well as the discrete-time solution.
"""
import abc
import typing

import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.random_variables as pnrv
import probnum.type
from probnum import filtsmooth

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


class ODESolution(filtsmooth.FiltSmoothPosterior):
    """ODE Solution interface."""

    def __init__(self, locations, states, derivatives=None):
        super().__init__(locations=locations, states=states)
        self.derivatives = (
            pnrv_list._RandomVariableList(derivatives)
            if derivatives is not None
            else None
        )

    # Not abstract, because providing interpolation could sometimes be tedious.
    def __call__(
        self, t: typing.Union[float, typing.List[float]]
    ) -> typing.Union[pnrv.RandomVariable, pnrv_list._RandomVariableList]:
        """Evaluate the time-continuous solution at time t.

        Parameters
        ----------
        t
            Location / time at which to evaluate the continuous ODE solution.

        Returns
        -------
        Probabilistic estimate of the continuous-time solution at time ``t``.
        """
        raise NotImplementedError("Dense output is not implemented.")

    def sample(self, t=None, size=(), random_state=None):
        """Sample from the ODE solution.

        Parameters
        ----------
        t
            Location / time at which to sample.
            If nothing is specified, samples at the ODE-solver grid points are computed.
            If it is a float, a sample of the ODE-solution at this time point is computed.
            Similarly, if it is a list of floats (or an array), samples at the specified grid-points are returned.
            This is not the same as computing i.i.d samples at the respective locations.
        size
            Number of samples.
        """
        raise NotImplementedError("Sampling is not implemented.")

    def transform_base_measure_realizations(
        self, base_measure_realizations, t=None, size=()
    ):
        raise NotImplementedError("Sampling not implemented.")
