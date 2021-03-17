"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable, and collects the time-grid as well as the discrete-time solution.
"""
import abc
import typing

import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.type
from probnum import randvars

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


class ODESolution(abc.ABC):
    """ODE Solution interface."""

    @property
    @abc.abstractmethod
    def t(self) -> np.ndarray:
        """Time points of the discrete-time solution."""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def y(self) -> pnrv_list._RandomVariableList:
        """Discrete-time solution."""
        raise NotImplementedError

    @cached_property
    def dy(self) -> pnrv_list._RandomVariableList:
        """First derivative of the discrete-time solution."""
        raise NotImplementedError("The first derivative has not been implemented")

    # Not abstract, because providing interpolation could sometimes be tedious.
    def __call__(
        self, t: typing.Union[float, typing.List[float]]
    ) -> typing.Union[randvars.RandomVariable, pnrv_list._RandomVariableList]:
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

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.y)

    def __getitem__(self, idx: int) -> randvars.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.y[idx]

    def sample(
        self,
        t: typing.Optional[typing.Union[float, typing.List[float]]] = None,
        size: typing.Optional[probnum.type.ShapeArgType] = (),
    ) -> np.ndarray:
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
