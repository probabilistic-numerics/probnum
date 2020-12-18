"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable, and collects the time-grid as well as the discrete-time solution.
Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc
import typing

import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.random_variables as pnrv
import probnum.type


class ODESolution(abc.ABC):
    """ODE Solution interface."""

    @property
    @abc.abstractmethod
    def t(self) -> np.ndarray:
        """Time points of the discrete-time solution."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self) -> pnrv_list._RandomVariableList:
        """Discrete-time solution."""
        raise NotImplementedError

    @property
    def dy(self) -> pnrv_list._RandomVariableList:
        """First derivative of the discrete-time solution."""
        raise NotImplementedError("The first derivative has not been implemented")

    # Not abstract, because providing interpolation could sometimes be tedious.
    def __call__(
        self, t: float
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

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.y)

    def __getitem__(self, idx: int) -> pnrv.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.y[idx]

    def sample(
        self,
        t: typing.Optional[float] = None,
        size: typing.Optional[probnum.type.ShapeArgType] = (),
    ) -> np.ndarray:
        """Sample from the ODE solution."""
        raise NotImplementedError("Sampling is not implemented.")
