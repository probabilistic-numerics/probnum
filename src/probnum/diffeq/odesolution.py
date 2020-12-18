"""ODESolution object, returned by `probsolve_ivp`

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc

import numpy as np

import probnum.random_variable as pnrv
from probnum import utils
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth import KalmanPosterior
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ODESolution(abc.ABC):
    """ODE Solution interface."""

    def __call__(self, t: float) -> pnrv.RandomVariable:
        """Evaluate the time-continuous solution at time t.

        Parameters
        ----------
        t
            Location / time at which to evaluate the continuous ODE solution.

        Returns
        -------
        :obj:`RandomVariable`
            Probabilistic estimate of the continuous-time solution at time ``t``.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> pnrv.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        raise NotImplementedError

    @property
    def t(self) -> float:
        """Time points of the discrete-time solution."""
        raise NotImplementedError

    @property
    def y(self) -> _RandomVariableList:
        """:obj:`list` of :obj:`RandomVariable`: Probabilistic discrete-time solution

        Probabilistic discrete-time solution at times :math:`t_1, ..., t_N`,
        as a list of random variables.
        To return means and covariances use ``y.mean`` and ``y.cov``.
        """
        raise NotImplementedError
