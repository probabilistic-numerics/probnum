"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable, and collects the time-grid as well as the discrete-time solution.
Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import abc

import probnum._randomvariablelist as pnrv_list
import probnum.random_variable as pnrv


class ODESolution(abc.ABC):
    """ODE Solution interface."""

    @property
    @abc.abstractmethod
    def t(self) -> float:
        """Time points of the discrete-time solution."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self) -> pnrv_list._RandomVariableList:
        """:obj:`list` of :obj:`RandomVariable`: Probabilistic discrete-time solution

        Probabilistic discrete-time solution at times :math:`t_1, ..., t_N`,
        as a list of random variables.
        To return means and covariances use ``y.mean`` and ``y.cov``.
        """
        raise NotImplementedError

    # Not abstract, because providing interpolation could sometimes be tedious.
    def __call__(self, t: float) -> pnrv.RandomVariable:
        """Evaluate the time-continuous solution at time t.

        Parameters
        ----------
        t
            Location / time at which to evaluate the continuous ODE solution.

        Returns
        -------
        Probabilistic estimate of the continuous-time solution at time ``t``.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.y)

    def __getitem__(self, idx: int) -> pnrv.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.y[idx]
