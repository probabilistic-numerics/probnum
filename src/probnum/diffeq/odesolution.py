"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable, and collects the time-grid as well as the discrete-time solution.
"""

from typing import List, Union

from probnum import _randomvariablelist, filtsmooth, random_variables
from probnum.filtsmooth.timeseriesposterior import (
    DenseOutputLocationArgType,
    DenseOutputValueType,
)


class ODESolution(filtsmooth.TimeSeriesPosterior):
    """ODE solution.

    Parameters
    ----------
    locations
        Locations of the time-grid that was used by the ODE solver.
    states
        Output of the ODE solver at the locations.
    derivatives
        Derivatives of the states at the locations. Optional. Default is None.
        Some ODE solvers provide these estimates, others do not.
    """

    def __init__(
        self,
        locations: np.ndarray,
        states: _randomvariablelist._RandomVariableList,
        derivatives: Optional[_randomvariablelist._RandomVariableList] = None,
    ):
        super().__init__(locations=locations, states=states)
        self.derivatives = (
            _randomvariablelist._RandomVariableList(derivatives)
            if derivatives is not None
            else None
        )

    # Not abstract, because providing interpolation could sometimes be tedious.
    def __call__(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:
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

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
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

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
    ) -> np.ndarray:
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
