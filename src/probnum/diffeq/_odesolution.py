"""ODESolution interface.

This object is returned by ODESolver.solve().

Provides dense output (by being callable), is sliceable,
and collects the time-grid as well as the discrete-time solution.
"""

from typing import Optional

import numpy as np

from probnum import filtsmooth, randvars
from probnum.typing import ArrayLike, FloatLike, IntLike, ShapeLike


class ODESolution(filtsmooth.TimeSeriesPosterior):
    """Interface for ODE solutions in ProbNum.

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
        states: randvars._RandomVariableList,
        derivatives: Optional[randvars._RandomVariableList] = None,
    ):
        super().__init__(locations=locations, states=states)
        self.derivatives = (
            randvars._RandomVariableList(derivatives)
            if derivatives is not None
            else None
        )

    def interpolate(
        self,
        t: FloatLike,
        previous_index: Optional[IntLike] = None,
        next_index: Optional[IntLike] = None,
    ) -> randvars.RandomVariable:
        raise NotImplementedError("Dense output is not implemented.")

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.states)

    def __getitem__(self, idx: int) -> randvars.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.states[idx]

    def sample(
        self,
        rng: np.random.Generator,
        t: Optional[ArrayLike] = None,
        size: Optional[ShapeLike] = (),
    ) -> np.ndarray:
        """Sample from the ODE solution.

        Parameters
        ----------
        rng
            Random number generator.
        t
            Location / time at which to sample.
            If nothing is specified, samples at the ODE-solver
            grid points are computed.
            If it is a float, a sample of the ODE-solution
            at this time point is computed.
            Similarly, if it is a list of floats (or an array),
            samples at the specified grid-points are returned.
            This is not the same as computing i.i.d samples at the respective
            locations.
        size
            Number of samples.
        """
        raise NotImplementedError("Sampling is not implemented.")

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: ArrayLike,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
