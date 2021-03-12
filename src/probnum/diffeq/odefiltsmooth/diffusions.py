"""Diffusion calibrations."""


import abc
from typing import Union

import numpy as np

from probnum import random_variables
from probnum.type import FloatArgType

DiffusionType = Union[FloatArgType]
"""Acceptable diffusion types are -- in principle -- floats and arrays of shape (d,).
In other words, everything that behaves well with `cov1 + diffusion*cov2` and
`chol1 + sqrt(diffusion) * chol2`.
For now, let's only use floats.
"""


class Diffusion:
    """Interface for (piecewise constant) diffusions and their calibration."""

    def __repr__(self):
        raise NotImplementedError

    def __call__(self, t) -> DiffusionType:
        """Evaluate the diffusion."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get the i-th diffusion from the list."""
        pass

    def calibrate_locally(self, meas_rv):
        """Compute a local diffusion estimate.

        Used for uncertainty calibration in the ODE solver.
        """
        std_like = meas_rv.cov_cholesky
        whitened_res = np.linalg.solve(std_like, meas_rv.mean)
        ssq = whitened_res @ whitened_res / meas_rv.size
        return ssq

    def update_current_information(self, diffusion, t):
        """Update the current information about the global diffusion.

        This could mean appending the diffusion to a list or updating a
        global estimate.
        """
        pass

    def postprocess_states(self, states, locations):
        """Postprocess a set of ODE solver states after seeing all the data."""
        pass


class ConstantDiffusion(Diffusion):
    """Piecewise constant diffusion and their calibration."""

    def __init__(self):
        self.diffusion = None
        self._seen_diffusions = 0

    def __repr__(self):
        return f"ConstantDiffusion({self.diffusion})"

    def __call__(self, t) -> DiffusionType:
        return self.diffusion

    def __getitem__(self, idx):
        return self.diffusion

    def postprocess_states(self, states, locations):
        return [
            random_variables.Normal(rv.mean, self.diffusion * rv.cov) for rv in states
        ]

    def update_current_information(self, diffusion, t):
        self._seen_diffusions += 1

        if self.diffusion is None:
            self.diffusion = diffusion
        else:
            # on the fly update for mean
            a = 1 / self._seen_diffusions
            b = 1 - a
            self.diffusion = a * diffusion + b * self.diffusion


class PiecewiseConstantDiffusion(Diffusion):
    """Piecewise constant diffusion."""

    def __init__(self):
        self.diffusions = []
        self.times = []

    def __repr__(self):
        return f"PiecewiseConstantDiffusion({self.diffusions})"

    def __call__(self, t) -> DiffusionType:
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.diffusions[idx]

    def postprocess_states(self, states, locations):
        return states

    def update_current_information(self, diffusion, t):
        """Update the current information about the global diffusion.

        This could mean appending the diffusion to a list or updating a
        global estimate.
        """
        self.diffusions.append(diffusion)
        self.times.append(t)
