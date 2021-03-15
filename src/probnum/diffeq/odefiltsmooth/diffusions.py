"""Diffusion calibrations.

Maybe move this to `statespace` or `filtsmooth` in the future, but for
now, it is well placed in `diffeq`.
"""


import abc
from typing import Union

import numpy as np

from probnum import random_variables
from probnum.type import FloatArgType

DiffusionType = Union[FloatArgType]
"""Acceptable diffusion types are -- in principle -- floats and arrays of shape (d,).
In other words, everything that behaves well with `cov1 + diffusion*cov2` and
`chol1 + sqrt(diffusion) * chol2`.
For now, let's only use floats, because it is not clear to me how to best implement the square-root behaviour.
"""


class Diffusion(abc.ABC):
    r"""Interface for diffusion models :math:`\sigma: \mathbb{R} \rightarrow \mathbb{R}^d` and their calibration."""

    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, t) -> DiffusionType:
        """Evaluate the diffusion."""
        raise NotImplementedError

    def calibrate_locally(self, meas_rv):
        """Compute a local diffusion estimate.

        Used for uncertainty calibration in the ODE solver.
        """
        std_like = meas_rv.cov_cholesky
        whitened_res = np.linalg.solve(std_like, meas_rv.mean)
        ssq = whitened_res @ whitened_res / meas_rv.size
        return ssq

    @abc.abstractmethod
    def update_current_information(self, diffusion, t):
        """Update the current information about the global diffusion.

        This could mean appending the diffusion to a list or updating a
        global estimate.
        """
        pass

    @abc.abstractmethod
    def calibrate_all_states(self, states, locations):
        """Calibrate a set of ODE solver states after seeing all the data."""
        pass


class ConstantDiffusion(Diffusion):
    """Constant diffusion and its calibration."""

    def __init__(self):
        self.diffusion = None
        self._seen_diffusions = 0

    def __repr__(self):
        return f"ConstantDiffusion({self.diffusion})"

    def __call__(self, t) -> DiffusionType:
        return self.diffusion

    def update_current_information(self, diffusion, t):
        """Update the current global MLE with a new diffusion."""
        self._seen_diffusions += 1

        if self.diffusion is None:
            self.diffusion = diffusion
        else:
            a = 1 / self._seen_diffusions
            b = 1 - a
            self.diffusion = a * diffusion + b * self.diffusion

    def calibrate_all_states(self, states, locations):
        return [
            random_variables.Normal(rv.mean, self.diffusion * rv.cov) for rv in states
        ]


class PiecewiseConstantDiffusion(Diffusion):
    r"""Piecewise constant diffusion.

    It is defined by a set of diffusions :math:`(\sigma_0, ..., \sigma_N)` and a set of locations :math:`(t_0, ...,  t_N)`.
    Based on this, it is defined as

    .. math::
        \sigma(t) = \left\{
        \begin{array}{ll}
        \sigma_0 & \text{ if } t < t_0\\
        \sigma_n & \text{ if } t_{n-1} \leq t < t_{n}, ~n=1, ..., N\\
        \sigma_N & \text{ if } t_{N} \leq t\\
        \end{array}
        \right.

    This definition implies that at all points :math:`t \geq t_{N-1}`, its value is `\sigma_N`.
    This choice of piecewise constant function is by definition right-continuous.
    """

    def __init__(self):
        self.diffusions = []
        self.locations = []

    def __repr__(self):
        return f"PiecewiseConstantDiffusion({self.diffusions})"

    def __call__(self, t) -> DiffusionType:
        """Evaluate the diffusion."""
        # Indices in self.locations that are larger than t
        # The first element in this list is the first time-point right of t.
        idx = np.nonzero(t < self.locations)[0]
        return self.diffusions[idx[0]] if len(idx) > 0 else self.diffusions[-1]

    def update_current_information(self, diffusion, t):
        """Append the most recent diffusion and location to a list.

        This function assumes that the list of times and diffusions is
        sorted, and that the input `t` lies "right of the final point"
        in the current list.
        """
        self.diffusions.append(diffusion)
        self.locations.append(t)

    def calibrate_all_states(self, states, locations):
        return states
