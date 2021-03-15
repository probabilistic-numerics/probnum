"""Diffusion calibrations.

Maybe move this to `statespace` or `filtsmooth` in the future, but for
now, it is well placed in `diffeq`.
"""


import abc
from typing import Union

import numpy as np

from probnum import random_variables
from probnum.type import FloatArgType

from ..steprule import ToleranceDiffusionType

DiffusionType = ToleranceDiffusionType


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
    def update_current_information(
        self, full_diffusion, error_free_diffusion, t
    ) -> DiffusionType:
        """Update the current information about the global diffusion and return a value
        that is used for local calibration and error estimation.

        This could mean appending the diffusion to a list or updating a
        global estimate. It could also mean returning the input value or
        the global mean.
        """
        pass

    @abc.abstractmethod
    def calibrate_all_states(self, states, locations):
        """Calibrate a set of ODE solver states after seeing all the data."""
        pass


class ConstantDiffusion(Diffusion):
    """Constant diffusion and its calibration.

    Parameters
    ----------
    use_global_estimate_as_local_estimate :
        Use the global diffusion estimate (which is the arithmetic mean of all previous diffusion estimates)
        for error estimation and re-prediction in the ODE solver. Optional. Default is `True`, which corresponds to the
        time-fixed diffusion model as used by Bosch et al. (2020) [1]_.

    References
    ----------
    .. [1] Bosch, N., and Hennig, P., and Tronarp, F..
        Calibrated Adaptive Probabilistic ODE Solvers.
        2021.
    """

    def __init__(self, use_global_estimate_as_local_estimate=True):
        self.diffusion = None
        self._seen_diffusions = 0
        self.use_global_estimate_as_local_estimate = (
            use_global_estimate_as_local_estimate
        )

    def __repr__(self):
        return f"ConstantDiffusion({self.diffusion})"

    def __call__(self, t) -> DiffusionType:
        return self.diffusion

    def update_current_information(
        self, full_diffusion, error_free_diffusion, t
    ) -> DiffusionType:
        """Update the current global MLE with a new diffusion."""
        self._seen_diffusions += 1

        if self.diffusion is None:
            self.diffusion = full_diffusion
        else:
            a = 1 / self._seen_diffusions
            b = 1 - a
            self.diffusion = a * full_diffusion + b * self.diffusion
        if self.use_global_estimate_as_local_estimate:
            return self.diffusion
        else:
            return error_free_diffusion

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

    Parameters
    ----------
    diffusions :
        List of diffusions. Optional. Default is `None`, which implies that a list is created from scratch.
    locations :
        List of locations corresponding to the diffusions. Optional. Default is `None`, which implies that a list is created from scratch.
    """

    def __init__(self, diffusions=None, locations=None):
        self.diffusions = [] if diffusions is None else diffusions
        self.locations = [] if locations is None else locations

    def __repr__(self):
        return f"PiecewiseConstantDiffusion({self.diffusions})"

    def __call__(self, t) -> DiffusionType:

        # Get indices in self.locations that are larger than t
        # The first element in this list is the first time-point right of t.
        # If the list is empty, we are extrapolating to the right.
        idx = np.nonzero(t < self.locations)[0]
        return self.diffusions[idx[0]] if len(idx) > 0 else self.diffusions[-1]

    def update_current_information(
        self, full_diffusion, error_free_diffusion, t
    ) -> DiffusionType:
        """Append the most recent diffusion and location to a list.

        This function assumes that the list of times and diffusions is
        sorted, and that the input `t` lies "right of the final point"
        in the current list.
        """
        self.diffusions.append(error_free_diffusion)
        self.locations.append(t)
        return error_free_diffusion

    def calibrate_all_states(self, states, locations):
        return states
