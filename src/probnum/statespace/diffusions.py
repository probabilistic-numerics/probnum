"""Diffusion models and their calibration."""


import abc
from typing import Union

import numpy as np
import scipy.linalg

from probnum import randvars
from probnum.typing import (
    ArrayLikeGetitemArgType,
    DenseOutputLocationArgType,
    FloatArgType,
    ToleranceDiffusionType,
)


class Diffusion(abc.ABC):
    r"""Interface for diffusion models :math:`\sigma: \mathbb{R} \rightarrow \mathbb{R}^d` and their calibration."""

    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self, t: DenseOutputLocationArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        r"""Evaluate the diffusion :math:`\sigma(t)` at :math:`t`."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(
        self, idx: ArrayLikeGetitemArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_locally(
        self,
        meas_rv: randvars.RandomVariable,
        meas_rv_assuming_zero_previous_cov: randvars.RandomVariable,
        t: FloatArgType,
    ) -> ToleranceDiffusionType:
        r"""Estimate the (local) diffusion and update current (global) estimation in-
        place.

        Used for uncertainty calibration in the ODE solver.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_in_place(self, local_estimate, t):
        raise NotImplementedError


class ConstantDiffusion(Diffusion):
    r"""Constant diffusion and its calibration."""

    def __init__(self):
        self.diffusion = None
        self._seen_diffusions = 0

    def __repr__(self):
        return f"ConstantDiffusion({self.diffusion})"

    def __call__(
        self, t: DenseOutputLocationArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        if self.diffusion is None:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )
        return self.diffusion * np.ones_like(t)

    def __getitem__(
        self, idx: ArrayLikeGetitemArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        if self.diffusion is None:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )

        return self.diffusion * np.ones_like(idx)

    def estimate_locally(
        self,
        meas_rv: randvars.RandomVariable,
        meas_rv_assuming_zero_previous_cov: randvars.RandomVariable,
        t: FloatArgType,
    ) -> ToleranceDiffusionType:
        new_increment = _compute_local_quasi_mle(meas_rv)
        return new_increment

    def update_in_place(self, local_estimate, t):

        if self.diffusion is None:
            self.diffusion = local_estimate
        else:
            a = 1 / self._seen_diffusions
            b = 1 - a
            self.diffusion = a * local_estimate + b * self.diffusion
        self._seen_diffusions += 1


class PiecewiseConstantDiffusion(Diffusion):
    r"""Piecewise constant diffusion.

    It is defined by a set of diffusions :math:`(\sigma_1, ..., \sigma_N)` and a set of locations :math:`(t_0, ...,  t_N)`
    through

    .. math::
        \sigma(t) = \left\{
        \begin{array}{ll}
        \sigma_1 & \text{ if } t < t_0\\
        \sigma_n & \text{ if } t_{n-1} \leq t < t_{n}, ~n=1, ..., N\\
        \sigma_N & \text{ if } t_{N} \leq t\\
        \end{array}
        \right.

    In other words, a tuple :math:`(t, \sigma)` always defines the diffusion *right* of :math:`t` as :math:`\sigma` (including the point :math:`t`),
    except for the very first tuple :math:`(t_0, \sigma_0)` which *also* defines the diffusion *left* of :math:`t`.
    This choice of piecewise constant function is continuous from the right.

    Parameters
    ----------
    t0
        Initial time point. This is the leftmost time-point of the interval on which the diffusion is calibrated.
    """

    def __init__(self, t0):
        self._diffusions = []
        self._locations = [t0]

    def __repr__(self):
        return f"PiecewiseConstantDiffusion({self.diffusions})"

    def __call__(
        self, t: DenseOutputLocationArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        if len(self._locations) <= 1:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )
        if np.isscalar(t):
            t = np.atleast_1d(t)
            t_has_been_promoted = True
        else:
            t_has_been_promoted = False

        # The "-1" in here makes up for the fact that the locations contains one more element than the diffusions.
        indices = np.searchsorted(self.locations, t) - 1
        indices[t < self.t0] = 0
        indices[t > self.tmax] = -1

        if t_has_been_promoted:
            indices = indices[0]

        return self[indices]

    def __getitem__(
        self, idx: ArrayLikeGetitemArgType
    ) -> Union[ToleranceDiffusionType, np.ndarray]:
        if len(self._locations) <= 1:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )
        return self.diffusions[idx]

    def estimate_locally(
        self,
        meas_rv: randvars.RandomVariable,
        meas_rv_assuming_zero_previous_cov: randvars.RandomVariable,
        t: FloatArgType,
    ) -> ToleranceDiffusionType:
        if not t >= self.tmax:
            raise ValueError(
                "This time-point is not right of the current rightmost time-point."
            )
        local_quasi_mle = _compute_local_quasi_mle(meas_rv_assuming_zero_previous_cov)
        return local_quasi_mle

    def update_in_place(self, local_estimate, t):

        self._diffusions.append(local_estimate)
        self._locations.append(t)

    @property
    def locations(self) -> np.ndarray:
        return np.asarray(self._locations)

    @property
    def diffusions(self) -> np.ndarray:
        return np.asarray(self._diffusions)

    @property
    def t0(self) -> float:
        return self.locations[0]

    @property
    def tmax(self) -> float:
        return self.locations[-1]


def _compute_local_quasi_mle(meas_rv):
    std_like = meas_rv.cov_cholesky
    whitened_res = scipy.linalg.solve_triangular(std_like, meas_rv.mean, lower=True)
    ssq = whitened_res @ whitened_res / meas_rv.size
    return ssq
