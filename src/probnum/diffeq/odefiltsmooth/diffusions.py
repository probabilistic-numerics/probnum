"""Diffusion calibrations.

Maybe move this to `statespace` or `filtsmooth` in the future, but for
now, it is well placed in `diffeq`.
"""


import abc

import numpy as np
import scipy.linalg

from probnum.type import ToleranceDiffusionType


class Diffusion(abc.ABC):
    r"""Interface for diffusion models :math:`\sigma: \mathbb{R} \rightarrow \mathbb{R}^d` and their calibration."""

    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, t) -> ToleranceDiffusionType:
        """Evaluate the diffusion."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_locally_and_update_in_place(
        self, meas_rv, meas_rv_assuming_zero_previous_covariance, t
    ):
        """Estimate the (local) diffusion and update current (global) estimation in-
        place.

        Used for uncertainty calibration in the ODE solver.
        """
        raise NotImplementedError


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

    def __init__(self):
        self.diffusion = None
        self._seen_diffusions = 0

    def __repr__(self):
        return f"ConstantDiffusion({self.diffusion})"

    def __call__(self, t) -> ToleranceDiffusionType:
        if self.diffusion is None:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )
        return self.diffusion * np.ones_like(t)

    def estimate_locally_and_update_in_place(
        self, meas_rv, meas_rv_assuming_zero_previous_cov, t
    ):
        new_increment = _compute_local_quasi_mle(meas_rv)
        if self.diffusion is None:
            self.diffusion = new_increment
        else:
            a = 1 / self._seen_diffusions
            b = 1 - a
            self.diffusion = a * new_increment + b * self.diffusion
        self._seen_diffusions += 1
        return new_increment


class PiecewiseConstantDiffusion(Diffusion):
    r"""Piecewise constant diffusion.

    It is defined by a set of diffusions :math:`(\sigma_0, ..., \sigma_N)` and a set of locations :math:`(t_0, ...,  t_N)`
    through

    .. math::
        \sigma(t) = \left\{
        \begin{array}{ll}
        \sigma_0 & \text{ if } t < t_0\\
        \sigma_n & \text{ if } t_{n-1} \leq t < t_{n}, ~n=1, ..., N\\
        \sigma_N & \text{ if } t_{N} \leq t\\
        \end{array}
        \right.

    In other words, a tuple :math:`(t, \sigma)` always defines the diffusion *right* of :math:`t` as :math:`\sigma` (including the point :math:`t`),
    except for the very first tuple :math:`(t_0, \sigma_0)` which *also* defines the diffusion *left* of :math:`t`.
    This choice of piecewise constant function is continuous from the right.

    Parameters
    ----------
    diffusions :
        List of diffusions. Optional. Default is `None`, which implies that a list is created from scratch.
        If values are provided, make sure they support an `.append` method (for instance, make them a list).
    locations :
        List of locations corresponding to the diffusions. Optional. Default is `None`, which implies that a list is created from scratch.
        If values are provided, make sure they support an `.append` method (for instance, make them a list).
    """

    def __init__(self, diffusions=None, locations=None):
        self._diffusions = [] if diffusions is None else diffusions
        self._locations = [] if locations is None else locations

    def __repr__(self):
        return f"PiecewiseConstantDiffusion({self.diffusions})"

    def __call__(self, t) -> ToleranceDiffusionType:
        if len(self._locations) == 0:
            raise NotImplementedError(
                "No diffusions seen yet. Call estimate_locally_and_update_in_place first."
            )

        indices = np.searchsorted(self.locations[:-1], t, side="right")

        return self.diffusions[indices]

    def estimate_locally_and_update_in_place(
        self, meas_rv, meas_rv_assuming_zero_previous_cov, t
    ):
        local_quasi_mle = _compute_local_quasi_mle(meas_rv_assuming_zero_previous_cov)
        self._diffusions.append(local_quasi_mle)
        self._locations.append(t)
        return local_quasi_mle

    @property
    def locations(self):
        return np.asarray(self._locations)

    @property
    def diffusions(self):
        return np.asarray(self._diffusions)


def _compute_local_quasi_mle(meas_rv):
    std_like = meas_rv.cov_cholesky
    whitened_res = scipy.linalg.solve_triangular(std_like, meas_rv.mean, lower=True)
    ssq = whitened_res @ whitened_res / meas_rv.size
    return ssq
