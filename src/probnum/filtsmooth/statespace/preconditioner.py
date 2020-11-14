"""Coordinate changes in state space models."""

import abc

try:
    # lru_cache and cached_property are only available in Python >=3.8
    from functools import cached_property, lru_cache
except ImportError:
    from cached_property import cached_property
    from lru_cache import lru_cache

import numpy as np
import scipy.special  # for vectorised factorial


class Preconditioner(abc.ABC):
    """
    Coordinate change transformations as preconditioners in state space models.

    For some models, this makes the filtering and smoothing steps more numerically stable.
    """

    @abc.abstractmethod
    def __call__(self, step) -> np.ndarray:
        # if more than step is needed, add them into the signature in the future
        raise NotImplementedError

    @cached_property
    def inverse(self) -> "Preconditioner":
        raise NotImplementedError


class TaylorCoordinates(Preconditioner):
    """
    Taylor coordinates.

    Similar to Nordsieck coordinates, but better. Used in IBM.
    """

    def __init__(self, powers, scales, spatialdim):
        # Clean way of assembling these coordinates cheaply,
        # because the powers and scales of the inverse
        # are better read off than inverted
        self.powers = powers
        self.scales = scales
        self.spatialdim = spatialdim

    @classmethod
    def from_order(cls, order, spatialdim):
        # used to conveniently initialise in the beginning
        powers = np.arange(order, -1, -1)
        scales = scipy.special.factorial(powers)
        return cls(powers=powers + 0.5, scales=scales, spatialdim=spatialdim)

    @lru_cache(maxsize=16)  # cache in case of fixed step-sizes
    def __call__(self, step):
        scaling_vector = step ** self.powers / self.scales
        return np.kron(np.eye(self.spatialdim), np.diag(scaling_vector))

    @cached_property
    def inverse(self) -> "TaylorCoordinates":
        return TaylorCoordinates(
            powers=-self.powers, scales=1.0 / self.scales, spatialdim=self.spatialdim
        )


#
#
#
# def smooth_step(self, unsmoothed_rv, smoothed_rv, start, stop, **kwargs):
#     """
#     A single smoother step.
#
#     Consists of predicting from the filtering distribution at time t
#     to time t+1 and then updating based on the discrepancy to the
#     smoothing solution at time t+1.
#
#     Parameters
#     ----------
#     unsmoothed_rv : RandomVariable
#         Filtering distribution at time t.
#     smoothed_rv : RandomVariable
#         Prediction at time t+1 of the filtering distribution at time t.
#     start : float
#         Time-point of the to-be-smoothed RV.
#     stop : float
#         Time-point of the already-smoothed RV.
#     """
#     # This implementation leverages preconditioning
#     # for numerically stable smoothing steps -- if applicable.
#     if dynamod.preconditioner is not None:
#         return _preconditioned_smooth_step()
#     else:
#         return _smooth_step()
#
# def _preconditioned_smooth_step()
#     unsmoothed_rv = dynamod.coord_change(unsmoothed_rv)
#     smoothed_rv = dynamod.coord_change.fetch(smoothed_rv)
#     predicted_rv, info = self.dynamod.transition_rv(
#         unsmoothed_rv, start, stop=stop, already_preconditioned=True, **kwargs
#     )
#     crosscov = dynamod.coord_change.push(info["crosscov"])
#
#     # Update
#     smoothing_gain = np.linalg.solve(predicted_rv.cov.T, crosscov.T).T
#     new_mean = unsmoothed_rv.mean + smoothing_gain @ (
#         smoothed_rv.mean - predicted_rv.mean
#     )
#     new_cov = (
#         unsmoothed_rv.cov
#         + smoothing_gain @ (smoothed_rv.cov - predicted_rv.cov) @ smoothing_gain.T
#     )
#     smoothed_rv = pnrv.Normal(new_mean, new_cov)
#     return dynamod.coord_change.push(smoothed_rv)
#
#
# class Transition:
#
#     # Useful preconditioners are class attribues,
#     # not instance attributes
#     # Default preconditioner is empty
#     preconditioner = None
#     # implementation...
