"""Coordinate changes in state space models."""

import abc
import functools

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

    @property
    def inverse(self) -> "Preconditioner":
        raise NotImplementedError


# Better called TaylorCoordinates or something like this?
class NordsieckCoordinates(Preconditioner):
    """NordsieckCoordinates."""

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
        powers = np.arange(0, order + 1)
        scales = np.flip(scipy.special.factorial(powers))
        return cls(powers, scales, spatialdim)

    @functools.lru_cache(maxsize=16)  # cache in case of fixed step-sizes
    def __call__(self, step):
        scaling_vector = step ** self.powers / self.scales
        return np.kron(np.eye(self.spatialdim), np.diag(scaling_vector))

    @functools.cached_property
    def inverse(self) -> "Preconditioner":
        return NordsieckCoordinates(
            powers=-self.powers, scales=1.0 / self.scales, spatialdim=self.spatialdim
        )


class TaylorCoordinates(Preconditioner):
    """
    Taylor coordinates.

    Similar to NordsieckCoordinates, but better. Used in IBM.
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

    @functools.lru_cache(maxsize=16)  # cache in case of fixed step-sizes
    def __call__(self, step):
        scaling_vector = step ** self.powers / self.scales
        return np.kron(np.eye(self.spatialdim), np.diag(scaling_vector))

    @functools.cached_property
    def inverse(self) -> "TaylorCoordinates":
        return TaylorCoordinates(
            powers=-self.powers, scales=1.0 / self.scales, spatialdim=self.spatialdim
        )


#
#
#
#
#
#
# class IBM(LTISDE):
#
#     preconditioner = NordsieckCoordinates
#
#     def __init__(self, ordint, spatialdim, diffconst):
#         self.diffconst = diffconst
#         self.equivalent_discretisation = self.discretise()
#         self.precond = self.preconditioner.from_order(
#             ordint, spatialdim
#         )  # initialise preconditioner class
#
#     def transition_rv(self, rv, start, stop, already_preconditioned=False):
#         step = stop - start
#         if not already_preconditioned:
#             rv = self.precond(step) @ rv
#             rv = self.transition_rv(rv, start, stop, already_preconditioned=True)
#             return self.precond.inverse(step) @ rv
#         else:
#             return self.equivalent_discretisation.transition_rv(rv)
#
#     def transition_realization(self, rv, start, stop, already_preconditioned=False):
#         step = stop - start
#         if not already_preconditioned:
#             rv = self.preconditioner(step) @ rv
#             rv = self.transition_realization(
#                 rv, start, stop, already_preconditioned=True
#             )
#             return self.preconditioner.inverse(step) @ rv
#         else:
#             return self.equivalent_discretisation.transition_realization(rv)


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
