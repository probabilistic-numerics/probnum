"""General Gaussian filters based on approximating intractable quantities with numerical
quadrature.

Examples include the unscented Kalman filter / RTS smoother which is
based on a third degree fully symmetric rule.
"""

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv

from .linearizing_transition import LinearizingTransition
from .unscentedtransform import UnscentedTransform


class UKFComponent(LinearizingTransition):
    """Interface for unscented Kalman filtering components."""

    def __init__(
        self, non_linear_model, dimension, spread=1e-4, priorpar=2.0, special_scale=0.0
    ):
        super().__init__(non_linear_model=non_linear_model)
        self.ut = UnscentedTransform(dimension, spread, priorpar, special_scale)

        # Determine the linearization.
        # Will be constructed later.
        self.sigma_points = None

    def linearize(self, at_this_rv: pnrv.RandomVariable):
        """Assembles the sigma-points."""
        self.sigma_points = self.ut.sigma_points(at_this_rv.mean, at_this_rv.cov)


class ContinuousUKFComponent(UKFComponent):
    """Continuous unscented Kalman filter transition."""

    def __init__(
        self, non_linear_model, dimension, spread=1e-4, priorpar=2.0, special_scale=0.0
    ):
        if not isinstance(non_linear_model, pnfss.SDE):
            raise TypeError("cont_model must be an SDE.")
        super().__init__(
            non_linear_model,
            dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

        raise NotImplementedError("Implementation incomplete.")

    def transition_realization(self, real, start, stop, linearise_at=None, **kwargs):
        raise NotImplementedError("TODO")  # Issue  #234

    def transition_rv(self, rv, start, stop, linearise_at=None, **kwargs):
        raise NotImplementedError("TODO")  # Issue  #234

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteUKFComponent(UKFComponent):
    """Discrete extended Kalman filter transition."""

    def __init__(
        self, non_linear_model, dimension, spread=1.0, priorpar=2.0, special_scale=0.0
    ):
        if not isinstance(non_linear_model, pnfss.DiscreteGaussian):
            raise TypeError("cont_model must be an SDE.")
        super().__init__(
            non_linear_model,
            dimension,
            spread=spread,
            priorpar=priorpar,
            special_scale=special_scale,
        )

    def transition_realization(self, real, start, **kwargs):
        return self.non_linear_model.transition_realization(real, start, **kwargs)

    def transition_rv(self, rv, start, linearise_at=None, **kwargs):
        compute_sigmapts_at = linearise_at if linearise_at is not None else rv
        self.linearize(at_this_rv=compute_sigmapts_at)

        proppts = self.ut.propagate(
            start, self.sigma_points, self.non_linear_model.dynamicsfun
        )
        meascov = self.non_linear_model.diffmatfun(start)
        mean, cov, crosscov = self.ut.estimate_statistics(
            proppts, self.sigma_points, meascov, rv.mean
        )
        return pnrv.Normal(mean, cov), {"crosscov": crosscov}

    @property
    def dimension(self):
        return self.ut.dimension

    @classmethod
    def from_ode(cls, ode, prior, evlvar):

        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        disc_model = pnfss.DiscreteGaussian(dyna, diff)
        return cls(disc_model, dimension=prior.dimension)
