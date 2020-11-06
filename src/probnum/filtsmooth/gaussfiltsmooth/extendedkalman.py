"""
Gaussian filtering and smoothing based on making intractable quantities
tractable through Taylor-method approximations, e.g. linearization.
"""
import numpy as np

from probnum.filtsmooth import statespace
from probnum.random_variables import Normal


class ContinuousEKFComponent(statespace.Transition):
    """Continuous extended Kalman filter transition."""

    def __init__(self, non_linear_sde):
        if not isinstance(non_linear_sde, statespace.SDE):
            raise TypeError("Continuous EKF transition requires a (non-linear) SDE.")
        self.cont_model = non_linear_sde
        raise NotImplementedError("Implementation incomplete.")

    def transition_realization(self, real, start, stop, **kwargs):
        raise NotImplementedError("TODO")  # Issue  #234

    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError("TODO")  # Issue  #234

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteEKFComponent(statespace.Transition):
    """Discrete extended Kalman filter transition."""

    def __init__(self, disc_model):
        self.disc_model = disc_model

    def transition_realization(self, real, start, **kwargs):
        return self.disc_model.transition_realization(real, start, **kwargs)

    def transition_rv(self, rv, start, **kwargs):
        diffmat = self.disc_model.diffusionmatrix(start)
        jacob = self.disc_model.jacobian(start, rv.mean)
        mpred = self.disc_model.dynamics(start, rv.mean)
        crosscov = rv.cov @ jacob.T
        cpred = jacob @ crosscov + diffmat
        return Normal(mpred, cpred), {"crosscov": crosscov}

    @property
    def dimension(self):
        raise NotImplementedError

    @classmethod
    def from_ode(cls, ode, prior, evlvar, ek0_or_ek1=0):
        spatialdim = prior.spatialdim
        h0 = prior.proj2coord(coord=0)
        h1 = prior.proj2coord(coord=1)

        def dyna(t, x, **kwargs):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t, **kwargs):
            return evlvar * np.eye(spatialdim)

        def jaco_ek1(t, x, **kwargs):
            return h1 - ode.jacobian(t, h0 @ x) @ h0

        def jaco_ek0(t, x, **kwargs):
            return h1

        if ek0_or_ek1 == 0:
            jaco = jaco_ek0
        elif ek0_or_ek1 == 1:
            jaco = jaco_ek1
        else:
            raise TypeError("ek0_or_ek1 must be 0 or 1, resp.")

        discrete_model = statespace.DiscreteGaussian(dyna, diff, jaco)
        return cls(discrete_model)
