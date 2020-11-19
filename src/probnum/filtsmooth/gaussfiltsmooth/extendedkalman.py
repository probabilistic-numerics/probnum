"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""
import functools

import numpy as np

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace


class ContinuousEKFComponent(statespace.Transition):
    """Continuous extended Kalman filter transition."""

    def __init__(self, non_linear_sde, num_steps):
        if not isinstance(non_linear_sde, statespace.SDE):
            raise TypeError("Continuous EKF transition requires a (non-linear) SDE.")
        self.non_linear_sde = non_linear_sde
        self.num_steps = num_steps

    def transition_realization(self, real, start, stop, linearise_at=None, **kwargs):
        compute_jacobian_at = linearise_at.mean if linearise_at else real
        jacobfun = functools.partial(
            self.non_linear_sde.jacobian, state=compute_jacobian_at
        )
        step = (stop - start) / self.num_steps
        return statespace.linear_sde_statistics(
            rv=pnrv.Normal(mean=real, cov=np.zeros((len(real), len(real)))),
            start=start,
            stop=stop,
            step=step,
            driftfun=self.non_linear_sde.drift,
            jacobfun=jacobfun,
            dispmatfun=self.non_linear_sde.dispersionmatrix,
        )

    def transition_rv(self, rv, start, stop, linearise_at=None, **kwargs):
        compute_jacobian_at = linearise_at.mean if linearise_at else rv.mean
        jacobfun = functools.partial(
            self.non_linear_sde.jacobian, state=compute_jacobian_at
        )
        step = (stop - start) / self.num_steps
        return statespace.linear_sde_statistics(
            rv=rv,
            start=start,
            stop=stop,
            step=step,
            driftfun=self.non_linear_sde.drift,
            jacobfun=jacobfun,
            dispmatfun=self.non_linear_sde.dispersionmatrix,
        )

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteEKFComponent(statespace.Transition):
    """Discrete extended Kalman filter transition."""

    def __init__(self, disc_model):
        self.disc_model = disc_model

    def transition_realization(self, real, start, **kwargs):
        return self.disc_model.transition_realization(real, start, **kwargs)

    def transition_rv(self, rv, start, linearise_at=None, **kwargs):
        diffmat = self.disc_model.diffusionmatrix(start)
        if linearise_at is None:
            jacob = self.disc_model.jacobian(start, rv.mean)
        else:
            jacob = self.disc_model.jacobian(start, linearise_at.mean)
        mpred = self.disc_model.dynamics(start, rv.mean)
        crosscov = rv.cov @ jacob.T
        cpred = jacob @ crosscov + diffmat
        return pnrv.Normal(mpred, cpred), {"crosscov": crosscov}

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
