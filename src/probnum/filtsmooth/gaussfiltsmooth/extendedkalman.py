"""Gaussian filtering and smoothing based on making intractable quantities tractable
through Taylor-method approximations, e.g. linearization."""

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
        super().__init__()

    def transition_realization(self, real, start, stop, linearise_at=None, **kwargs):

        compute_jacobian_at = linearise_at.mean if linearise_at is not None else real

        def jacobfun(t, x=compute_jacobian_at):
            # replaces functools (second variable may not be called x)
            return self.non_linear_sde.jacobfun(t, x)

        step = (stop - start) / self.num_steps
        return statespace.linear_sde_statistics(
            rv=pnrv.Normal(mean=real, cov=np.zeros((len(real), len(real)))),
            start=start,
            stop=stop,
            step=step,
            driftfun=self.non_linear_sde.driftfun,
            jacobfun=jacobfun,
            dispmatfun=self.non_linear_sde.dispmatfun,
        )

    def transition_rv(self, rv, start, stop, linearise_at=None, **kwargs):

        compute_jacobian_at = linearise_at.mean if linearise_at is not None else rv.mean

        def jacobfun(t, x=compute_jacobian_at):
            # replaces functools (second variable may not be called x)
            return self.non_linear_sde.jacobfun(t, x)

        step = (stop - start) / self.num_steps
        return statespace.linear_sde_statistics(
            rv=rv,
            start=start,
            stop=stop,
            step=step,
            driftfun=self.non_linear_sde.driftfun,
            jacobfun=jacobfun,
            dispmatfun=self.non_linear_sde.dispmatfun,
        )

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteEKFComponent(statespace.Transition):
    """Discrete extended Kalman filter transition."""

    def __init__(self, disc_model):
        self.disc_model = disc_model
        super().__init__()

    def transition_realization(self, real, start, **kwargs):
        return self.disc_model.transition_realization(real, start, **kwargs)

    def transition_rv(self, rv, start, linearise_at=None, **kwargs):
        diffmat = self.disc_model.diffmatfun(start)
        compute_jacobian_at = linearise_at.mean if linearise_at is not None else rv.mean
        jacob = self.disc_model.jacobfun(start, compute_jacobian_at)
        mpred = self.disc_model.dynamicsfun(start, rv.mean)
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

        def dyna(t, x):
            return h1 @ x - ode.rhs(t, h0 @ x)

        def diff(t):
            return evlvar * np.eye(spatialdim)

        def jaco_ek1(t, x):
            return h1 - ode.jacobian(t, h0 @ x) @ h0

        def jaco_ek0(t, x):
            return h1

        if ek0_or_ek1 == 0:
            jaco = jaco_ek0
        elif ek0_or_ek1 == 1:
            jaco = jaco_ek1
        else:
            raise TypeError("ek0_or_ek1 must be 0 or 1, resp.")

        discrete_model = statespace.DiscreteGaussian(dyna, diff, jaco)
        return cls(discrete_model)
