"""
Gaussian filtering and smoothing based on making intractable quantities
tractable through Taylor-method approximations, e.g. linearization.
"""

from probnum.filtsmooth import statespace
from probnum.random_variables import Normal


class ContinuousEKF(statespace.Transition):
    """Continuous extended Kalman filter transition."""

    def __init__(self, cont_model):
        if not isinstance(cont_model, statespace.SDE):
            raise TypeError("Continuous EKF transition requires a (non-linear) SDE.")
        self.cont_model = cont_model

    def transition_realization(self, real, start, stop, **kwargs):
        return self.cont_model.transition_realization(real, start, stop, **kwargs)

    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError("TODO")

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteEKF(statespace.Transition):
    """Discrete extended Kalman filter transition."""

    def __init__(self, disc_model):
        self.disc_model = disc_model

    def transition_realization(self, real, start, stop, **kwargs):
        return self.disc_model.transition_realization(real, start, stop, **kwargs)

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

    @staticmethod
    def from_ode(self, ode, integrator):
        """Will replace `ivp2ekf` soon... """
        raise NotImplementedError
