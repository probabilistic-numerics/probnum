"""
General Gaussian filters based on approximating intractable
quantities with numerical quadrature.
Examples include the unscented Kalman filter / RTS smoother
which is based on a third degree fully symmetric rule.
"""


from probnum.filtsmooth import statespace
from probnum.filtsmooth.gaussfiltsmooth import unscentedtransform as ut
from probnum.random_variables import Normal


class ContinuousUKF(statespace.Transition):
    """Continuous unscented Kalman filter transition."""

    def __init__(self, cont_model):
        if not isinstance(cont_model, statespace.ContinuousModel):
            raise TypeError
        self.cont_model = cont_model

    def transition_realization(self, real, start, stop, **kwargs):
        return self.cont_model.transition_realization(real, start, stop, **kwargs)

    def transition_rv(self, rv, start, stop, **kwargs):
        raise NotImplementedError  # Todo

    @property
    def dimension(self):
        raise NotImplementedError


class DiscreteUKF(statespace.Transition):
    """Discrete extended Kalman filter transition."""

    def __init__(self, disc_model, alpha, beta, kappa):
        self.disc_model = disc_model
        self.ut = ut.UnscentedTransform(self.disc_model.dimension, alpha, beta, kappa)

    def transition_realization(self, real, start, stop, **kwargs):
        return self.disc_model.transition_realization(real, start, stop, **kwargs)

    def transition_rv(self, rv, start, **kwargs):
        sigmapts = ut.sigma_points(rv.mean, rv.cov)
        proppts = ut.propagate(start, sigmapts, self.disc_model.dynamics)
        meascov = self.disc_model.diffusionmatrix(start, **kwargs)
        mean, cov, crosscov = ut.estimate_statistics(
            proppts, sigmapts, meascov, rv.mean
        )
        return Normal(mean, cov), {"crosscov": crosscov}

    @property
    def dimension(self):
        raise NotImplementedError

    @staticmethod
    def from_ode(self, ode, integrator):
        """Will replace `ivp2ekf` soon... """
        raise NotImplementedError
