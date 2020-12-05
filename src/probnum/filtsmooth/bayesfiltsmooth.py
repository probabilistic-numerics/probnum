"""Interfaces for Bayesian filtering and smoothing."""

from abc import ABC, abstractmethod


class BayesFiltSmooth(ABC):
    """Bayesian filtering and smoothing."""

    def __init__(self, dynamics_model, measurement_model, initrv):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv

    @abstractmethod
    def filter_step(self, start, stop, randvar, data, **kwargs):
        """Filter step.

        For e.g. Gaussian filters, this means a prediction step followed
        by an update step.
        """
        errormsg = (
            "filter_step(...) is not implemented for "
            + "the Bayesian filter {}.".format(type(self).__name__)
        )
        raise NotImplementedError(errormsg)

    def smoother_step(self, **kwargs):
        """Smoother step."""
        errormsg = (
            "smoother_step(...) is not implemented for "
            + "the Bayesian smoother {}.".format(type(self).__name__)
        )
        raise NotImplementedError(errormsg)
