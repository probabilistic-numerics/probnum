"""
Interfaces for Bayesian filtering and smoothing.
"""

from abc import ABC, abstractmethod


class BayesFiltSmooth(ABC):
    """
    Bayesian filtering and smoothing.
    """

    def __init__(self, dynamod, measmod, initrv):
        self.dynamod = dynamod
        self.measmod = measmod
        self.initrv = initrv

    @abstractmethod
    def filter_step(self, start, stop, randvar, data, **kwargs):
        """
        Filter step.

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

    @property
    def dynamicmodel(self):
        """
        Convenience function for accessing ``self.dynamod``.
        """
        return self.dynamod

    @property
    def measurementmodel(self):
        """
        Convenience function for accessing ``self.measmod``.
        """
        return self.measmod

    @property
    def initialrandomvariable(self):
        """
        Convenience function for accessing ``self.initrv``.
        """
        return self.initrv

    @property
    def initialdistribution(self):
        """
        Convenience function for accessing ``self.initdist``.
        """
        return self.initrv
