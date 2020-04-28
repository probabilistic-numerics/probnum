"""
Interfaces for Bayesian filtering and smoothing.
"""

from abc import ABC, abstractmethod


class BayesFiltSmooth:
    """
    """
    def __init__(self, dynamod, measmod, initrv):
        """ """
        self.dynamod = dynamod
        self.measmod = measmod
        self.initrv = initrv

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
        return self.initrv.distribution


class BayesSmoother(BayesFiltSmooth, ABC):
    """
    Abstract interface for Bayesian smoothing.

    Builds on top of a Bayesian filter.
    """
    def __init__(self, bayesfilt):
        """ """
        self.bayesfilt = bayesfilt
        super().__init__(dynamod=bayesfilt.dynamod,
                         measmod=bayesfilt.measmod,
                         initrv=bayesfilt.initrv)

    def smooth(self, data, times, **kwargs):
        """ """
        errormsg = ("smooth(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def smooth_stream(self, datastream, times, **kwargs):
        """ """
        errormsg = ("smooth_stream(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def smooth_set(self, dataset, times, **kwargs):
        """ """
        errormsg = ("smooth_set(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def smooth_filteroutput(self, **kwargs):
        """ """
        errormsg = ("smooth_filteroutput(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def smoothing_step(self, **kwargs):
        """ """
        errormsg = ("smoothing_step(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)


class BayesFilter(BayesFiltSmooth, ABC):
    """
    Abstract interface for Bayesian filtering.
    """

    def predict(self, start, stop, randvar, **kwargs):
        """
        Prediction step of the Bayesian filter.

        Not required for all filters, e.g. the Particle Filter only
        has an `update()` method.
        """
        errormsg = ("predict(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    @abstractmethod
    def update(self, start, stop, randvar, data, **kwargs):
        """
        Update step of the Bayesian filter.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def filter(self, data, times, **kwargs):
        """ """
        errormsg = ("filter(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def filter_set(self, dataset, times, **kwargs):
        """ """
        errormsg = ("filter_set(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)

    def filter_stream(self, datastream, times, **kwargs):
        """
        """
        errormsg = ("filter_stream(...) is not implemented for "
                    + "the Bayesian filter {}.".format(type(self).__name__))
        raise NotImplementedError(errormsg)
