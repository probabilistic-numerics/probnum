"""Random variables representing in- and outputs of probabilistic numerical methods."""

import scipy.stats
import numpy as np


class RandomVariable:

    def __init__(self, mean=None, covariance=None):
        self.mean = mean
        self.covariance = covariance  # TODO: variance or covariance here?
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def sample(self, size=1):
        """
        Returns realizations from the associated random variable.

        Parameters
        ----------
        size : tuple, default=1
            Shape of the realizations.

        Returns
        -------

        """
        raise NotImplementedError()


class Normal(RandomVariable):

    def __init__(self, mean=0, covariance=1):
        super().__init__(mean=mean, covariance=covariance)

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.covariance)

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.covariance)

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean, cov=self.covariance, size=size)
