"""Random variables representing in- and outputs of probabilistic numerical methods."""

import scipy.stats
import numpy as np


class RandomVariable:
    """
    Random variables are the main objects used by probabilistic numerical methods.

    Each probabilistic numerical method should take a random variable encoding the prior distribution and output a
    random variable whose distribution encodes the uncertainty arising from finite computation. The generic signature
    of a probabilistic numerical method is:

    ```
    output_rv = probnum_method(input_rv, method_params)
    ```
    Instances of this class must at least provide a mean and a sampling method.
    """

    def __init__(self, mean=None, covariance=None):
        self.mean = mean
        self.covariance = covariance  # TODO: variance or covariance here?
        # TODO: add some type checking
        # TODO: allow construction from scipy distribution object
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def cdf(self, x):
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
        #todo: allow for linear operators as mean and covariance
        super().__init__(mean=mean, covariance=covariance)

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.covariance)

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.covariance)

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean, cov=self.covariance)

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean, cov=self.covariance, size=size)
