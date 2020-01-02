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

    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov  # TODO: variance or covariance here?
        # TODO: add some type checking
        # TODO: allow construction from scipy distribution object
        raise NotImplementedError()

    # TODO: implement addition and multiplication with constant matrices / vectors
    # Example of spmatrix class: https://github.com/scipy/scipy/blob/v0.19.0/scipy/sparse/base.py#L62-L1108

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

    def __init__(self, mean=0, cov=1):
        #todo: allow for linear operators as mean and covariance
        super().__init__(mean=mean, cov=cov)

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def cdf(self, x):
        return scipy.stats.multivariate_normal.cdf(x, mean=self.mean, cov=self.cov)

    def sample(self, size=1):
        return np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
