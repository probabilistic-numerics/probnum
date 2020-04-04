"""
Univariate normal distribution.
"""

import numpy as np
import scipy.stats
import scipy.sparse
import scipy._lib._util

from probnum.prob.distributions.dirac import Dirac
from probnum.prob.distributions.normal._normal import _Normal


class _UnivariateNormal(_Normal):
    """
    The univariate normal distribution.
    """

    def __init__(self, mean=0., cov=1., random_state=None):
        super().__init__(mean=float(mean), cov=float(cov), random_state=random_state)

    def var(self):
        return self.cov()

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, loc=self.mean(), scale=self.std())

    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mean(), scale=self.std())

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, loc=self.mean(), scale=self.std())

    def logcdf(self, x):
        return scipy.stats.norm.logcdf(x, loc=self.mean(), scale=self.std())

    def sample(self, size=()):
        return scipy.stats.norm.rvs(loc=self.mean(), scale=self.std(),
                                    size=size, random_state=self.random_state)

    def reshape(self, shape):
        raise NotImplementedError

    # Arithmetic Operations ###############################

    def __matmul__(self, other):
        """
        TODO
        ----
        Implement special rules for matrix-variate RVs and
        Kronecker structured covariances (see e.g. p.64
        Thm. 2.3.10 of Gupta: Matrix-variate distribution)
        """
        if isinstance(other, Dirac):
            delta = other.mean()
            return _Normal(mean=np.squeeze(self.mean() @ delta),
                          cov=np.squeeze(delta @ (self.cov() @ delta.transpose())),
                          random_state=self.random_state)
        return NotImplemented






