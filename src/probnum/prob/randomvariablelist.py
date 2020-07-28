import numpy as np


class _RandomVariableList(list):
    """List of RandomVariables with convenient access to means, covariances, etc."""

    @property
    def mean(self):
        return np.stack([rv.mean() for rv in self])

    @property
    def cov(self):
        return np.stack([rv.cov() for rv in self])

    @property
    def var(self):
        return np.stack([rv.distribution.var() for rv in self])

    @property
    def std(self):
        return np.stack([rv.distribution.std() for rv in self])
