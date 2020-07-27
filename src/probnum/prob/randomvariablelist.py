import numpy as np


class _RandomVariableList(list):
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
