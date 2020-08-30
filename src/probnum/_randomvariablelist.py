import numpy as np


class _RandomVariableList(list):
    """
    List of RandomVariables with convenient access to means, covariances, etc.

    Parameters
    ----------
    rv_list : :obj:`list` of :obj:`RandomVariable`
    """

    def __init__(self, rv_list):
        if not isinstance(rv_list, list):
            raise TypeError("RandomVariableList expects a list.")
        super().__init__(rv_list)

    def mean(self):
        return np.stack([rv.mean for rv in self])

    def cov(self):
        return np.stack([rv.cov for rv in self])

    def var(self):
        return np.stack([rv.var for rv in self])

    def std(self):
        return np.stack([rv.std for rv in self])

    def __getitem__(self, idx):
        """Make sure to wrap the result into a _RandomVariableList if necessary"""
        result = super().__getitem__(idx)
        if isinstance(result, list):
            result = _RandomVariableList(result)
        return result
