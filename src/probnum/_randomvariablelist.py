from typing import Union

import numpy as np

import probnum as pn


class _RandomVariableList(list):
    """List of RandomVariables with convenient access to means, covariances, etc.

    Parameters
    ----------
    rv_list :
        :obj:`list` of :obj:`RandomVariable`
    """

    def __init__(self, rv_list: list):
        if not isinstance(rv_list, list):
            raise TypeError("RandomVariableList expects a list.")

        # First element as a proxy for checking all elements
        if not isinstance(rv_list[0], pn.RandomVariable):
            raise TypeError(
                "RandomVariableList expects RandomVariable elements, but "
                + f"first element has type {type(rv_list[0])}."
            )
        super().__init__(rv_list)

    @property
    def mean(self) -> np.ndarray:
        return np.stack([rv.mean for rv in self])

    @property
    def cov(self) -> np.ndarray:
        return np.stack([rv.cov for rv in self])

    @property
    def var(self) -> np.ndarray:
        return np.stack([rv.var for rv in self])

    @property
    def std(self) -> np.ndarray:
        return np.stack([rv.std for rv in self])

    def __getitem__(self, idx) -> Union["pn.RandomVariable", list]:
        result = super().__getitem__(idx)
        # Make sure to wrap the result into a _RandomVariableList if necessary
        if isinstance(result, list):
            result = _RandomVariableList(result)
        return result
