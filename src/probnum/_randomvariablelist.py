from typing import Union

import numpy as np

from probnum import randvars

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


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

        # If not empty:
        if len(rv_list) > 0:

            # First element as a proxy for checking all elements
            if not isinstance(rv_list[0], randvars.RandomVariable):
                raise TypeError(
                    "RandomVariableList expects RandomVariable elements, but "
                    + f"first element has type {type(rv_list[0])}."
                )
        super().__init__(rv_list)

    def __getitem__(self, idx) -> Union[randvars.RandomVariable, "_RandomVariableList"]:

        result = super().__getitem__(idx)
        # Make sure to wrap the result into a _RandomVariableList if necessary
        if isinstance(result, list):
            result = _RandomVariableList(result)
        return result

    @cached_property
    def mean(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.mean for rv in self])

    @cached_property
    def cov(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.cov for rv in self])

    @cached_property
    def var(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.var for rv in self])

    @cached_property
    def std(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.std for rv in self])

    @property
    def shape(self):
        first_rv = np.asarray(self[0].mean)
        return (len(self),) + first_rv.shape

    @cached_property
    def mode(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.mode for rv in self])

    # For discrete random variables:

    @cached_property
    def support(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.support for rv in self])

    @cached_property
    def probabilities(self) -> np.ndarray:
        if len(self) == 0:
            return np.array([])
        return np.stack([rv.probabilities for rv in self])

    # Purely for lists of categorical random variables.
    def resample(self, rng: np.random.Generator) -> "_RandomVariableList":
        if len(self) == 0:
            return _RandomVariableList([])
        return _RandomVariableList([rv.resample(rng=rng) for rv in self])
