"""Array object in NumPy."""
from typing import Literal, Tuple, Union

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    generic as Scalar,
    ndarray as Array,
    ndim,
)

Device = Literal["cpu"]


def to_numpy(*arrays: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return arrays[0]

    return tuple(arrays)
