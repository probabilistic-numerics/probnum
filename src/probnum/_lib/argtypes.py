import numbers
from typing import Iterable, Tuple, Union

import numpy as np


IntArgType = Union[int, numbers.Integral, np.integer]
FloatArgType = Union[float, numbers.Real, np.floating]

ShapeArgType = Union[IntArgType, Iterable[IntArgType]]
DTypeArgType = Union[np.dtype, str]

ArrayLikeGetitemArgType = Union[
    int,
    slice,
    np.ndarray,
    np.newaxis,
    None,
    type(Ellipsis),
    Tuple[Union[int, slice, np.ndarray, np.newaxis, None, type(Ellipsis)], ...],
]

RandomStateArgType = Union[None, int, np.random.RandomState, np.random.Generator]
