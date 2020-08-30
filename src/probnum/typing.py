import numbers
from typing import Iterable, Tuple, Union

import numpy as np

########################################################################################
# API Types
########################################################################################

ShapeType = Tuple[int, ...]

RandomStateType = Union[np.random.RandomState, np.random.Generator]
"""Type of a random number generator."""

########################################################################################
# Argument Types
########################################################################################

IntArgType = Union[int, numbers.Integral, np.integer]
FloatArgType = Union[float, numbers.Real, np.floating]

ShapeArgType = Union[IntArgType, Iterable[IntArgType]]
"""Type of a public API argument for supplying a shape. Values of this type should
always be converted into :class:`ShapeType` using the function
:func:`probnum.utils.as_shape` before further internal processing."""

DTypeArgType = Union[np.dtype, str]
"""Type of a public API argument for supplying a dtype. Values of this type should
always be converted into :class:`np.dtype` using the function
:func:`np.dtype` before further internal processing."""

ScalarArgType = Union[int, float, complex, numbers.Number, np.float_]
"""Type of a public API argument for supplying a scalar value. Values of this type
should always be converted into :class:`np.generic` using the function
:func:`probnum.utils.as_scalar` before further internal processing."""

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
"""Type of a public API argument for supplying a random number generator. Values of this
type should always be converted into :class:`RandomStateType` using the function
:func:`probnum.utils.as_random_state` before further internal processing."""
