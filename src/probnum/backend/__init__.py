from ._select import Backend, select_backend as _select_backend

# pylint: disable=undefined-all-variable
__all__ = [
    "ndarray",
    # DTypes
    "dtype",
    "asdtype",
    "bool",
    "int32",
    "int64",
    "single",
    "double",
    "csingle",
    "cdouble",
    "cast",
    "promote_types",
    "is_floating",
    "finfo",
    # Shape Arithmetic
    "reshape",
    "atleast_1d",
    "atleast_2d",
    "broadcast_arrays",
    "broadcast_shapes",
    "ndim",
    "swapaxes",
    # Constructors
    "array",
    "asarray",
    "diag",
    "eye",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "linspace",
    # Constants
    "pi",
    "inf",
    # Operations
    "sin",
    "exp",
    "log",
    "sqrt",
    "sum",
    "maximum",
]

BACKEND = _select_backend()

# isort: off

from ._dispatcher import Dispatcher

from ._core import *

from . import (
    autodiff,
    linalg,
    random,
    special,
)

# isort: on
