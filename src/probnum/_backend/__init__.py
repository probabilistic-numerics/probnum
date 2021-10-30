import enum


class Backend(enum.Enum):
    JAX = "jax"
    PYTORCH = "pytorch"
    NUMPY = "numpy"


# isort: off
from ._select import select_backend as _select_backend

# isort: on

BACKEND = _select_backend()


__all__ = [
    "ndarray",
    # DTypes
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
    # Shape Arithmetic
    "atleast_1d",
    "atleast_2d",
    "broadcast_arrays",
    "broadcast_shapes",
    "ndim",
    # Constructors
    "array",
    "diag",
    "eye",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "linspace",
    # Constants
    "pi",
    # Operations
    "exp",
    "log",
    "sqrt",
    "sum",
    # Automatic Differentiation
    "grad",
]

# isort: off

from ._dispatcher import BackendDispatcher
from . import linalg

# isort: on


if BACKEND is Backend.NUMPY:
    from ._numpy import *
elif BACKEND is Backend.JAX:
    from ._jax import *
elif BACKEND is Backend.PYTORCH:
    from ._pytorch import *
