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
    "array",
    "atleast_1d",
    "atleast_2d",
    "broadcast_arrays",
    "broadcast_shapes",
    "exp",
    "grad",
    "ndim",
    "ones_like",
    "sqrt",
    "sum",
    "zeros",
    "zeros_like",
]


if BACKEND is Backend.NUMPY:
    from ._numpy import *
elif BACKEND is Backend.JAX:
    from ._jax import *
elif BACKEND is Backend.PYTORCH:
    from ._pytorch import *
