"""Core of the compute backend.

The interface provided by this module follows the Python array API standard
(https://data-apis.org/array-api/latest/index.html), which defines a common
API for array and tensor Python libraries.
"""

from typing import AbstractSet, Optional, Union

from probnum import backend as _backend
from probnum.backend.typing import IntLike, ShapeLike, ShapeType

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _core
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _core
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _core

# Assignments for common docstrings across backends

# DType
asdtype = _core.asdtype
cast = _core.cast
promote_types = _core.promote_types
result_type = _core.result_type
is_floating = _core.is_floating
is_floating_dtype = _core.is_floating_dtype
finfo = _core.finfo

# Array Shape
atleast_1d = _core.atleast_1d
atleast_2d = _core.atleast_2d
broadcast_shapes = _core.broadcast_shapes
ndim = _core.ndim

# Constructors
diag = _core.diag

# Element-wise Binary Operations
maximum = _core.maximum
minimum = _core.minimum

# (Partial) Views
diagonal = _core.diagonal
moveaxis = _core.moveaxis

# Contractions
einsum = _core.einsum

# Reductions
all = _core.all
any = _core.any

# Concatenation and Stacking
tile = _core.tile
kron = _core.kron

# Misc
to_numpy = _core.to_numpy

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method


def asshape(x: ShapeLike, ndim: Optional[IntLike] = None) -> ShapeType:
    """Convert a shape representation into a shape defined as a tuple of ints.

    Parameters
    ----------
    x
        Shape representation.
    """

    try:
        # x is an `IntLike`
        shape = (int(x),)
    except (TypeError, ValueError):
        # x is an iterable
        try:
            shape = tuple(int(item) for item in x)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"The given shape {x} must be an integer or an iterable of integers."
            ) from err

    if ndim is not None:
        ndim = int(ndim)

        if len(shape) != ndim:
            raise TypeError(f"The given shape {shape} must have {ndim} dimensions.")

    return shape


def vectorize(
    pyfunc,
    /,
    *,
    excluded: Optional[AbstractSet[Union[int, str]]] = None,
    signature: Optional[str] = None,
):
    return _core.vectorize(pyfunc, excluded=excluded, signature=signature)


__all__ = [
    # DTypes
    "asdtype",
    "cast",
    "promote_types",
    "result_type",
    "is_floating",
    "is_floating_dtype",
    "finfo",
    # Array Shape
    "asshape",
    "atleast_1d",
    "atleast_2d",
    "broadcast_shapes",
    "ndim",
    # Constructors
    "diag",
    # Element-wise Binary Operations
    "maximum",
    "minimum",
    # (Partial) Views
    "diagonal",
    "moveaxis",
    # Contractions
    "einsum",
    # Reductions
    "all",
    "any",
    # Concatenation and Stacking
    "tile",
    "kron",
    # Misc
    "to_numpy",
    "vectorize",
    # Just-in-Time Compilation
    "jit",
    "jit_method",
]
