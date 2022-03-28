"""Core of the compute backend.

The interface provided by this module follows the Python array API standard
(https://data-apis.org/array-api/latest/index.html), which defines a common
common API for array and tensor Python libraries.
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
bool = _core.bool
int32 = _core.int32
int64 = _core.int64
single = _core.single
double = _core.double
csingle = _core.csingle
cdouble = _core.cdouble
cast = _core.cast
promote_types = _core.promote_types
result_type = _core.result_type
is_floating = _core.is_floating
is_floating_dtype = _core.is_floating_dtype
finfo = _core.finfo

# Array Shape
reshape = _core.reshape
atleast_1d = _core.atleast_1d
atleast_2d = _core.atleast_2d
broadcast_arrays = _core.broadcast_arrays
broadcast_shapes = _core.broadcast_shapes
broadcast_to = _core.broadcast_to
ndim = _core.ndim
squeeze = _core.squeeze
expand_dims = _core.expand_dims
swapaxes = _core.swapaxes

# Constructors
array = _core.array
diag = _core.diag
eye = _core.eye
full = _core.full
full_like = _core.full_like
ones = _core.ones
ones_like = _core.ones_like
zeros = _core.zeros
zeros_like = _core.zeros_like
linspace = _core.linspace
arange = _core.arange
meshgrid = _core.meshgrid

# Constants
inf = _core.inf
pi = _core.pi

# Element-wise Unary Operations
sign = _core.sign
abs = _core.abs
exp = _core.exp
isfinite = _core.isfinite
log = _core.log
sin = _core.sin
sqrt = _core.sqrt


# Element-wise Binary Operations
maximum = _core.maximum

# (Partial) Views
diagonal = _core.diagonal
moveaxis = _core.moveaxis
flip = _core.flip

# Contractions
einsum = _core.einsum

# Reductions
all = _core.all
any = _core.any
sum = _core.sum
max = _core.max

# Concatenation and Stacking
concatenate = _core.concatenate
stack = _core.stack
hstack = _core.hstack
vstack = _core.vstack
tile = _core.tile
kron = _core.kron

# Misc
to_numpy = _core.to_numpy

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method


def as_shape(x: ShapeLike, ndim: Optional[IntLike] = None) -> ShapeType:
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
    "bool",
    "int32",
    "int64",
    "single",
    "double",
    "csingle",
    "cdouble",
    "cast",
    "promote_types",
    "result_type",
    "is_floating",
    "is_floating_dtype",
    "finfo",
    # Array Shape
    "as_shape",
    "reshape",
    "atleast_1d",
    "atleast_2d",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "ndim",
    "squeeze",
    "expand_dims",
    "swapaxes",
    # Constructors
    "array",
    "diag",
    "eye",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "arange",
    "linspace",
    "meshgrid",
    # Constants
    "inf",
    "pi",
    # Element-wise Unary Operations
    "sign",
    "abs",
    "exp",
    "isfinite",
    "log",
    "sin",
    "sqrt",
    # Element-wise Binary Operations
    "maximum",
    # (Partial) Views
    "diagonal",
    "moveaxis",
    "flip",
    # Contractions
    "einsum",
    # Reductions
    "all",
    "any",
    "sum",
    "max",
    # Concatenation and Stacking
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "tile",
    "kron",
    # Misc
    "to_numpy",
    "vectorize",
    # Just-in-Time Compilation
    "jit",
    "jit_method",
]
