from typing import Optional

from probnum import backend as _backend
from probnum.typing import DTypeLike, IntLike, ScalarLike, ShapeLike, ShapeType

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _core
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _core
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _core

# Assignments for common docstrings across backends
ndarray = _core.ndarray

# DType
dtype = _core.dtype
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
asarray = _core.asarray
diag = _core.diag
eye = _core.eye
full = _core.full
full_like = _core.full_like
ones = _core.ones
ones_like = _core.ones_like
zeros = _core.zeros
zeros_like = _core.zeros_like
linspace = _core.linspace

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

# Contractions
einsum = _core.einsum

# Reductions
all = _core.all
sum = _core.sum

# Concatenation and Stacking
concatenate = _core.concatenate
stack = _core.stack
hstack = _core.hstack
vstack = _core.vstack
tile = _core.tile

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
    except TypeError:
        # x is an iterable
        try:
            _ = iter(x)
        except TypeError as e:
            raise TypeError(
                f"The given shape {x} must be an integer or an iterable of integers."
            ) from e

        shape = tuple(int(item) for item in x)

    if ndim is not None:
        ndim = int(ndim)

        if len(shape) != ndim:
            raise TypeError(f"The given shape {shape} must have {ndim} dimensions.")

    return shape


def as_scalar(x: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    """Convert a scalar into a NumPy scalar.

    Parameters
    ----------
    x
        Scalar value.
    dtype
        Data type of the scalar.
    """

    if ndim(x) != 0:
        raise ValueError("The given input is not a scalar.")

    return asarray(x, dtype=dtype)[()]


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
    "asarray",
    "as_scalar",
    "diag",
    "eye",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "linspace",
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
    # Contractions
    "einsum",
    # Reductions
    "all",
    "sum",
    # Concatenation and Stacking
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "tile",
    # Misc
    "to_numpy",
    # Just-in-Time Compilation
    "jit",
    "jit_method",
]
