from probnum import backend as _backend
from probnum.typing import ArrayType, DTypeArgType, ScalarArgType

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

# Shape Arithmetic
reshape = _core.reshape
atleast_1d = _core.atleast_1d
atleast_2d = _core.atleast_2d
broadcast_arrays = _core.broadcast_arrays
broadcast_shapes = _core.broadcast_shapes
ndim = _core.ndim

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
exp = _core.exp
isfinite = _core.isfinite
log = _core.log
sin = _core.sin
sqrt = _core.sqrt

# Element-wise Binary Operations
maximum = _core.maximum

# Reductions
all = _core.all
sum = _core.sum

# Misc
to_numpy = _core.to_numpy

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method


def as_scalar(x: ScalarArgType, dtype: DTypeArgType = None) -> ArrayType:
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
