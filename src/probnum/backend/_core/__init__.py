from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _core
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _core
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _core

# Assignments for common docstrings across backends
ndarray = _core.ndarray

# DType
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
finfo = _core.finfo

# Shape Arithmetic
atleast_1d = _core.atleast_1d
atleast_2d = _core.atleast_2d
broadcast_arrays = _core.broadcast_arrays
broadcast_shapes = _core.broadcast_shapes
ndim = _core.ndim

# Constructors
array = _core.array
asarray = _core.asarray
diag = _core.diag
eye = _core.eye
ones = _core.ones
ones_like = _core.ones_like
zeros = _core.zeros
zeros_like = _core.zeros_like
linspace = _core.linspace

# Constants
pi = _core.pi
inf = _core.inf

# Operations
sin = _core.sin
exp = _core.exp
log = _core.log
sqrt = _core.sqrt
sum = _core.sum
maximum = _core.maximum

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method
