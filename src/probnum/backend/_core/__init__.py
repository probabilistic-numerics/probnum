"""Core of the compute backend."""


from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _core
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _core
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _core

# Logical functions
all = _core.all
any = _core.any

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method


__all__ = [
    # Reductions
    "all",
    "any",
    # Just-in-Time Compilation
    "jit",
    "jit_method",
]
