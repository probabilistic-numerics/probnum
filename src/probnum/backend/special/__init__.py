from .. import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from ._numpy import *
elif BACKEND is Backend.JAX:
    from ._jax import *
elif BACKEND is Backend.TORCH:
    from ._torch import *


__all__ = [
    "gamma",
    "kv",
    "ndtr",
    "ndtri",
]
