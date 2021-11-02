from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _autodiff
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _autodiff
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _autodiff

grad = _autodiff.grad
