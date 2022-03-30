from typing import Callable

from .. import BACKEND, Backend
from ..typing import Scalar

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["cond"]

def cond(pred: Scalar, true_fn: Callable, false_fn: Callable, *operands):
    return _impl.cond(pred, true_fn, false_fn, *operands)
