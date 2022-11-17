"""Core of the compute backend."""

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


# Contractions
einsum = _core.einsum

# Reductions
all = _core.all
any = _core.any

# Just-in-Time Compilation
jit = _core.jit
jit_method = _core.jit_method


def vectorize(
    pyfunc,
    /,
    *,
    excluded: Optional[AbstractSet[Union[int, str]]] = None,
    signature: Optional[str] = None,
):
    return _core.vectorize(pyfunc, excluded=excluded, signature=signature)


__all__ = [
    # Contractions
    "einsum",
    # Reductions
    "all",
    "any",
    # Misc
    "vectorize",
    # Just-in-Time Compilation
    "jit",
    "jit_method",
]
