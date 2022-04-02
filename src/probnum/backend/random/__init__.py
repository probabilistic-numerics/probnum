from __future__ import annotations

from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl

_SeedType = _impl.SeedType

# Seed constructors
seed = _impl.seed
split = _impl.split

# Sample functions
uniform = _impl.uniform
standard_normal = _impl.standard_normal
gamma = _impl.gamma
uniform_so_group = _impl.uniform_so_group
