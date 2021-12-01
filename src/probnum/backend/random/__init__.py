from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _random
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _random
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _random

# Seed constructors
seed = _random.seed
split = _random.split

# Sample functions
standard_normal = _random.standard_normal
gamma = _random.gamma
