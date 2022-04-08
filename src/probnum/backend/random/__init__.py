"""Functionality for random number generation."""
from __future__ import annotations

from typing import Sequence

from probnum import backend as _backend
from probnum.backend.typing import Seed

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl

RNGState = _impl.RNGState
"""State of the random number generator."""

# RNG state constructors
def rng_state(seed: Seed) -> RNGState:
    """Create a state of a random number generator from a seed.

    Parameters
    ----------
    seed
        Seed for the random number generator.

    Returns
    -------
    rng_state
        State of a random number generator.
    """
    return _impl.rng_state(seed=seed)


def split(rng_state: RNGState, num: int = 2) -> Sequence[RNGState]:
    """Split the RNG state into multiple.

    Parameters
    ----------
    rng_state
        Base RNG state.
    num
        Number of RNG states to split into.

    Returns
    -------
    rng_states
        Sequence of RNG states.
    """
    return _impl.split(rng_state=rng_state, num=num)


# Sample functions
uniform = _impl.uniform
standard_normal = _impl.standard_normal
gamma = _impl.gamma
uniform_so_group = _impl.uniform_so_group
