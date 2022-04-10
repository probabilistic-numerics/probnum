"""Functionality for random number generation."""
from __future__ import annotations

from typing import Sequence

from probnum import backend
from probnum.backend.typing import FloatLike, SeedType, ShapeLike

if backend.BACKEND is backend.Backend.NUMPY:
    from . import _numpy as _impl
elif backend.BACKEND is backend.Backend.JAX:
    from . import _jax as _impl
elif backend.BACKEND is backend.Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "RNGState",
    "rng_state",
    "split",
    "gamma",
    "standard_normal",
    "uniform",
    "uniform_so_group",
]


RNGState = _impl.RNGState
"""State of the random number generator."""


def rng_state(seed: SeedType) -> RNGState:
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
    """Split the random number generator state into multiple.

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


def uniform(
    rng_state: RNGState,
    shape: ShapeLike = (),
    dtype: backend.Dtype = backend.float64,
    minval: FloatLike = 0.0,
    maxval: FloatLike = 1.0,
) -> backend.Array:
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval ``[minval, maxval)``
    (includes ``minval``, but excludes ``maxval``). In other words, any value within the
    given interval is equally likely to be drawn by :meth:`uniform`.

    Parameters
    ----------
    rng_state
        Random number generator state.
    shape
        Sample shape.
    dtype
        Sample data type.
    minval
        Lower bound of the sampled values. All values generated will be greater than
        or equal to ``minval``.
    maxval
        Upper bound of the sampled values. All values generated will be strictly smaller
        than ``maxval``.

    Returns
    -------
    samples
        Samples from the uniform distribution.
    """
    return _impl.uniform(
        rng_state=rng_state,
        shape=backend.asshape(shape),
        dtype=dtype,
        minval=backend.asscalar(minval, dtype=dtype),
        maxval=backend.asscalar(maxval, dtype=dtype),
    )


def standard_normal(
    rng_state: RNGState,
    shape: ShapeLike = (),
    dtype: backend.Dtype = backend.float64,
) -> backend.Array:
    """Draw samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    rng_state
        Random number generator state.
    shape
        Sample shape.
    dtype
        Sample data type.

    Returns
    -------
    samples
        Samples from the standard normal distribution.
    """
    return _impl.standard_normal(
        rng_state=rng_state,
        shape=backend.asshape(shape),
        dtype=dtype,
    )


def gamma(
    rng_state: RNGState,
    shape_param: FloatLike,
    scale_param: FloatLike = 1.0,
    shape: ShapeLike = (),
    dtype: backend.Dtype = backend.float64,
) -> backend.Array:
    """Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters, shape
    (sometimes designated “k”) and scale (sometimes designated “theta”), where both
    parameters are > 0.

    Parameters
    ----------
    rng_state
        Random number generator state.
    shape_param
        Shape parameter of the Gamma distribution.
    scale_param
        Scale parameter of the Gamma distribution.
    shape
        Sample shape.
    dtype
        Sample data type.

    Returns
    -------
    samples
        Samples from the Gamma distribution.
    """
    return _impl.gamma(
        rng_state=rng_state,
        shape_param=backend.asscalar(shape_param),
        scale_param=backend.asscalar(scale_param),
        shape=backend.asshape(shape),
        dtype=dtype,
    )


def uniform_so_group(
    rng_state: RNGState,
    n: int,
    shape: ShapeLike = (),
    dtype: backend.Dtype = backend.float64,
) -> backend.Array:
    """Draw samples from the Haar distribution, i.e. from the uniform distribution on
    SO(n).

    The generated samples are randomly drawn orthogonal matrices with determinant 1,
    i.e. elements of the special orthogonal group SO(n).

    Parameters
    ----------
    rng_state
        Random number generator state.
    n
        Matrix dimension.
    shape
        Sample shape.
    dtype
        Sample data type.

    Returns
    -------
    samples
        Samples from the Haar distribution.
    """
    return _impl.uniform_so_group(
        rng_state=rng_state,
        n=n,
        shape=backend.asshape(shape),
        dtype=dtype,
    )
