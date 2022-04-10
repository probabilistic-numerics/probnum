"""Functionality for random number generation implemented in the NumPy backend."""
from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from probnum import backend
from probnum.backend.typing import SeedType, ShapeType

RNGState = np.random.SeedSequence


def rng_state(seed: SeedType) -> RNGState:
    return np.random.SeedSequence(seed)


def split(rng_state: RNGState, num: int = 2) -> Sequence[RNGState]:
    return rng_state.spawn(num)


def _rng_from_rng_state(rng_state: RNGState) -> np.random.Generator:
    """Create a random generator instance initialized with the given state."""
    if not isinstance(rng_state, RNGState):
        raise TypeError(
            "`rng_state`s should always have type :class:`~backend.random.RNGState`."
        )

    return np.random.default_rng(rng_state)


def uniform(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: backend.Dtype = np.double,
    minval: np.ndarray = np.array(0.0),
    maxval: np.ndarray = np.array(1.0),
) -> np.ndarray:
    return np.asarray(
        (maxval - minval)
        * _rng_from_rng_state(rng_state).random(
            size=shape,
            dtype=dtype,
        )
        + minval
    )


def standard_normal(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: np.dtype = np.double,
) -> np.ndarray:
    return np.asarray(
        _rng_from_rng_state(rng_state).standard_normal(size=shape, dtype=dtype)
    )


def gamma(
    rng_state: RNGState,
    shape_param: np.ndarray,
    scale_param: np.ndarray = np.array(1.0),
    shape: ShapeType = (),
    dtype: np.dtype = np.double,
) -> np.ndarray:
    return np.asarray(
        _rng_from_rng_state(rng_state).standard_gamma(
            shape=shape_param, size=shape, dtype=dtype
        )
        * scale_param
    )


def uniform_so_group(
    rng_state: RNGState,
    n: int,
    shape: ShapeType = (),
    dtype: np.dtype = np.double,
) -> np.ndarray:
    if n == 1:
        return np.ones(shape + (1, 1), dtype=dtype)

    return np.asarray(
        _uniform_so_group_pushforward_fn(
            standard_normal(rng_state, shape=shape + (n - 1, n), dtype=dtype)
        )
    )


@functools.partial(np.vectorize, signature="(M,N)->(N,N)")
def _uniform_so_group_pushforward_fn(omega: np.ndarray) -> np.ndarray:
    n = omega.shape[1]

    assert omega.shape == (n - 1, n)

    X = np.triu(omega)

    # Copied and modified from https://github.com/scipy/scipy/blob/1c98aa98a55e2aaf2c15c16b47ee5e258bfcd170/scipy/stats/_multivariate.py#L3373-L3387
    H = np.eye(n, dtype=omega.dtype)
    D = np.empty((n,), dtype=omega.dtype)
    for idx in range(n - 1):
        x = X[idx, idx:]
        norm2 = np.dot(x, x)
        x0 = x[0].item()
        D[idx] = np.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[idx] * np.sqrt(norm2)
        x /= np.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
        # Householder transformation
        H[:, idx:] -= np.outer(np.dot(H[:, idx:], x), x)
    D[-1] = (-1) ** (n - 1) * D[:-1].prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H
