import functools
from typing import Optional, Sequence

import numpy as np

from probnum import backend
from probnum.typing import DTypeLike, FloatLike, ShapeLike


def seed(seed: Optional[int]) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed

    return np.random.SeedSequence(seed)


def split(
    seed: np.random.SeedSequence, num: int = 2
) -> Sequence[np.random.SeedSequence]:
    return seed.spawn(num)


def uniform(
    seed: np.random.SeedSequence,
    shape: ShapeLike = (),
    dtype: DTypeLike = np.double,
    minval: FloatLike = 0.0,
    maxval: FloatLike = 1.0,
) -> np.ndarray:
    return _make_rng(seed).uniform(
        size=shape,
        dtype=dtype,
        low=backend.as_scalar(minval),
        high=backend.as_scalar(maxval),
    )


def standard_normal(
    seed: np.random.SeedSequence,
    shape: ShapeLike = (),
    dtype: DTypeLike = np.double,
) -> np.ndarray:
    return _make_rng(seed).standard_normal(size=shape, dtype=dtype)


def gamma(
    seed: np.random.SeedSequence,
    shape_param: FloatLike,
    scale_param: FloatLike = 1.0,
    shape: ShapeLike = (),
    dtype: DTypeLike = np.double,
) -> np.ndarray:
    return (
        _make_rng(seed).standard_gamma(shape=shape_param, size=shape, dtype=dtype)
        * scale_param
    )


def uniform_so_group(
    seed: np.random.SeedSequence,
    n: int,
    shape: ShapeLike = (),
    dtype: DTypeLike = np.double,
) -> np.ndarray:
    if n == 1:
        return np.ones(shape + (1, 1), dtype=dtype)

    return _uniform_so_group_pushforward_fn(
        standard_normal(seed, shape=shape + (n - 1, n), dtype=dtype)
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


def _make_rng(seed: np.random.SeedSequence) -> np.random.Generator:
    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError("`seed`s should always be created by")

    return np.random.default_rng(seed)
