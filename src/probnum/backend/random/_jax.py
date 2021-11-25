import secrets
from typing import Optional, Sequence

import jax


def seed(seed: Optional[int]) -> jax.numpy.ndarray:
    if seed is None:
        seed = secrets.randbits(128)

    if not isinstance(seed, int):
        return seed

    return jax.random.PRNGKey(seed)


def split(seed: jax.numpy.ndarray, num: int = 2) -> Sequence[jax.numpy.ndarray]:
    return jax.random.split(key=seed, num=num)


def standard_normal(seed: jax.numpy.ndarray, shape=(), dtype=jax.numpy.double):
    return jax.random.normal(key=seed, shape=shape, dtype=dtype)
