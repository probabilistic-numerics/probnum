from typing import Optional, Sequence

import numpy as np


def seed(seed: Optional[int]) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed

    return np.random.SeedSequence(seed)


def split(
    seed: np.random.SeedSequence, num: int = 2
) -> Sequence[np.random.SeedSequence]:
    return seed.spawn(num)


def standard_normal(seed: np.random.SeedSequence, shape=(), dtype=np.double):
    return _make_rng(seed).standard_normal(size=shape, dtype=dtype)


def gamma(seed: np.random.SeedSequence, a, scale=1.0, shape=(), dtype=np.double):
    return _make_rng(seed).gamma(shape=a, scale=scale, size=shape, dtype=dtype)


def _make_rng(seed: np.random.SeedSequence) -> np.random.Generator:
    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError("`seed`s should always be created by")

    return np.random.default_rng(seed)
