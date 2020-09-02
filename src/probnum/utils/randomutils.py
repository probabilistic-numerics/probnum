"""Utility functions related to random states and random number generators."""

from typing import Union

import numpy as np


def derive_random_seed(*args: Union[np.random.RandomState, np.random.Generator]) -> int:
    """Turn a random state or a generator into a random seed."""
    def _sample(rng: Union[np.random.RandomState, np.random.Generator]) -> int:
        if isinstance(rng, np.random.RandomState):
            return rng.randint(0, 2 ** 32, size=None, dtype=int)
        elif isinstance(rng, np.random.Generator):
            return rng.integers(0, 2 ** 32, size=None, dtype=int, endpoint=False)
        else:
            raise ValueError("Unsupported type of random number generator")

    seed = _sample(args[0])

    for i in range(1, len(args)):
        seed = seed ^ _sample(args[i])

    return seed
