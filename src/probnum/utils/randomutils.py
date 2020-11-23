"""Utility functions for random objects."""

import copy
from typing import Union

import numpy as np


def derive_random_seed(*rngs: Union[np.random.RandomState, np.random.Generator]) -> int:
    """Derive a new random seed from a set of random number generator(s).

    Draws a single integer sample from each generator and combines them via a "bitwise
    exclusive or" (XOR) operation into a common seed. Note that in other frameworks this
    function is also used to combine hashes.

    Parameters
    ----------
    rngs
        Random number generators.
    """

    def _sample(rng: Union[np.random.RandomState, np.random.Generator]) -> int:
        if isinstance(rng, np.random.RandomState):
            return copy.copy(rng).randint(0, 2 ** 32, size=None, dtype=int)
        elif isinstance(rng, np.random.Generator):
            return copy.copy(rng).integers(
                0, 2 ** 32, size=None, dtype=int, endpoint=False
            )
        else:
            raise ValueError("Unsupported type of random number generator")

    seed = _sample(rngs[0])

    for i in range(1, len(rngs)):
        seed = seed ^ _sample(rngs[i])

    return seed
