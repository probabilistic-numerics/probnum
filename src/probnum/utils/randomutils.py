from typing import Union
import copy

import numpy as np


def derive_random_seed(*rngs: Union[np.random.RandomState, np.random.Generator]) -> int:
    """
    Derive a new random seed from a set of random number generators.

    Draws a single integer sample from each generator and combines them via a "bitwise
    exclusive or" (XOR) operation into a common seed.

    Parameters
    ----------
    rngs :
        Random number generators.

    Returns
    -------
    seed :
        Random seed derived from the given generators.
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
