"""Random linear systems."""
from typing import Optional, Union

import numpy as np
import scipy.sparse

import probnum.linops as linops
from probnum.problems import LinearSystem
from probnum.type import RandomStateArgType
from probnum.utils import as_random_state

# pylint: disable="invalid-name"


def random_linear_system(
    A: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    random_state: Optional[RandomStateArgType] = None,
) -> LinearSystem:
    """Generate a random linear system.

    Randomly creates a linear system with a solution and right hand side for the
    given matrix or linear operator.

    Parameters
    ----------
    A :
        System matrix for the random linear system.
    random_state
        Random state of the random variable. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """
    dims = A.shape
    rng = as_random_state(random_state)
    solution = rng.normal(size=(dims[1], 1))
    right_hand_side = A @ solution

    return LinearSystem(A=A, solution=solution, b=right_hand_side)
