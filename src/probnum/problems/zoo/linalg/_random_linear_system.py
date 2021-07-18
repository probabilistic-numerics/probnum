"""Generate random linear systems as test problems."""

from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.sparse

from probnum import linops, problems, randvars
from probnum.problems.zoo.linalg._random_spd_matrix import random_spd_matrix


def random_linear_system(
    rng: np.random.Generator,
    random_matrix: Optional[
        Callable[
            [np.random.Generator, Optional[Any]],
            Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
        ]
    ] = random_spd_matrix,
    solution: Optional[randvars.RandomVariable] = None,
    **kwargs,
) -> problems.LinearSystem:
    """Random linear system.

    Generate a random linear system by sampling a system matrix with the given properties.
    The solution is chosen to be a realization from the provided random variable. If None
    the solution is drawn from a standard normal distribution with iid components.

    Parameters
    ----------
    rng
        Random number generator.
    random_matrix
        Callable returning a matrix or linear operator if given a random number generator instance.
    kwargs
        Miscellaneous arguments passed onto the matrix-generating function ``random_matrix``.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive-definite matrix.
    random_sparse_spd_matrix : Generate a random sparse symmetric positive-definite matrix.

    Examples
    --------
    #TODO
    """
    # Generate system matrix
    A = random_matrix(rng=rng, **kwargs)

    # Sample solution
    if solution is None:
        n = A.shape[1]
        x = randvars.Normal(mean=0, cov=1).sample(size=(n,))
    else:
        if A.shape[1] != solution.shape[0]:
            raise ValueError(
                f"Shape of the system matrix: {A.shape} must match shape of the solution: {solution.shape}."
            )
        x = solution.sample(size=())

    return problems.LinearSystem(A=A, b=A @ x, solution=x)
