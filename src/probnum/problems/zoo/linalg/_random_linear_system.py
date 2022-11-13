"""Generate random linear systems as test problems."""
from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.sparse

from probnum import backend, linops, problems, randvars
from probnum.backend.random import RNGState
from probnum.typing import LinearOperatorLike


def random_linear_system(
    rng_state: RNGState,
    matrix: Union[
        LinearOperatorLike,
        Callable[
            [RNGState, Optional[Any]],
            Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
        ],
    ],
    solution_rv: Optional[randvars.RandomVariable] = None,
    **kwargs,
) -> problems.LinearSystem:
    """Random linear system.

    Generate a random linear system from a (random) matrix. If ``matrix`` is a callable
    instead of a matrix or linear operator, the system matrix is sampled by passing the
    random generator state ``rng_state``. The solution of the linear system is set to a
    realization from ``solution_rv``. If ``None`` the solution is drawn from a
    standard normal distribution with iid components.

    Parameters
    ----------
    rng_state
        State of the random number generator.
    matrix
        Matrix, linear operator or callable returning either for a given RNG state.
    solution_rv
        Random variable from which the solution of the linear system is sampled.
    kwargs
        Miscellaneous arguments passed onto the matrix-generating callable ``matrix``.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive-definite matrix.
    random_sparse_spd_matrix : Generate a random sparse symmetric
        positive-definite matrix.

    Examples
    --------
    >>> from probnum import backend
    >>> from probnum.problems.zoo.linalg import random_linear_system
    >>> rng_state = backend.random.rng_state(42)

    Linear system with given system matrix.

    >>> unitary_matrix = backend.random.uniform_so_group(rng_state, n=5)
    >>> linsys_unitary = random_linear_system(rng_state, unitary_matrix)
    >>> np.abs(np.linalg.det(linsys_unitary.A))
    1.0

    Linear system with random symmetric positive-definite matrix.

    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> linsys_spd = random_linear_system(rng_state, random_spd_matrix, shape=(2,2))
    >>> linsys_spd
    LinearSystem(A=array([[10.61706238, -0.78723358],
           [-0.78723358, 10.06458988]]), b=array([3.96470544, 5.76555243]),
           solution=array([0.41832997, 0.60557617]))


    Linear system with random sparse matrix.

    >>> import scipy.sparse
    >>> random_sparse_matrix = lambda rng_state, m, n: scipy.sparse.random(
    ...     m=m,
    ...     n=n,
    ...     random_state=rng_state,
    ... )
    >>> linsys_sparse = random_linear_system(rng_state, random_sparse_matrix, m=4, n=2)
    >>> isinstance(linsys_sparse.A, scipy.sparse.spmatrix)
    True
    """

    # Generate system matrix
    if isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator)):
        A = matrix
    else:
        rng_state, matrix_rng_state = backend.random.split(rng_state, num=2)

        A = matrix(rng_state=matrix_rng_state, **kwargs)

    # Sample solution
    if solution_rv is None:
        n = A.shape[1]
        x = backend.random.standard_normal(rng_state, shape=(n,))
    else:
        if A.shape[1] != solution_rv.shape[0]:
            raise ValueError(
                f"Shape of the system matrix: {A.shape} must match shape \
                of the solution: {solution_rv.shape}."
            )
        x = solution_rv.sample(rng_state=rng_state, sample_shape=())

    return problems.LinearSystem(A=A, b=A @ x, solution=x)
