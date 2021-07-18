"""Generate random linear systems as test problems."""

from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.sparse

from probnum import linops, problems, randvars
from probnum.typing import LinearOperatorArgType


def random_linear_system(
    rng: np.random.Generator,
    matrix: Union[
        LinearOperatorArgType,
        Callable[
            [np.random.Generator, Optional[Any]],
            Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
        ],
    ],
    solution_rv: Optional[randvars.RandomVariable] = None,
    **kwargs,
) -> problems.LinearSystem:
    """Random linear system.

    Generate a random linear system from a (random) matrix. If ``matrix`` is a callable instead of a matrix or
    linear operator, the system matrix is sampled by passing the random generator instance ``rng``. The solution
    of the linear system is set to a realization from ``solution_rv``. If ``None`` the solution is drawn from a
    standard normal distribution with iid components.

    Parameters
    ----------
    rng
        Random number generator.
    matrix
        Matrix, linear operator or callable returning either for a given random number generator instance.
    solution_rv
        Random variable from which the solution of the linear system is sampled.
    kwargs
        Miscellaneous arguments passed onto the matrix-generating callable ``matrix``.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive-definite matrix.
    random_sparse_spd_matrix : Generate a random sparse symmetric positive-definite matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.problems.zoo.linalg import random_linear_system
    >>> rng = np.random.default_rng(42)

    Linear system with given system matrix.

    >>> import scipy.stats
    >>> unitary_matrix = scipy.stats.unitary_group.rvs(dim=5, random_state=rng)
    >>> linsys_unitary = random_linear_system(rng, unitary_matrix)
    >>> np.abs(np.linalg.det(linsys_unitary.A))
    1.0

    Linear system with random symmetric positive-definite matrix.

    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> linsys_spd = random_linear_system(rng, random_spd_matrix, dim=2)
    >>> linsys_spd
    LinearSystem(A=array([[ 9.62543582,  3.14955953],
           [ 3.14955953, 13.28720426]]), b=array([-2.7108139 ,  1.10779288]), solution=array([-0.33488503,  0.16275307]))


    Linear system with random sparse matrix.

    >>> import scipy.sparse
    >>> random_sparse_matrix = lambda rng,m,n: scipy.sparse.random(m=m, n=n, random_state=rng)
    >>> linsys_sparse = random_linear_system(rng, random_sparse_matrix, m=4, n=2)
    >>> isinstance(linsys_sparse.A, scipy.sparse.spmatrix)
    True
    """
    # Generate system matrix
    if isinstance(matrix, (np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator)):
        A = matrix
    else:
        A = matrix(rng=rng, **kwargs)

    # Sample solution
    if solution_rv is None:
        n = A.shape[1]
        x = randvars.Normal(mean=0.0, cov=1.0).sample(size=(n,), rng=rng)
    else:
        if A.shape[1] != solution_rv.shape[0]:
            raise ValueError(
                f"Shape of the system matrix: {A.shape} must match shape of the solution: {solution_rv.shape}."
            )
        x = solution_rv.sample(size=(), rng=rng)

    return problems.LinearSystem(A=A, b=A @ x, solution=x)
