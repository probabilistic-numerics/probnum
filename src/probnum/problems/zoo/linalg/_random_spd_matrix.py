"""Random symmetric positive definite matrices."""

from typing import Optional, Sequence

import numpy as np
import scipy.stats

import probnum.utils as _utils
from probnum.typing import IntArgType, RandomStateArgType


def random_spd_matrix(
    dim: IntArgType,
    spectrum: Sequence = None,
    random_state: Optional[RandomStateArgType] = None,
) -> np.ndarray:
    """Random symmetric positive definite matrix.

    Constructs a random symmetric positive definite matrix from a given spectrum. An
    orthogonal matrix :math:`Q` with :math:`\\operatorname{det}(Q)` (a rotation) is
    sampled with respect to the Haar measure and the diagonal matrix
    containing the eigenvalues is rotated accordingly resulting in :math:`A=Q
    \\operatorname{diag}(\\lambda_1, \\dots, \\lambda_n)Q^\\top`. If no spectrum is
    provided, one is randomly drawn from a Gamma distribution.

    Parameters
    ----------
    dim
        Matrix dimension.
    spectrum
        Eigenvalues of the matrix.
    random_state
        Random state of the random variable. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    random_sparse_spd_matrix : Generate a random sparse symmetric positive definite matrix.

    Examples
    --------
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> mat = random_spd_matrix(dim=5, random_state=0)
    >>> mat
    array([[10.49868572, -0.80840778,  0.79781892,  1.9229059 ,  0.73413367],
           [-0.80840778, 15.79117417,  0.52641887, -1.8727916 , -0.9309482 ],
           [ 0.79781892,  0.52641887, 15.56457452,  1.26004438, -1.44969733],
           [ 1.9229059 , -1.8727916 ,  1.26004438,  8.59057287, -0.44955394],
           [ 0.73413367, -0.9309482 , -1.44969733, -0.44955394,  9.77198568]])

    Check for symmetry and positive definiteness.

    >>> np.all(mat == mat.T)
    True
    >>> np.linalg.eigvals(mat)
    array([ 6.93542496, 10.96494454,  9.34928449, 16.25401501, 16.71332395])
    """
    # Initialization
    random_state = _utils.as_random_state(random_state)

    if spectrum is None:
        # Create a custom ordered spectrum if none is given.
        spectrum_shape: float = 10.0
        spectrum_scale: float = 1.0
        spectrum_offset: float = 0.0

        spectrum = scipy.stats.gamma.rvs(
            spectrum_shape,
            loc=spectrum_offset,
            scale=spectrum_scale,
            size=dim,
            random_state=random_state,
        )
        spectrum = np.sort(spectrum)[::-1]

    else:
        spectrum = np.asarray(spectrum)
        if not np.all(spectrum > 0):
            raise ValueError(f"Eigenvalues must be positive, but are {spectrum}.")

    # Early exit for d=1 -- special_ortho_group does not like this case.
    if dim == 1:
        return spectrum.reshape((1, 1))

    # Draw orthogonal matrix with respect to the Haar measure
    orth_mat = scipy.stats.special_ortho_group.rvs(dim, random_state=random_state)
    spd_mat = orth_mat @ np.diag(spectrum) @ orth_mat.T

    # Symmetrize to avoid numerically not symmetric matrix
    # Since A commutes with itself (AA' = A'A = AA) the eigenvalues do not change.
    return 0.5 * (spd_mat + spd_mat.T)


def random_sparse_spd_matrix(
    dim: IntArgType,
    density: float,
    chol_entry_min: float = 0.1,
    chol_entry_max: float = 1.0,
    format="csr",  # pylint: disable="redefined-builtin"
    random_state: Optional[RandomStateArgType] = None,
) -> np.ndarray:
    r"""Random sparse symmetric positive definite matrix.

    Constructs a random sparse symmetric positive definite matrix for a given degree
    of sparsity. The matrix is constructed from its Cholesky factor :math:`L`. Its
    diagonal is set to one and all other nonzero entries of the lower triangle are
    sampled from a uniform distribution with bounds :code:`[chol_entry_min,
    chol_entry_max]`. The resulting sparse matrix is then given by :math:`A=LL^\top`.

    Parameters
    ----------
    dim
        Matrix dimension.
    density
        Degree of sparsity of the off-diagonal entries of the Cholesky factor.
        Between 0 and 1 where 1 represents a dense matrix.
    chol_entry_min
        Lower bound on the entries of the Cholesky factor.
    chol_entry_max
        Upper bound on the entries of the Cholesky factor.
    format
        Sparse matrix format.
    random_state
        Random state of the random variable. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive definite matrix.

    Examples
    --------
    >>> from probnum.problems.zoo.linalg import random_sparse_spd_matrix
    >>> sparsemat = random_sparse_spd_matrix(dim=5, density=0.1, random_state=42)
    >>> sparsemat.todense()
    matrix([[1.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 1.        , 0.        , 0.37381802, 0.        ],
            [0.        , 0.        , 1.        , 0.        , 0.        ],
            [0.        , 0.37381802, 0.        , 1.13973991, 0.        ],
            [0.        , 0.        , 0.        , 0.        , 1.        ]])
    """

    # Initialization
    random_state = _utils.as_random_state(random_state)
    if not 0 <= density <= 1:
        raise ValueError(f"Density must be between 0 and 1, but is {density}.")
    chol = scipy.sparse.eye(dim, format="csr")
    num_off_diag_cholesky = int(0.5 * dim * (dim - 1))
    num_nonzero_entries = int(num_off_diag_cholesky * density)

    if num_nonzero_entries > 0:

        sparse_matrix = scipy.sparse.rand(
            m=dim,
            n=dim,
            format="csr",
            density=density,
            random_state=random_state,
        )

        # Rescale entries
        sparse_matrix.data *= chol_entry_max - chol_entry_min
        sparse_matrix.data += chol_entry_min

        # Extract lower triangle
        chol += scipy.sparse.tril(A=sparse_matrix, k=-1, format=format)

    return (chol @ chol.T).asformat(format=format)
