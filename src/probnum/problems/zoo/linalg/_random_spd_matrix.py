"""Random symmetric positive definite matrices."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.stats

from probnum import backend
from probnum.backend.random import RNGState
from probnum.backend.typing import ShapeLike


def random_spd_matrix(
    rng_state: RNGState,
    shape: ShapeLike,
    spectrum: Sequence = None,
) -> backend.Array:
    r"""Random symmetric positive definite matrix.

    Constructs a random symmetric positive definite matrix from a given spectrum. An
    orthogonal matrix :math:`Q` with :math:`\operatorname{det}(Q)` (a rotation) is
    sampled with respect to the Haar measure and the diagonal matrix
    containing the eigenvalues is rotated accordingly resulting in :math:`A=Q
    \operatorname{diag}(\lambda_1, \dots, \lambda_n)Q^\top`. If no spectrum is
    provided, one is randomly drawn from a Gamma distribution.

    Parameters
    ----------
    rng_state
        State of the random number generator.
    shape
        Shape of the resulting matrix.
    spectrum
        Eigenvalues of the matrix.

    See Also
    --------
    random_sparse_spd_matrix : Generate a random
        sparse symmetric positive definite matrix.

    Examples
    --------
    >>> from probnum import backend
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> rng_state = backend.random.rng_state(1)
    >>> mat = random_spd_matrix(rng_state, shape=(5, 5))
    >>> mat
    array([[ 8.93286789,  0.46676405, -2.10171474,  1.44158222, -0.32869563],
           [ 0.46676405,  7.63938418, -2.45135608,  2.03734623,  0.8095071 ],
           [-2.10171474, -2.45135608,  8.52968389, -0.11968995,  1.74237472],
           [ 1.44158222,  2.03734623, -0.11968995,  8.58417432, -1.61553113],
           [-0.32869563,  0.8095071 ,  1.74237472, -1.61553113,  8.1054103 ]])

    Check for symmetry and positive definiteness.

    >>> backend.all(mat == mat.T)
    True
    >>> backend.linalg.eigvalsh(mat)
    array([ 3.51041217,  7.80937731,  8.49510526,  8.76024149, 13.21638435])
    """
    shape = backend.asshape(shape)

    if not shape == () and shape[0] != shape[1]:
        raise ValueError(f"Shape must represent a square matrix, but is {shape}.")

    gamma_rng_state, so_rng_state = backend.random.split(rng_state, num=2)

    # Initialization
    if spectrum is None:
        spectrum = backend.random.gamma(
            gamma_rng_state,
            shape_param=10.0,
            scale_param=1.0,
            shape=shape[:1],
        )
    else:
        spectrum = backend.asarray(spectrum)

        if spectrum.shape != shape[:1]:
            raise ValueError(f"Size of the spectrum and shape are not compatible.")

        if not backend.all(spectrum > 0):
            raise ValueError(f"Eigenvalues must be positive, but are {spectrum}.")

    if len(shape) == 0:
        return spectrum

    if shape[0] == 1:
        return spectrum.reshape((1, 1))

    # Draw orthogonal matrix with respect to the Haar measure
    orth_mat = backend.random.uniform_so_group(so_rng_state, n=shape[0])
    spd_mat = (orth_mat * spectrum[None, :]) @ orth_mat.T

    # Symmetrize to avoid numerically not symmetric matrix
    # Since A commutes with itself (AA' = A'A = AA) the eigenvalues do not change.
    return 0.5 * (spd_mat + spd_mat.T)


def random_sparse_spd_matrix(
    rng_state: RNGState,
    shape: ShapeLike,
    density: float,
    chol_entry_min: float = 0.1,
    chol_entry_max: float = 1.0,
    format="csr",  # pylint: disable="redefined-builtin"
) -> scipy.sparse.spmatrix:
    r"""Random sparse symmetric positive definite matrix.

    Constructs a random sparse symmetric positive definite matrix for a given degree
    of sparsity. The matrix is constructed from its Cholesky factor :math:`L`. Its
    diagonal is set to one and all other nonzero entries of the lower triangle are
    sampled from a uniform distribution with bounds :code:`[chol_entry_min,
    chol_entry_max]`. The resulting sparse matrix is then given by :math:`A=LL^\top`.

    Parameters
    ----------
    rng_state
        State of the random number generator.
    shape
        Shape of the resulting matrix.
    density
        Degree of sparsity of the off-diagonal entries of the Cholesky factor.
        Between 0 and 1 where 1 represents a dense matrix.
    chol_entry_min
        Lower bound on the entries of the Cholesky factor.
    chol_entry_max
        Upper bound on the entries of the Cholesky factor.
    format
        Sparse matrix format.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive definite matrix.

    Examples
    --------
    >>> from probnum import backend
    >>> from probnum.problems.zoo.linalg import random_sparse_spd_matrix
    >>> rng_state = backend.random.rng_state(42)
    >>> sparsemat = random_sparse_spd_matrix(rng_state, shape=(5,5), density=0.1)
    >>> sparsemat
    <5x5 sparse matrix of type '<class 'numpy.float64'>'
        with 9 stored elements in Compressed Sparse Row format>
    >>> sparsemat.todense()
    matrix([[1.        , 0.        , 0.87273813, 0.        , 0.        ],
            [0.        , 1.        , 0.        , 0.        , 0.        ],
            [0.87273813, 0.        , 1.76167184, 0.        , 0.        ],
            [0.        , 0.        , 0.        , 1.        , 0.72763123],
            [0.        , 0.        , 0.        , 0.72763123, 1.5294472 ]])
    """

    # Initialization
    if not 0 <= density <= 1:
        raise ValueError(f"Density must be between 0 and 1, but is {density}.")
    if not shape == () and shape[0] != shape[1]:
        raise ValueError(f"Shape must represent a square matrix, but is {shape}.")

    chol = scipy.sparse.eye(shape[0], format="csr")
    num_off_diag_cholesky = int(0.5 * shape[0] * (shape[0] - 1))
    num_nonzero_entries = int(num_off_diag_cholesky * density)

    if num_nonzero_entries > 0:
        sparse_matrix = scipy.sparse.rand(
            m=shape[0],
            n=shape[0],
            format="csr",
            density=density,
            random_state=np.random.default_rng(rng_state),
        )

        # Rescale entries
        sparse_matrix.data *= chol_entry_max - chol_entry_min
        sparse_matrix.data += chol_entry_min

        # Extract lower triangle
        chol += scipy.sparse.tril(A=sparse_matrix, k=-1, format=format)

    return (chol @ chol.T).asformat(format=format)
