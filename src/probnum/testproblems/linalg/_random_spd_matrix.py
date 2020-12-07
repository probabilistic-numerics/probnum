"""Random symmetric positive definite matrices."""

from typing import Optional, Sequence

import numpy as np
import scipy.stats

from probnum.type import IntArgType, RandomStateArgType


def random_spd_matrix(
    dim: IntArgType,
    spectrum: Sequence = None,
    random_state: Optional[RandomStateArgType] = None,
) -> np.ndarray:
    """Random symmetric positive definite matrix.

    Constructs a random symmetric positive definite matrix from a given spectrum. An
    orthogonal matrix :math:`Q` with :math:`\\operatorname{det}(Q)` (a rotation) is
    sampled with respect to the Haar measure and the spectrum is rotated accordingly
    resulting in :math:`A=Q \\operatorname{diag}(\\lambda_1, \\dots, \\lambda_n)
    Q^\\top`. If no spectrum is provided, one is randomly drawn from a Gamma
    distribution.

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
    """

    if random_state is None:
        random_state = np.random.default_rng()

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
    random_state: Optional[RandomStateArgType] = None,
) -> np.ndarray:
    """Random sparse symmetric positive definite matrix.

    Constructs a random sparse symmetric positive definite matrix for a given degree
    of sparsity. The matrix is constructed from its Cholesky factor :math:`L`. Its
    diagonal is set to one and all other entries of the lower triangle are sampled
    from a uniform distribution with bounds :code:`[chol_entry_min, chol_entry_max]`.
    The resulting sparse matrix is then given by :math:`A=LL^\\top`.

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
    random_state
        Random state of the random variable. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    random_spd_matrix : Generate a random symmetric positive definite matrix.
    """
    if not 0 <= density <= 1:
        raise ValueError(f"Density must be between 0 and 1, but is {density}.")

    # Initialization
    chol = np.eye(dim)
    num_off_diag_cholesky = int(0.5 * dim * (dim - 1))
    num_nonzero_entries = int(num_off_diag_cholesky * density)

    if random_state is None:
        random_state = np.random.default_rng()

    if num_nonzero_entries > 0:
        # Draw entries of lower triangle (below diagonal) according to sparsity level
        entry_ids = np.mask_indices(n=dim, mask_func=np.tril, k=-1)
        _idx = random_state.choice(
            a=num_off_diag_cholesky, size=num_nonzero_entries, replace=False
        )
        nonzero_entry_ids = (entry_ids[0][_idx], entry_ids[1][_idx])

        # Fill Cholesky factor
        chol[nonzero_entry_ids] = random_state.uniform(
            low=chol_entry_min, high=chol_entry_max, size=num_nonzero_entries
        )

    return chol @ chol.T
