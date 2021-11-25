"""Orthogonalization of vectors."""
import numpy as np


def gram_schmidt(
    vector: np.ndarray, orthogonal_basis: np.ndarray, is_orthonormal=False
) -> np.ndarray:
    """Orthogonalize a vector with respect to an orthogonal basis.

    Parameters
    ----------
    vector
        Vector to orthogonalize.
    orthogonal_basis
        Orthogonal basis.
    is_orthonormal
        Whether the basis is assumed to be orthonormal. If `False`, vectors are normalized.
    """
    if not is_orthonormal:
        norms = np.sqrt(np.sum(orthogonal_basis * orthogonal_basis, axis=0))
        orthonormal_basis = orthogonal_basis / norms
        return vector - orthonormal_basis @ orthonormal_basis.T @ vector

    return vector - orthogonal_basis @ orthogonal_basis.T @ vector


def double_gram_schmidt(
    vector: np.ndarray, orthogonal_basis: np.ndarray, is_orthonormal=False
) -> np.ndarray:
    """Orthogonalize a vector with respect to an orthogonal basis.

    This performs the Gram-Schmidt orthogonalization process twice. This is generally more
    stable than just reorthogonalizing once. [1]_ [2]_

    Parameters
    ----------
    vector
        Vector to orthogonalize.
    orthonormal_basis
        Orthonormal basis.
    is_orthonormal
        Whether the basis is assumed to be orthonormal. If `False` vectors are normalized.

    References
    ----------
    .. [1] L. Giraud, J. Langou, M. Rozloznik, and J. van den Eshof, Rounding error analysis of the
           classical Gram-Schmidt orthogonalization process, Numer. Math., 101 (2005), pp. 87–100
    .. [2] L. Giraud, J. Langou, and M. Rozloznik, The loss of orthogonality in the Gram-Schmidt
           orthogonalization process, Comput. Math. Appl., 50 (2005)
    """
    if not is_orthonormal:
        norms = np.sqrt(np.sum(orthogonal_basis * orthogonal_basis, axis=0))
        orthogonal_basis = orthogonal_basis / norms
        is_orthonormal = True

    return gram_schmidt(
        gram_schmidt(
            vector=vector,
            orthogonal_basis=orthogonal_basis,
            is_orthonormal=is_orthonormal,
        ),
        orthogonal_basis=orthogonal_basis,
        is_orthonormal=is_orthonormal,
    )
