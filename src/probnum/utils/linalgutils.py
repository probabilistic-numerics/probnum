""" Linear algebra utilities """

import numpy as np
import scipy.linalg


def cholesky_rank_1_update(
    L: np.ndarray, v: np.ndarray, overwrite_L: bool = False, overwrite_v: bool = False,
) -> np.ndarray:
    r""" Compute the Cholesky factorization of a symmetric rank-1 update to a symmetric
    positive definite matrix with given Cholesky factorization in a fast and stable
    manner.

    Specifically, given the lower triangular Cholesky factor :math:`L` of
    :math:`A = L L^T \in \mathbb{R}^{N \times N}`, this function computes the lower
    triangular Cholesky factor :math:`L'` of :math:`A' = L' L'^T`, where
    :math:`A' = A + v v^T` for some vector :math:`v \in \mathbb{R}^n`.

    We implement the method in section 2 of [1]_. This algorithm computes the Cholesky
    decomposition of :math:`A'` from :math:`L` in :math:`\mathbb{O}(N^2)` time, which
    is faster than the :math:`\mathbb{O}(N^3)` time complexity of naively applying a
    Cholesky algorithm to :math:`A'` directly.

    Parameters
    ----------
    L : shape=(N, N), dtype=np.float64
        Lower triangular Cholesky factor of :math:`A`.
        The algorithm is most efficient if this array is given in column-major layout,
        a.k.a. Fortran-contiguous or f-contiguous memory order.
        Passing anything else than a valid lower triangular Cholesky factor in the lower
        triangular part of `L`, e.g. a matrix with zeros on the diagonal, will lead to
        undefined behavior.
        The entries in the strict upper triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix. This behavior is useful when using the Cholesky factors returned
        by :func:`scipy.linalg.cho_factor` which contain arbitrary values on the
        irrelevant triangular part of the matrix.
    v : shape=(N,), dtype=np.float64
        The vector :math:`v` defining the symmetric rank-1 matrix :math:`v v^T`.
    overwrite_L : optional
        If set to :code:`True`, the function will overwrite the array :code:`L` with the
        upper Cholesky factor :math:`L'` of :math:`A'`, i.e. the result is computed
        in-place.
        Passing `False` here ensures that the array :code:`L` is not modified.
    overwrite_v : optional
        If set to `True`, the function will reuse the array :code:`v` as an internal
        computation buffer, which will modify :code:`v`.
        Passing `False` here ensures that the array :code:`v` is not modified.
        In this case, an additional array of shape :code:`(N,)` and dtype
        :class:`np.float64` must be allocated.

    Returns
    -------
    L_prime : shape=(N, N), dtype=np.float64
        Lower triangular Cholesky factor :math:`L'` of :math:`A + v v^T = L' L'^T`.
        The diagonal entries of this matrix are guaranteed to be positive.
        The strict upper triangular part of this matrix will contain arbitrary values.
        The matrix will inherit the memory order from :code:`L`.

    Raises
    ------
    ValueError
        If :code:`L` does not have shape :code:`(N, N)` for some :code:`N`.
    TypeError
        If :code:`L` does not have dtype :class:`np.float64`.
    ValueError
        If :code:`v` does not have shape :code:`(N,)`, while :code:`L` has shape
        :code:`(N, N)`.
    TypeError
        If :code:`v` does not have dtype :class:`np.float64`.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
    """

    # Check L
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(
            f"The given Cholesky factor `L_T` is not a square matrix (given shape: "
            f"{L.shape})."
        )

    if L.dtype != np.float64:
        raise TypeError(
            f"The given Cholesky factor `L_T` does not have dtype `np.float64` (given "
            f"dtype: {L.dtype.name})"
        )

    # Check v
    if v.ndim != 1 or v.shape[0] != L.shape[0]:
        raise ValueError(
            f"The shape of the given vector `v` is compatible with the shape of the "
            f"given Cholesky factor `L_T`. Expected shape {(L.shape[0],)} but got "
            f"{v.shape}."
        )

    if v.dtype != np.float64:
        raise TypeError(
            f"The given vector `v` does not have dtype `np.float64` (given dtype: "
            f"{L.dtype.name})"
        )

    # Copy on demand
    if not overwrite_L:
        L = L.copy(order="K")

    if not overwrite_v:
        v = v.copy()

    # The algorithm in [1] uses L^T instead of L
    L_T = L.T

    assert not L_T.flags.owndata

    if L_T.flags.c_contiguous:
        _cholesky_rank_1_update_row_major(L_T, v)
    else:
        assert L_T.flags.f_contiguous

        _cholesky_rank_1_update_column_major(L_T, v)

    return L


def _cholesky_rank_1_update_row_major(L_T: np.ndarray, v: np.ndarray) -> None:
    N = L_T.shape[0]

    for k in range(N):
        # TODO: There is a seemingly unnecessary check in the Seeger code here

        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(L_T[k, k], v[k])

        # Apply Givens rotation to diagonal term and corresponding entry
        L_T[k, k], v[k] = scipy.linalg.blas.drot(L_T[k, k], v[k], c, s)

        # TODO: There is a seemingly unnecessary check in the Seeger code here

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_T[k, k] < 0.0:
            L_T[k, k] = -L_T[k, k]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries of L_T and v
        if k + 1 < N:
            scipy.linalg.blas.drot(
                L_T[k, (k + 1) :],
                v[(k + 1) :],
                c,
                s,
                overwrite_x=True,
                overwrite_y=True,
            )


def _cholesky_rank_1_update_column_major(L_T: np.ndarray, v: np.ndarray) -> None:
    N = L_T.shape[0]

    row_inc = 1
    column_inc = N

    k = 0

    drot_n = N - 1
    drot_L_T = L_T.ravel("F")
    drot_off_L_T = column_inc
    drot_inc_L_T = column_inc
    drot_v = v
    drot_offv = 1
    drot_incv = 1

    while k < N:
        # TODO: There is a seemingly unnecessary check in the Seeger code here

        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(L_T[k, k], v[k])

        # Apply Givens rotation to diagonal term and corresponding entry
        L_T[k, k], v[k] = scipy.linalg.blas.drot(L_T[k, k], v[k], c, s)

        # TODO: There is a seemingly unnecessary check in the Seeger code here

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_T[k, k] < 0.0:
            L_T[k, k] = -L_T[k, k]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries of L_T and v
        if drot_n > 0:
            scipy.linalg.blas.drot(
                n=drot_n,
                x=drot_L_T,
                offx=drot_off_L_T,
                incx=drot_inc_L_T,
                y=drot_v,
                offy=drot_offv,
                incy=drot_incv,
                c=c,
                s=s,
                overwrite_x=True,
                overwrite_y=True,
            )

        k += 1

        drot_n -= 1
        drot_off_L_T += row_inc + column_inc
        drot_offv += 1


def cholesky_rank_1_downdate(
    L_T: np.ndarray, v: np.ndarray, reuse_L_T: bool = False, reuse_v: bool = False,
) -> np.ndarray:
    r""" Compute the Cholesky factorization of a symmetric rank-1 udowndate to a
    symmetric positive definite matrix with given Cholesky factorization in a fast and
    stable manner.

    Specifically, given the upper triangular Cholesky factor :math:`L^T` of
    :math:`A = L L^T \in \R^{N \times N}`, this function computes the upper triangular
    Cholesky factor :math:`L'^T` of :math:`A' = L' L'^T`, where :math:`A' = A - v v^T`
    for some vector :math:`v \in \R^n`, if :math:`A'` is positive definite.

    We implement the method in [1, section 3]. This algorithm computes the Cholesky
    decomposition of :math:`A'` from :math:`L_T` in :math:`O(N^2)` time, which is faster
    than the :math:`O(N^3)` time complexity of naively applying a Cholesky algorithm to
    :math:`A'` directly.

    Args:
        L_T:
            Upper triangular Cholesky factor of :math:`A` of shape `(N, N)`. Dtypes
            other than :class:`np.float64` will be cast to :class:`np.float64`
            (essentially triggering a copy). The algorithm is more efficient if this
            array is stored in C-contiguous (i.e. row-major) memory order. The entries
            in the lower triangular part of `L_T` will be ignored by the algorithm.
        v:
            The vector :math:`v` defining the symmetric rank-1 downdate with shape
            `(N,)`. Dtypes other than :class:`np.float64` will be cast to
            :class:`np.float64` (essentially triggering a copy).
        reuse_L_T:
            If set to `True`, the function might reuse the array `L_T` to store the
            upper Cholesky factor of :math:`A'`. In this case, the result is computed
            essentially in-place. Note that passing `True` here does not guarantee that
            `L_T` is reused. However, in this case, additional memory is only allocated
            if absolutely necessary, e.g. if the array has the wrong dtype.
            Passing `False` here will ensure that the array `L_T` is not modified.
        reuse_v:
            If set to `True`, the function might reuse the array `v` as an internal
            computation buffer. In this case, the array `v` might be modified. Note that
            passing `True` here does not guarantee that `v` is reused. However, in this
            case, additional memory is only allocated if absolutely necessary, e.g. if
            the array has the wrong dtype.
            Passing `False` here will ensure that the array `v` is not modified and an
            additional array of shape `(N)` and dtype :class:`np.float64` will always
            be allocated.

    Returns:
        Upper triangular Cholesky factor of :math:`A - v v^T` with dtype
        :class:`np.float64` and shape `(N, N)`. The diagonal entries of this matrix are
        guaranteed to be positive. The entries in the lower triangular part of this
        matrix will be the same as those in the input array `L_T`.

    Raises:
        ValueError: If `L_T` does not have shape `(N, N)` for some `N`.
        ValueError: If `v` does not have shape `(N,)`, while `L_T` has shape `(N, N)`.
        scipy.linalg.LinAlgError: If :math:`A'` is not positive definite.
        ValueError: If `L_T` has zeros among its diagonal entries.

    References:
        [1] Seeger, Matthias, Low Rank Updates for the Cholesky Decomposition, 2008.
    """

    if L_T.ndim != 2 or L_T.shape[0] != L_T.shape[1]:
        raise ValueError("The given Cholesky factor `L_T` is not a square matrix.")

    N = L_T.shape[0]

    if v.ndim != 1 or v.shape[0] != L_T.shape[0]:
        raise ValueError(
            f"The shape of the given vector `v` is compatible with the shape of the "
            f"given Cholesky factor `L_T`. Expected shape {(L_T.shape[0],)} but got "
            f"{v.shape}."
        )

    # Copy
    if reuse_L_T or L_T.dtype is not np.float64:
        L_T = np.array(L_T, dtype=np.float64, order="C")

    if reuse_v or v.dtype is not np.float64:
        v = np.array(v, dtype=np.float64)

    # Compute p
    scipy.linalg.blas.dtrsv(
        a=L_T, x=v, trans=1, overwrite_x=True,
    )

    p = v

    # Compute rho
    rho_sq = 1 - scipy.linalg.blas.ddot(p, p)

    if rho_sq <= 0.0:
        # The updated matrix is positive definite if and only if rho ** 2 is positive
        raise scipy.linalg.LinAlgError(
            "The downdate would not result in a positive definite matrix."
        )

    rho = np.sqrt(rho_sq)

    # "Create" q
    q_0 = rho
    q_1_to_n = p

    # Create temporary vector accumulating Givens rotations of the appended zero vector
    # in the augmented matrix from the left hand side of [1, equation 2]
    temp = np.zeros(N, dtype=np.float64)

    for k in range(N - 1, -1, -1):
        # Generate Givens rotation
        c, s = scipy.linalg.blas.drotg(q_0, q_1_to_n[k],)

        # Apply Givens rotation to q
        q_0, q_1_to_n[k] = scipy.linalg.blas.drot(q_0, q_1_to_n[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate q_0 to a negative
        # value. However, for the algorithm to work, it is important that q_0 remains
        # positive. As a remedy, we add another 180 degree rotation to the Givens
        # rotation matrix. This flips the sign of q_0 while ensuring that the resulting
        # transformation is still a Givens rotation.
        if q_0 < 0.0:
            q_0 = -q_0
            c = -c
            s = -s

        # Apply (possibly modified) Givens rotation to the augmented matrix [0 L]^T
        if L_T[k, k] == 0.0:
            # This can only happen if L_T is not an upper triangular matrix with
            # non-zero diagonal
            raise ValueError(
                "The given Cholesky factor `L_T` does not have a non-zero diagonal."
            )

        scipy.linalg.blas.drot(
            temp[k:], L_T[k, k:], c, -s, overwrite_x=True, overwrite_y=True,
        )

        # Applying the Givens rotation might lead to a negative diagonal element in L_T.
        # However, by convention, the diagonal entries of a Cholesky factor are
        # positive. As a remedy, we simply rescale the whole row. Note that this is
        # possible, since rescaling a row is equivalent to a mirroring along one
        # dimension which is in turn an orthogonal transformation.
        if L_T[k, k] < 0.0:
            L_T[k, k:] = -L_T[k, k:]

    return L_T
