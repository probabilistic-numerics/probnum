import numpy as np
import scipy.linalg
from scipy.linalg import cholesky


def cholesky_solve(
    cholesky: np.ndarray,
    b: np.ndarray,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
):
    if b.ndim in (1, 2):
        return scipy.linalg.cho_solve(
            (cholesky, lower),
            b,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )

    # In order to apply __matmul__ broadcasting, we need to reshape the stack of
    # matrices `b` into a matrix whose first axis corresponds to the penultimate axis in
    # the matrix stack and whose second axis is a flattened/raveled representation of
    # all the remaining axes

    # We can handle a stack of vectors in a simplified manner
    stack_of_vectors = b.shape[-1] == 1

    if stack_of_vectors:
        cols_batch_first = b[..., 0]
    else:
        cols_batch_first = np.swapaxes(b, -2, -1)

    cols_batch_last = np.array(cols_batch_first.T, copy=False, order="F")

    # Flatten the trailing axes and remember shape to undo flattening operation later
    unflatten_shape = cols_batch_last.shape
    cols_flat_batch_last = cols_batch_last.reshape(
        (cols_batch_last.shape[0], -1),
        order="F",
    )

    assert cols_flat_batch_last.flags.f_contiguous

    sols_flat_batch_last = scipy.linalg.cho_solve(
        (cholesky, lower),
        cols_flat_batch_last,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )

    assert sols_flat_batch_last.flags.f_contiguous

    # Undo flattening operation
    sols_batch_last = sols_flat_batch_last.reshape(unflatten_shape, order="F")

    sols_batch_first = sols_batch_last.T

    if stack_of_vectors:
        return sols_batch_first[..., None]

    return np.swapaxes(sols_batch_first, -2, -1)
