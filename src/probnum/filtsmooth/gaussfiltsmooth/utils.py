"""Utility functions for Gaussian filtering and smoothing."""


def cholesky_sum_choleskies(A, B):
    """Computes cholesky factorisation CCt = AAt + BBt."""
    X = np.vstack((A.T, B.T))
    l = np.linalg.qr(X, mode="r").T
    l = l @ np.diag(np.sign(np.diag(l)))
    return l


def cholesky_prod(H, L):
    """Computes the cholesky factorisation CCt = H L Lt Ht for non-square H."""

    l = np.linalg.qr(L.T @ H.T, mode="r").T

    l = l @ np.diag(np.sign(np.diag(l)))

    return l
