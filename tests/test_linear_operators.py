import pytest
import numpy as np
from probnum.linalg import linear_operators as linops


def test_linop_construction():
    """Create linear operators via various construction methods."""

    def mv(v):
        return np.array([v[0], v[0] + v[1]])

    linops.LinearOperator(shape=(2, 2), matvec=mv)


def test_matvec():
    """Matrix vector multiplication."""
    A = 2 * np.eye(5)
    Aop = linops.aslinearoperator(A)
    np.allclose(A, Aop @ np.eye(5))
