"""Tests for linear system beliefs."""

from typing import Union

import numpy as np
import pytest
import scipy.linalg

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_dimension_mismatch_raises_value_error():
    """Test whether mismatched components result in a ValueError."""
    m, n, nrhs = 5, 3, 2
    A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
    Ainv = A
    x = rvs.Normal(mean=np.zeros((n, nrhs)), cov=np.eye(n * nrhs))
    b = rvs.Constant(np.ones((m, nrhs)))

    # A does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=rvs.Constant(np.ones((m + 1, nrhs))))

    # A does not match x
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=rvs.Normal(mean=np.zeros((n + 1, nrhs)), cov=np.eye((n + 1) * nrhs)),
            b=b,
        )

    # x does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=rvs.Normal(mean=np.zeros((n, nrhs + 1)), cov=np.eye(n * (nrhs + 1))),
            b=b,
        )

    # A does not match Ainv
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=rvs.Normal(mean=np.ones((m + 1, n)), cov=np.eye((m + 1) * n)),
            x=x,
            b=b,
        )


def test_beliefs_are_two_dimensional():
    """Check whether all beliefs over quantities of interest are two dimensional."""
    m, n = 5, 3
    A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
    Ainv = A
    x = rvs.Normal(mean=np.zeros(n), cov=np.eye(n))
    b = rvs.Constant(np.ones(m))
    belief = LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b)

    assert belief.A.ndim == 2
    assert belief.Ainv.ndim == 2
    assert belief.x.ndim == 2
    assert belief.b.ndim == 2


def test_non_two_dimensional_raises_value_error():
    """Test whether specifying higher-dimensional random variables raise a
    ValueError."""
    A = rvs.Constant(np.eye(5))
    Ainv = rvs.Constant(np.eye(5))
    x = rvs.Constant(np.ones((5, 1)))
    b = rvs.Constant(np.ones((5, 1)))

    # A.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A[:, None], Ainv=Ainv, x=x, b=b)

    # Ainv.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv[:, None], x=x, b=b)

    # x.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x[:, None], b=b)

    # b.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b[:, None])


# Classmethod tests


def test_from_solution_array(
    belief_class: LinearSystemBelief,
    linsys: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether a linear system belief can be created from a solution estimate given
    as an array."""
    x0 = random_state.normal(size=linsys.A.shape[1])
    belief_class.from_solution(x0=x0, problem=linsys)


def test_from_solution_generates_consistent_inverse_belief(
    belief_class: LinearSystemBelief,
    linsys: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether the belief for the inverse generated from a solution guess matches
    the belief for the solution."""
    x0 = random_state.normal(size=linsys.A.shape[1])
    belief = belief_class.from_solution(x0=x0, problem=linsys)
    np.testing.assert_allclose(belief.x.mean, belief.Ainv.mean @ linsys.b)


def test_from_solution_creates_better_initialization(belief_class: LinearSystemBelief):
    """Test whether if possible, a better initial value x1 is constructed from x0."""
    x0_list = []
    linsys = LinearSystem(
        A=np.array([[4, 2, -6, 4], [2, 2, -3, 1], [-6, -3, 13, 0], [4, 1, 0, 30]]),
        solution=np.array([2, 0, -1, 2]),
        b=np.array([22, 9, -25, 68]),
    )

    # <b, x0> < 0
    x0_list.append(-linsys.b)

    # <b, x0> = 0, b != 0
    x0_list.append(np.array([0.5, -1, 0, -1 / 34])[:, None])
    pytest.approx((x0_list[1].T @ linsys.b).item(), 0.0)

    for x0 in x0_list:
        belief = belief_class.from_solution(x0=x0, problem=linsys)
        assert (belief.x.mean.T @ linsys.b).item() > 0.0, (
            "Inner product <x0, b>="
            f"{(belief.x.mean.T @ linsys.b).item():.4f} is not positive.",
        )
        error_x0 = ((linsys.solution - x0).T @ linsys.A @ (linsys.solution - x0)).item()
        error_x1 = (
            (linsys.solution - belief.x.mean).T
            @ linsys.A
            @ (linsys.solution - belief.x.mean)
        ).item()
        assert error_x1 < error_x0, (
            "Initialization for the solution x0 is not better in A-norm "
            "than the user-specified one.",
        )

    # b = 0
    linsys_homogeneous = LinearSystem(A=linsys.A, b=np.zeros_like(linsys.b))
    belief = belief_class.from_solution(
        x0=np.ones_like(linsys.b), problem=linsys_homogeneous
    )
    np.testing.assert_allclose(belief.x.mean, np.zeros_like(linsys.b))


def test_from_matrix(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from a system matrix estimate
    given as an array, sparse matrix or linear operator."""
    if (belief_class is WeakMeanCorrespondenceBelief) and not isinstance(
        mat, linops.LinearOperator
    ):
        with pytest.raises(TypeError):
            # Inefficient belief construction via explicit inversion raises error
            belief_class.from_matrix(A0=mat, problem=linsys)
    else:
        belief_class.from_matrix(A0=mat, problem=linsys)


def test_from_inverse(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from an inverse estimate given
    as an array, sparse matrix or linear operator."""
    if (belief_class is WeakMeanCorrespondenceBelief) and not isinstance(
        mat, linops.LinearOperator
    ):
        with pytest.raises(TypeError):
            belief_class.from_inverse(Ainv0=mat, problem=linsys)
    else:
        belief_class.from_inverse(Ainv0=mat, problem=linsys)


def test_from_matrices(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from an estimate of the system
    matrix and its inverse given as arrays, sparse matrices or linear operators."""
    belief_class.from_matrices(A0=mat, Ainv0=mat, problem=linsys)
