"""Tests for beliefs about quantities of interest of a linear system."""
import numpy as np

from probnum import backend, linops, randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest


def test_init_invalid_belief():
    """Test whether instantiating a belief over neither x nor Ainv raises an error."""
    with pytest.raises(TypeError):
        LinearSystemBelief(x=None, Ainv=None)


def test_dimension_mismatch_raises_value_error():
    """Test whether mismatched components result in a ValueError."""
    m, n, nrhs = 5, 3, 2
    A = randvars.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
    Ainv = A
    x = randvars.Normal(mean=np.zeros((n, nrhs)), cov=np.eye(n * nrhs))
    b = randvars.Constant(np.ones((m, nrhs)))

    # A does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A, Ainv=Ainv, x=x, b=randvars.Constant(np.ones((m + 1, nrhs)))
        )

    # A does not match x
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=randvars.Normal(mean=np.zeros((n + 1, nrhs)), cov=np.eye((n + 1) * nrhs)),
            b=b,
        )

    # x does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=randvars.Normal(mean=np.zeros((n, nrhs + 1)), cov=np.eye(n * (nrhs + 1))),
            b=b,
        )

    # A does not match Ainv
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=randvars.Normal(mean=np.ones((m + 1, n)), cov=np.eye((m + 1) * n)),
            x=x,
            b=b,
        )


def test_non_two_dimensional_raises_value_error():
    """Test whether specifying higher-dimensional random variables raise a
    ValueError."""
    A = randvars.Constant(np.eye(5))
    Ainv = randvars.Constant(np.eye(5))
    x = randvars.Constant(np.ones((5, 1)))
    b = randvars.Constant(np.ones((5, 1)))

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


# def test_non_randvar_arguments_raises_type_error():
#     A = np.eye(5)
#     Ainv = np.eye(5)
#     x = np.ones((5, 1))
#     b = np.ones((5, 1))

#     with pytest.raises(TypeError):
#         LinearSystemBelief(x=x)

#     with pytest.raises(TypeError):
#         LinearSystemBelief(Ainv=Ainv)

#     with pytest.raises(TypeError):
#         LinearSystemBelief(x=randvars.Constant(x), A=A)

#     with pytest.raises(TypeError):
#         LinearSystemBelief(x=randvars.Constant(x), b=b)


def test_induced_solution_belief():
    """Test whether a consistent belief over the solution is inferred from a belief over
    the inverse."""
    rng_state = backend.random.rng_state(8294)
    rng_state_A, rng_state_b = backend.random.split(rng_state=rng_state)
    n = 5
    A = randvars.Constant(random_spd_matrix(rng_state=rng_state_A, shape=(n, n)))
    Ainv = randvars.Normal(
        mean=linops.Scaling(factors=1 / np.diag(A.mean)),
        cov=linops.SymmetricKronecker(linops.Identity(n)),
    )
    b = randvars.Constant(
        backend.random.standard_normal(rng_state=rng_state_b, shape=(n, 1))
    )
    prior = LinearSystemBelief(A=A, Ainv=Ainv, x=None, b=b)

    x_infer = Ainv @ b
    np.testing.assert_allclose(prior.x.mean, x_infer.mean)
    np.testing.assert_allclose(prior.x.cov.todense(), x_infer.cov.todense())
