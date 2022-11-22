"""Test cases given by different matrices or linear operators."""

import os

import scipy

from probnum import backend, linops
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix
from probnum.randprocs import kernels

from pytest_cases import case, parametrize
from tests.utils.random import rng_state_from_sampling_args

m_rows = [1, 2, 10, 100]
n_cols = [1, 2, 10, 100]


@case(tags=["symmetric", "positive_definite"])
@parametrize("n", n_cols)
def case_random_spd_matrix(n: int) -> backend.Array:
    rng_state = rng_state_from_sampling_args(n)
    return random_spd_matrix(rng_state=rng_state, shape=(n, n))


@case(tags=["symmetric", "positive_definite"])
def case_random_sparse_spd_matrix() -> scipy.sparse.spmatrix:
    rng_state = backend.random.rng_state(1)
    return random_sparse_spd_matrix(
        rng_state=rng_state, shape=(1000, 1000), density=0.01
    )


@case(tags=["symmetric", "positive_definite"])
@parametrize("n", n_cols)
def case_kernel_matrix(n: int) -> backend.Array:
    """Kernel Gram matrix."""
    rng_state = rng_state_from_sampling_args(n)
    x_min, x_max = (-4.0, 4.0)
    X = backend.random.uniform(
        rng_state=rng_state, minval=x_min, maxval=x_max, shape=(n, 1)
    )
    kern = kernels.ExpQuad(input_shape=1, lengthscale=1.0)
    return kern(X)


@case(tags=["symmetric", "positive_definite"])
def case_poisson() -> backend.Array:
    """Poisson equation with Dirichlet conditions.

        - Laplace(u) = f    in the interior
                    u = u_D  on the boundary
    where
        u_D = 1 + x^2 + 2y^2
        f = -4

    Linear system resulting from discretization on an elliptic grid.
    """
    fpath = os.path.join(os.path.dirname(__file__), "../../resources")
    return scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")


@case(tags=["symmetric", "positive_definite"])
def case_scaling_linop() -> linops.Scaling:
    return linops.Scaling(backend.arange(10))
