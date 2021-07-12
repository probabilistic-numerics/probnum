"""Test cases given by different matrices or linear operators."""

import os

import numpy as np
import scipy
from pytest_cases import case, parametrize

from probnum import kernels, linops
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

m_rows = [1, 2, 10, 100]
n_cols = [1, 2, 10, 100]


@case(tags=["symmetric", "positive_definite"])
@parametrize("n", n_cols)
def case_random_spd_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    return random_spd_matrix(dim=n, rng=rng)


@case(tags=["symmetric", "positive_definite"])
def case_random_sparse_spd_matrix(rng: np.random.Generator) -> scipy.sparse.spmatrix:
    return random_sparse_spd_matrix(dim=1000, density=0.01, rng=rng)


@case(tags=["symmetric", "positive_definite"])
@parametrize("n", n_cols)
def case_kernel_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Kernel Gram matrix."""
    x_min, x_max = (-4.0, 4.0)
    X = rng.uniform(x_min, x_max, (n, 1))
    kern = kernels.ExpQuad(input_dim=1, lengthscale=1)
    return kern(X)


@case(tags=["symmetric", "positive_definite"])
def case_poisson() -> np.ndarray:
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
    return linops.Scaling(np.arange(10))
