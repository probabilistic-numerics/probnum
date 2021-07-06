"""Test cases given by different matrices or linear operators."""

import os

import numpy as np
import pytest
import scipy

from probnum import kernels, linops
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

m_rows = [1, 2, 10, 100]
n_cols = [1, 2, 10, 100]


class SPDMatrix:
    """Symmetric positive definite matrices."""

    @pytest.mark.parametrize("n", n_cols)
    def case_random_spd_matrix(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return random_spd_matrix(dim=n, random_state=rng)

    def case_random_sparse_spd_matrix(
        self, rng: np.random.Generator
    ) -> scipy.sparse.spmatrix:
        return random_sparse_spd_matrix(dim=1000, density=0.01, random_state=rng)

    @pytest.mark.parametrize("n", n_cols)
    def case_kernel_matrix(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Kernel Gram matrix."""
        x_min, x_max = (-4.0, 4.0)
        X = rng.uniform(x_min, x_max, (n, 1))
        kern = kernels.ExpQuad(input_dim=1, lengthscale=1)
        return kern(X)

    def case_poisson(self) -> np.ndarray:
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

    def case_scaling_linop(self) -> linops.Scaling:
        return linops.Scaling(np.arange(10))
