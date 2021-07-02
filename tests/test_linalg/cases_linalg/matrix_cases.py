"""Test cases given by different matrices or linear operators."""

import os
from typing import Union

import numpy as np
import pytest
import pytest_cases
import scipy

from probnum import kernels, linops, problems
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

mrows = [1, 2, 10, 100]
ncols = [1, 2, 10, 100]


class SPDMatrix:
    @pytest.mark.parametrize("n", ncols)
    def case_random_spd_matrix(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return random_spd_matrix(dim=n, random_state=rng)

    def case_random_sparse_spd_matrix(self, rng: np.random.Generator) -> np.ndarray:
        return random_sparse_spd_matrix(dim=100, density=0.01, random_state=rng)

    def case_random_sparse_matrix(
        self, rng: np.random.Generator
    ) -> scipy.sparse.csr_matrix:
        matrix = scipy.sparse.rand(
            1000, 1000, density=0.01, format="coo", dtype=np.double, random_state=rng
        )
        matrix.setdiag(2)
        return matrix.tocsr()

    @pytest.mark.parametrize("n", ncols)
    def case_kernel_matrix(self, n: int, rng: np.random.Generator) -> np.ndarray:
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
        return linops.Scaling(np.arange(10, 1))


@pytest_cases.parametrize_with_cases("spd_matrix", cases=SPDMatrix)
def case_spd_linsys(
    spd_matrix: Union[np.ndarray, linops.LinearOperator], rng: np.random.Generator
) -> problems.LinearSystem:
    solution = rng.normal(size=spd_matrix.shape[1])
    return problems.LinearSystem(
        A=spd_matrix, b=spd_matrix @ solution, solution=solution
    )
