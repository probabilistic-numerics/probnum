"""Benchmarks for linear solvers."""

import os

import numpy as np
import scipy.sparse

from benchmarks.benchmark_utils import SPD_MATRIX_5x5
from probnum.linalg import problinsolve


def load_poisson_linear_system():
    """Poisson equation with Dirichlet conditions.

      - Laplace(u) = f    in the interior
                 u = u_D  on the boundary
    where
        u_D = 1 + x^2 + 2y^2
        f = -4

    Linear system resulting from discretization on an elliptic grid.
    """
    # pylint: disable=invalid-name
    fpath = os.path.join(os.path.dirname(__file__), "../tests/resources")
    A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
    f = np.load(file=fpath + "/rhs_poisson.npy")
    return A, f


class LinSolve:
    """Benchmark solving a linear system."""

    param_names = ["system"]
    params = [["sparse", "dense"]]  # , "large-scale"]

    def setup(self, system):
        # Seed
        np.random.seed(42)

        if system == "sparse":
            (
                self.A,
                self.b,
            ) = load_poisson_linear_system()
        elif system == "dense":
            self.A = SPD_MATRIX_5x5
            self.b = np.random.normal(size=self.A.shape[0])
        elif system == "large-scale":
            self.A = None
            self.b = None

    def time_solve(self, system):
        """Time solving a linear system."""
        problinsolve(A=self.A, b=self.b)

    def mem_solve(self, system):
        """Time solving a linear system."""
        problinsolve(A=self.A, b=self.b)

    def peakmem_solve(self, system):
        """Time solving a linear system."""
        problinsolve(A=self.A, b=self.b)


class PosteriorDist:
    """Benchmark sampling from the posterior distribution."""

    param_names = ["output"]
    params = [["solution", "matrix", "matrix_inverse"]]

    def setup(self, output):
        # pylint: disable=invalid-name

        # Sparse system
        self.A, self.b = load_poisson_linear_system()

        # Solve linear system
        self.xhat, self.Ahat, self.Ainvhat, _ = problinsolve(A=self.A, b=self.b)

        # Benchmark parameters
        self.n_samples = 10

    def time_sample(self, output):
        """Time sampling from the posterior distribution."""
        if output == "solution":
            self.xhat.sample(self.n_samples)
        elif output == "matrix":
            self.Ahat.sample(self.n_samples)
        elif output == "matrix_inverse":
            self.Ainvhat.sample(self.n_samples)
