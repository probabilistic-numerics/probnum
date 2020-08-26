"""
Benchmarks for linear solvers.
"""

import os
import numpy as np
import scipy.sparse

from probnum.linalg import problinsolve


def load_poisson_linear_system():
    """
    Poisson equation with Dirichlet conditions.

      - Laplace(u) = f    in the interior
                 u = u_D  on the boundary
    where
        u_D = 1 + x^2 + 2y^2
        f = -4

    Linear system resulting from discretization on an elliptic grid.
    """
    fpath = os.path.join(os.path.dirname(__file__), "../tests/resources")
    A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")  # pylint: disable=invalid-name
    f = np.load(file=fpath + "/rhs_poisson.npy")  # pylint: disable=invalid-name
    return A, f


class LinSolve:
    """Benchmark solving a linear system."""

    param_names = ["system"]
    params = [["sparse", "dense"]]  # , "large-scale"]

    def setup(self, system):
        # pylint: disable=missing-function-docstring,attribute-defined-outside-init

        # Seed
        np.random.seed(42)

        if system == "sparse":
            self.A, self.b = load_poisson_linear_system()  # pylint: disable=invalid-name
        elif system == "dense":
            self.A = np.array(
                [
                    [2.3, -2.3, 3.5, 4.2, 1.8],
                    [-2.3, 3.0, -3.5, -4.8, -1.9],
                    [3.5, -3.5, 6.9, 5.8, 0.8],
                    [4.2, -4.8, 5.8, 10.1, 6.3],
                    [1.8, -1.9, 0.8, 6.3, 12.1],
                ]
            )
            self.b = np.random.normal(size=self.A.shape[0])
        elif system == "large-scale":
            self.A = None
            self.b = None

    def time_solve(self, system):
        """Time solving a linear system"""
        # pylint: disable=unused-argument
        problinsolve(A=self.A, b=self.b)

    def mem_solve(self, system):
        """Time solving a linear system"""
        # pylint: disable=unused-argument

        # I would remove the self.xhat, ... bit but I don't know what mem_solve really needs... (NK)
        self.xhat, self.Ahat, self.Ainvhat, _ = problinsolve(A=self.A, b=self.b)

    def peakmem_solve(self, system):
        """Time solving a linear system"""
        # pylint: disable=unused-argument
        problinsolve(A=self.A, b=self.b)


class PosteriorDist:
    """Benchmark sampling from the posterior distribution."""

    param_names = ["output"]
    params = [["solution", "matrix", "matrix_inverse"]]

    def setup(self, output):
        # pylint: disable=missing-function-docstring,attribute-defined-outside-init
        # pylint: disable=unused-argument

        # Sparse system
        self.A, self.b = load_poisson_linear_system()  # pylint: disable=invalid-name

        # Solve linear system
        self.xhat, self.Ahat, self.Ainvhat, _ = problinsolve(A=self.A, b=self.b)   # pylint: disable=invalid-name

        # Benchmark parameters
        self.n_samples = 10

    def time_sample(self, output):
        """Time sampling from the posterior distribution"""
        if output == "solution":
            self.xhat.sample(self.n_samples)
        elif output == "matrix":
            self.Ahat.sample(self.n_samples)
        elif output == "matrix_inverse":
            self.Ainvhat.sample(self.n_samples)
