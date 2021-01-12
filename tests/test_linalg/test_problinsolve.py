"""Tests for linear solvers."""
import os
import unittest

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from probnum import linalg, linops
from probnum import random_variables as rvs
from probnum.linalg import problinsolve
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix
from tests.testing import NumpyAssertions

# pylint: disable="invalid-name"


class LinearSolverTests(unittest.TestCase, NumpyAssertions):
    """General test case for linear solvers."""

    @classmethod
    def setUpClass(cls) -> None:

        cls.rng = np.random.default_rng(42)

        # Symmetric positive definite matrices
        dim_spd = 100
        cls.spd_system = LinearSystem.from_matrix(
            A=random_spd_matrix(dim=dim_spd, random_state=cls.rng),
            random_state=cls.rng,
        )

        cls.sparse_spd_system = LinearSystem.from_matrix(
            A=random_sparse_spd_matrix(dim=dim_spd, density=0.01, random_state=cls.rng),
            random_state=cls.rng,
        )

        # Discretized Poisson equation with Dirichlet boundary conditions
        fpath = os.path.join(os.path.dirname(__file__), "../resources")
        _A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
        _b = np.load(file=fpath + "/rhs_poisson.npy")
        cls.poisson_linear_system = LinearSystem(
            A=_A,
            solution=scipy.sparse.linalg.spsolve(A=_A, b=_b),
            b=_b,
        )

        # Kernel matrices
        n = 100
        x_min, x_max = (-4.0, 4.0)
        X = np.random.uniform(x_min, x_max, (n, 1))

        # RBF kernel
        lengthscale = 1
        var = 1
        X_norm = np.sum(X ** 2, axis=-1)
        K_rbf = var * np.exp(
            -1
            / (2 * lengthscale ** 2)
            * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
        ) + 10 ** -2 * np.eye(n)
        cls.rbf_kernel_linear_system = LinearSystem.from_matrix(
            A=K_rbf, random_state=cls.rng
        )

    def setUp(self):
        """Create joint resources for subclasses."""
        self.linear_solvers = [problinsolve]  # , bayescg]

    def test_randvar_output(self):
        """Probabilistic linear solvers output random variables."""
        n = 10
        A = random_spd_matrix(dim=n, random_state=self.rng)
        b = np.random.rand(n)
        for linsolve in self.linear_solvers:
            with self.subTest():
                x, A, Ainv, _ = linsolve(A=A, b=b)
                for rv in [x, A, Ainv]:
                    self.assertIsInstance(
                        rv,
                        rvs.RandomVariable,
                        msg="Output of probabilistic linear solver is not a random "
                        "variable.",
                    )

    def test_system_dimension_mismatch(self):
        """Test whether linear solvers throw an exception for input with mismatched
        dimensions."""
        A = np.zeros(shape=[3, 3])
        b = np.zeros(shape=[4])
        x0 = np.zeros(shape=[1])
        for linsolve in self.linear_solvers:
            with self.subTest():
                with self.assertRaises(
                    ValueError, msg="Invalid input formats should raise a ValueError."
                ):
                    # A, b dimension mismatch
                    linsolve(A=A, b=b)
                    # A, x0 dimension mismatch
                    linsolve(A=A, b=np.zeros(A.shape[0]), x0=x0)
                    # A not square
                    linsolve(
                        A=np.zeros([3, 4]),
                        b=np.zeros(A.shape[0]),
                        x0=np.zeros(shape=[A.shape[1]]),
                    )

    def test_zero_rhs(self):
        """Linear system with zero right hand side."""
        n = 10
        A = random_spd_matrix(dim=n, random_state=self.rng)
        b = np.zeros(n)
        tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

        for linsolve in self.linear_solvers:
            with self.subTest():
                for tol in tols:
                    x, _, _, _ = linsolve(A=A, b=b, atol=tol)
                    self.assertAllClose(x.mean, 0, atol=1e-15)

    def test_multiple_rhs(self):
        """Linear system with matrix right hand side."""
        n = 10
        A = random_spd_matrix(dim=n, random_state=self.rng)
        B = np.random.rand(10, 5)

        for linsolve in self.linear_solvers:
            with self.subTest():
                x, _, _, info = linsolve(A=A, b=B)
                self.assertEqual(
                    x.shape,
                    B.shape,
                    msg="Shape of solution and right hand side do not match.",
                )
                self.assertAllClose(x.mean, np.linalg.solve(A, B))

    def test_spd_matrix(self):
        """Random symmetric positive definite matrix."""
        for linsolve in self.linear_solvers:
            with self.subTest():
                x, _, _, _ = linsolve(A=self.spd_system.A, b=self.spd_system.b)
                self.assertAllClose(
                    x.mean,
                    self.spd_system.solution.ravel(),
                    rtol=1e-5,
                    atol=1e-5,
                    msg="Solution does not match true solution.",
                )

    def test_sparse_spd_matrix(self):
        """Sparse random symmetric positive definite matrix."""
        for linsolve in self.linear_solvers:
            with self.subTest():
                x, _, _, _ = linsolve(
                    A=self.sparse_spd_system.A, b=self.sparse_spd_system.b
                )
                self.assertAllClose(
                    x.mean,
                    self.sparse_spd_system.solution.ravel(),
                    rtol=1e-5,
                    atol=1e-5,
                    msg="Solution does not match true solution.",
                )

    def test_sparse_poisson(self):
        """(Sparse) linear system from Poisson PDE with boundary conditions."""
        for linsolve in self.linear_solvers:
            with self.subTest():
                x, _, _, _ = linsolve(
                    A=self.poisson_linear_system.A, b=self.poisson_linear_system.b
                )
                self.assertAllClose(
                    x.mean.reshape(-1, 1),
                    self.poisson_linear_system.solution,
                    rtol=1e-5,
                    msg="Solution from probabilistic linear solver does"
                    + " not match scipy.sparse.linalg.spsolve.",
                )

    def test_posterior_means_symmetric(self):
        """"""
        pass

    def test_posterior_means_positive_definite(self):
        """"""
        pass


class ProbLinSolveTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the :func:`problinsolve` interface function."""

    def setUp(self):
        """Test resources."""

    def test_prior_information(self):
        """The solver should automatically handle different types of prior
        information."""
        pass

    def test_prior_dimension_mismatch(self):
        """Test whether the probabilistic linear solver throws an exception for priors
        with mismatched dimensions."""
        A = np.zeros(shape=[3, 3])
        with self.assertRaises(
            ValueError, msg="Invalid input formats should raise a ValueError."
        ):
            # A inverse not square
            problinsolve(
                A=A,
                b=np.zeros(A.shape[0]),
                Ainv0=np.zeros([2, 3]),
                x0=np.zeros(shape=[A.shape[1]]),
            )
            # A, Ainv dimension mismatch
            problinsolve(
                A=A,
                b=np.zeros(A.shape[0]),
                Ainv0=np.zeros([2, 2]),
                x0=np.zeros(shape=[A.shape[1]]),
            )


class BayesCGTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the :func:`bayescg` interface function."""
