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


class ProbLinSolveTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the :func:`problinsolve` interface function."""

    def setUp(self):
        """Test resources."""

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

    # def test_solution_equivalence(self):
    #     """The iteratively computed solution should match the induced solution
    #     estimate: x_k = E[A^-1] b"""
    #     A, f = self.poisson_linear_system
    #
    #     for matblinsolve in self.problinsolvers:
    #         with self.subTest():
    #             # Solve linear system
    #             u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f)
    #
    #             # E[x] = E[A^-1] b
    #             self.assertAllClose(u_solver.mean, (Ainvhat @ f[:, None]).mean.ravel(), rtol=1e-5,
    #                                 msg="Solution from matrix-based probabilistic linear solver does not match the " +
    #                                     "estimated inverse, i.e. x =/= Ainv @ b ")


class BayesCGTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the :func:`bayescg` interface function."""


class LinearSolverTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear solvers."""

    def setUp(self):
        """Resources for tests."""

        self.rng = np.random.default_rng()

        # Poisson equation with Dirichlet conditions.
        #
        #   - Laplace(u) = f    in the interior
        #              u = u_D  on the boundary
        # where
        #     u_D = 1 + x^2 + 2y^2
        #     f = -4
        #
        # Linear system resulting from discretization on an elliptic grid.
        fpath = os.path.join(os.path.dirname(__file__), "../resources")
        A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
        f = np.load(file=fpath + "/rhs_poisson.npy")
        self.poisson_linear_system = A, f

        # Kernel matrices
        np.random.seed(42)

        # Toy data
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
        )
        K_rbf = K_rbf + 10 ** -2 * np.eye(n)
        x_true = np.random.normal(size=(n,))
        b = K_rbf @ x_true
        self.rbf_kernel_linear_system = K_rbf, b, x_true

        # Probabilistic linear solvers
        self.problinsolvers = [linalg.problinsolve]

    def test_residual_matches_error(self):
        """Test whether the residual norm matches the error of the computed solution
        estimate."""
        A, b, x_true = self.rbf_kernel_linear_system

        for plinsolve in self.problinsolvers:
            with self.subTest():
                x_est, Ahat, Ainvhat, info = plinsolve(A=A, b=b)
                self.assertAlmostEqual(
                    info["resid_l2norm"],
                    np.linalg.norm(A @ x_est.mean - b),
                    msg="Residual in output info does not match l2-error of solution "
                    "estimate.",
                )

    def test_posterior_uncertainty_zero_in_explored_space(self):
        """Test whether the posterior uncertainty over the matrices A and Ainv is zero
        in the already explored spaces span(S) and span(Y)."""
        A, b, x_true = self.rbf_kernel_linear_system
        n = A.shape[0]

        for calibrate in [False, 0.0]:  # , 10 ** -6, 2.8]:
            # TODO (probnum#100) expand this test to the prior covariance class
            # admitting calibration
            with self.subTest():
                # Define callback function to obtain search directions
                S = []  # search directions
                Y = []  # observations

                # pylint: disable=cell-var-from-loop
                def callback_postparams(xk, Ak, Ainvk, sk, yk, alphak, resid):
                    S.append(sk)
                    Y.append(yk)

                # Solve linear system
                u_solver, Ahat, Ainvhat, info = linalg.problinsolve(
                    A=A,
                    b=b,
                    assume_A="sympos",
                    callback=callback_postparams,
                    calibration=calibrate,
                )
                # Create arrays from lists
                S = np.squeeze(np.array(S)).T
                Y = np.squeeze(np.array(Y)).T

                self.assertAllClose(
                    Ahat.cov.A @ S,
                    np.zeros_like(S),
                    atol=1e-6,
                    msg="Uncertainty over A in explored space span(S) not zero.",
                )
                self.assertAllClose(
                    Ainvhat.cov.A @ Y,
                    np.zeros_like(S),
                    atol=1e-6,
                    msg="Uncertainty over Ainv in explored space span(Y) not zero.",
                )

    def test_posterior_covariance_posdef(self):
        """Posterior covariances of the output must be positive (semi-) definite."""
        # Initialization
        A, f = self.poisson_linear_system
        eps = 10 ** -12

        for matblinsolve in self.problinsolvers:
            with self.subTest():
                # Solve linear system
                u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f)

                # Check positive definiteness
                self.assertArrayLess(
                    np.zeros(np.shape(A)[0]),
                    np.real_if_close(np.linalg.eigvals(Ahat.cov.A.todense())) + eps,
                    msg="Covariance of A not positive semi-definite.",
                )
                self.assertArrayLess(
                    np.zeros(np.shape(A)[0]),
                    np.real_if_close(np.linalg.eigvals(Ainvhat.cov.A.todense())) + eps,
                    msg="Covariance of Ainv not positive semi-definite.",
                )

    def test_matrixprior(self):
        """Solve random linear system with a matrix-based linear solver."""
        np.random.seed(1)
        # Linear system
        n = 10
        A = np.random.rand(n, n)
        A = A.dot(A.T) + n * np.eye(n)  # Symmetrize and make diagonally dominant
        x_true = np.random.normal(size=(n,))
        b = A @ x_true

        # Prior distributions on A
        covA = linops.SymmetricKronecker(A=np.eye(n))
        Ainv0 = rvs.Normal(mean=np.eye(n), cov=covA)

        for matblinsolve in self.problinsolvers:
            with self.subTest():
                x, Ahat, Ainvhat, info = matblinsolve(A=A, Ainv0=Ainv0, b=b)

                self.assertAllClose(
                    x.mean,
                    x_true,
                    rtol=1e-6,
                    atol=1e-6,
                    msg="Solution for matrixvariate prior does not match true solution.",
                )

    def test_searchdir_conjugacy(self):
        """Search directions should remain A-conjugate up to machine precision, i.e.
        s_i^T A s_j = 0 for i != j."""
        searchdirs = []

        # Define callback function to obtain search directions
        def callback_searchdirs(xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs):
            searchdirs.append(sk)

        # Solve linear system
        A, f = self.poisson_linear_system

        for plinsolve in self.problinsolvers:
            with self.subTest():
                plinsolve(A=A, b=f, callback=callback_searchdirs)

                # Compute pairwise inner products in A-space
                search_dir_arr = np.squeeze(np.array(searchdirs)).T
                inner_prods = search_dir_arr.T @ A @ search_dir_arr

                # Compare against identity matrix
                self.assertAllClose(
                    np.diag(np.diag(inner_prods)),
                    inner_prods,
                    atol=1e-7,
                    msg="Search directions from solver are not A-conjugate.",
                )

    def test_posterior_mean_CG_equivalency(self):
        """The probabilistic linear solver(s) should recover CG iterates as a posterior
        mean for specific covariances."""

        # Linear system
        A, b = self.poisson_linear_system

        # Callback function to return CG iterates
        cg_iterates = []

        def callback_iterates_CG(xk):
            cg_iterates.append(
                np.eye(np.shape(A)[0]) @ xk
            )  # identity hack to actually save different iterations

        # Solve linear system

        # Initial guess as chosen by PLS: x0 = Ainv.mean @ b
        x0 = b

        # Conjugate gradient method
        xhat_cg, info_cg = scipy.sparse.linalg.cg(
            A=A, b=b, x0=x0, tol=10 ** -6, callback=callback_iterates_CG
        )
        cg_iters_arr = np.array([x0] + cg_iterates)

        # Matrix priors (encoding weak symmetric posterior correspondence)
        Ainv0 = rvs.Normal(
            mean=linops.Identity(A.shape[1]),
            cov=linops.SymmetricKronecker(A=linops.Identity(A.shape[1])),
        )
        A0 = rvs.Normal(
            mean=linops.Identity(A.shape[1]),
            cov=linops.SymmetricKronecker(A),
        )
        for kwargs in [{"assume_A": "sympos", "rtol": 10 ** -6}]:
            with self.subTest():
                # Define callback function to obtain search directions
                pls_iterates = []

                # pylint: disable=cell-var-from-loop
                def callback_iterates_PLS(
                    xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs
                ):
                    pls_iterates.append(xk.mean)

                # Probabilistic linear solver
                xhat_pls, _, _, info_pls = linalg.problinsolve(
                    A=A,
                    b=b,
                    Ainv0=Ainv0,
                    A0=A0,
                    callback=callback_iterates_PLS,
                    **kwargs
                )
                pls_iters_arr = np.array([x0] + pls_iterates)

                self.assertAllClose(xhat_pls.mean, xhat_cg, rtol=10 ** -12)
                self.assertAllClose(pls_iters_arr, cg_iters_arr, rtol=10 ** -12)

    def test_prior_distributions(self):
        """The solver should automatically handle different types of prior
        information."""
        pass

    def test_iterative_covariance_trace_update(self):
        """The solver's returned value for the trace must match the actual trace of the
        solution covariance."""
        A, b, x_true = self.rbf_kernel_linear_system

        for calib_method in [None, 0, 1.0, "adhoc", "weightedmean", "gpkern"]:
            with self.subTest():
                x_est, Ahat, Ainvhat, info = linalg.problinsolve(
                    A=A, b=b, calibration=calib_method
                )
                self.assertAlmostEqual(
                    info["trace_sol_cov"],
                    x_est.cov.trace(),
                    msg="Iteratively computed trace not equal to trace of solution "
                    "covariance.",
                )

    def test_uncertainty_calibration_error(self):
        """Test if the available uncertainty calibration procedures affect the error of
        the returned solution."""
        tol = 10 ** -6
        A, b, x_true = self.rbf_kernel_linear_system

        for calib_method in [None, 0, "adhoc", "weightedmean", "gpkern"]:
            with self.subTest():
                x_est, Ahat, Ainvhat, info = linalg.problinsolve(
                    A=A, b=b, calibration=calib_method
                )
                self.assertLessEqual(
                    (x_true - x_est.mean).T @ A @ (x_true - x_est.mean),
                    tol,
                    msg="Estimated solution not sufficiently close to true solution.",
                )
