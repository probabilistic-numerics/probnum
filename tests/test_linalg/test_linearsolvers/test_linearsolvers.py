"""Tests for linear solvers."""

import unittest
from tests.testing import NumpyAssertions
import os

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from probnum.linalg import linearsolvers, linops
from probnum import prob


class LinearSolverTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for linear solvers."""

    def setUp(self):
        """Resources for tests."""

        # Poisson equation with Dirichlet conditions.
        # 
        #   - Laplace(u) = f    in the interior
        #              u = u_D  on the boundary
        # where
        #     u_D = 1 + x^2 + 2y^2
        #     f = -4
        # 
        # Linear system resulting from discretization on an elliptic grid.
        fpath = os.path.join(os.path.dirname(__file__), '../../resources')
        A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
        f = np.load(file=fpath + "/rhs_poisson.npy")
        self.poisson_linear_system = A, f

        # Probabilistic linear solvers
        self.problinsolvers = [linearsolvers.problinsolve]  # , linearsolvers.bayescg]

        # Matrix-based linear solvers
        self.matblinsolvers = [linearsolvers.problinsolve]

        # Solution-based linear solvers
        self.solblinsolvers = [linearsolvers.bayescg]

    def test_dimension_mismatch(self):
        """Test whether linear solvers throw an exception for input with mismatched dimensions."""
        A = np.zeros(shape=[3, 3])
        b = np.zeros(shape=[4])
        x0 = np.zeros(shape=[1])
        for plinsolve in [linearsolvers.problinsolve, linearsolvers.bayescg]:
            with self.subTest():
                with self.assertRaises(ValueError, msg="Invalid input formats should raise a ValueError."):
                    # A, b dimension mismatch
                    plinsolve(A=A, b=b)
                    # A, x0 dimension mismatch
                    plinsolve(A=A, b=np.zeros(A.shape[0]), x0=x0)
                    # A not square
                    plinsolve(A=np.zeros([3, 4]), b=np.zeros(A.shape[0]),
                              x0=np.zeros(shape=[A.shape[1]]))
                    # A inverse not square
                    plinsolve(A=A, b=np.zeros(A.shape[0]),
                              Ainv=np.zeros([2, 3]),
                              x0=np.zeros(shape=[A.shape[1]]))
                    # A, Ainv dimension mismatch
                    plinsolve(A=A, b=np.zeros(A.shape[0]),
                              Ainv=np.zeros([2, 2]),
                              x0=np.zeros(shape=[A.shape[1]]))

    # TODO: Write linear systems as parameters and test for output properties separately to run all combinations

    def test_randvar_output(self):
        """Probabilistic linear solvers output random variables."""
        np.random.seed(1)
        n = 10
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        b = np.random.rand(n)
        for plinsolve in self.problinsolvers:
            with self.subTest():
                x, A, Ainv, _ = plinsolve(A=A, b=b)
                for rv in [x, A, Ainv]:
                    self.assertIsInstance(rv, prob.RandomVariable,
                                          msg="Output of probabilistic linear solver is not a random variable.")

    def test_symmetric_posterior_params(self):
        """Test whether posterior parameters are symmetric."""
        np.random.seed(1)
        n = 10
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        b = np.random.rand(n)

        for matblinsolve in self.matblinsolvers:
            with self.subTest():
                _, _, Ainv, _ = matblinsolve(A=A, b=b)
                Ainv_mean = Ainv.mean().todense()
                Ainv_cov_A = Ainv.cov().A.todense()
                Ainv_cov_B = Ainv.cov().B.todense()
                self.assertAllClose(Ainv_mean,
                                    Ainv_mean.T, rtol=1e-6)
                self.assertAllClose(Ainv_cov_A,
                                    Ainv_cov_B, rtol=1e-6)
                self.assertAllClose(Ainv_cov_A,
                                    Ainv_cov_A.T, rtol=1e-6)

    def test_zero_rhs(self):
        """Linear system with zero right hand side."""
        np.random.seed(1234)
        n = 10
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        b = np.zeros(n)
        tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

        for plinsolve in self.problinsolvers:
            with self.subTest():
                for tol in tols:
                    x, _, _, info = plinsolve(A=A, b=b, atol=tol)
                    self.assertAllClose(x.mean(), 0, atol=1e-15)

    def test_multiple_rhs(self):
        """Linear system with matrix right hand side."""
        np.random.seed(42)
        n = 10
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        B = np.random.rand(10, 5)

        for plinsolve in self.problinsolvers:
            with self.subTest():
                x, _, _, info = plinsolve(A=A, b=B)
                assert x.shape == B.shape, "Shape of solution and right hand side do not match."

    def test_spd_matrix(self):
        """Random spd matrix."""
        np.random.seed(1234)
        n = 40
        A = np.random.rand(n, n)
        A = 0.5 * (A + A.T) + n * np.eye(n)
        b = np.random.rand(n)
        x = np.linalg.solve(A, b)

        for plinsolve in self.problinsolvers:
            with self.subTest():
                x_solver, _, _, info = plinsolve(A=A, b=b)
                self.assertAllClose(x_solver.mean(), x, rtol=1e-4)

    # TODO: run this test for a set of different linear systems
    def test_sparse_poisson(self):
        """(Sparse) linear system from Poisson PDE with boundary conditions."""
        A, f = self.poisson_linear_system
        u = scipy.sparse.linalg.spsolve(A=A, b=f)

        for plinsolve in self.problinsolvers:
            with self.subTest():
                u_solver, Ahat, Ainvhat, info = plinsolve(A=A, b=f)
                self.assertAllClose(u_solver.mean(), u, rtol=1e-5,
                                    msg="Solution from probabilistic linear solver does" +
                                        " not match scipy.sparse.linalg.spsolve.")

    # def test_solution_equivalence(self):
    #     """The induced distributions on the solution should match the estimated solution distributions: E[x] = E[A^-1] b"""
    #     A, f = self.poisson_linear_system
    # 
    #     for matblinsolve in self.matblinsolvers:
    #         with self.subTest():
    #             # Solve linear system
    #             u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f)
    # 
    #             # E[x] = E[A^-1] b
    #             self.assertAllClose(u_solver.mean(), (Ainvhat @ f[:, None]).mean().ravel(), rtol=1e-5,
    #                                 msg="Solution from matrix-based probabilistic linear solver does not match the " +
    #                                     "estimated inverse, i.e. u =/= Ainv @ b ")

    def test_posterior_distribution_parameters(self):
        """Compute the posterior parameters of the matrix-based probabilistic linear solvers directly and compare."""
        # Initialization
        A, f = self.poisson_linear_system
        S = []  # search directions
        Y = []  # observations

        # Priors
        H0 = linops.Identity(A.shape[0])  # inverse prior mean
        A0 = linops.Identity(A.shape[0])  # prior mean
        WH0 = H0  # inverse prior Kronecker factor
        WA0 = A  # prior Kronecker factor
        covH = linops.SymmetricKronecker(WH0, WH0)
        covA = linops.SymmetricKronecker(WA0, WA0)
        Ahat0 = prob.RandomVariable(distribution=prob.Normal(mean=A0, cov=covA))
        Ainvhat0 = prob.RandomVariable(distribution=prob.Normal(mean=H0, cov=covH))

        # Define callback function to obtain search directions
        def callback_postparams(xk, Ak, Ainvk, sk, yk, alphak, resid):
            S.append(sk)
            Y.append(yk)

        # Solve linear system
        u_solver, Ahat, Ainvhat, info = linearsolvers.problinsolve(A=A, b=f, A0=Ahat0, Ainv0=Ainvhat0,
                                                                   callback=callback_postparams, calibrate=False)

        # Create arrays from lists
        S = np.squeeze(np.array(S)).T
        Y = np.squeeze(np.array(Y)).T

        # E[A] and E[A^-1]
        def posterior_mean(A0, WA0, S, Y):
            """Compute posterior mean of the symmetric probabilistic linear solver."""
            Delta = (Y - A0 @ S)
            U_T = np.linalg.solve(S.T @ (WA0 @ S), (WA0 @ S).T)
            U = U_T.T
            Ak = A0 + Delta @ U_T + U @ Delta.T - U @ S.T @ Delta @ U_T
            return Ak

        Ak = posterior_mean(A0.todense(), WA0, S, Y)
        Hk = posterior_mean(H0.todense(), WH0, Y, S)

        self.assertAllClose(Ahat.mean().todense(), Ak, rtol=1e-5,
                            msg="The matrix estimated by the probabilistic linear solver does not match the " +
                                "directly computed one.")
        self.assertAllClose(Ainvhat.mean().todense(), Hk, rtol=1e-5,
                            msg="The inverse matrix estimated by the probabilistic linear solver does not" +
                                "match the directly computed one.")

        # Cov[A] and Cov[A^-1]
        def posterior_cov_kronfac(WA0, S):
            """Compute the covariance symmetric Kronecker factor of the probabilistic linear solver."""
            U_AT = np.linalg.solve(S.T @ (WA0 @ S), (WA0 @ S).T)
            covfac = WA0 @ (np.identity(np.shape(WA0)[0]) - S @ U_AT)
            return covfac

        A_covfac = posterior_cov_kronfac(WA0, S)
        H_covfac = posterior_cov_kronfac(WH0, Y)

        self.assertAllClose(Ahat.cov().A.todense(), A_covfac, rtol=1e-5,
                            msg="The covariance estimated by the probabilistic linear solver does not match the " +
                                "directly computed one.")
        self.assertAllClose(Ainvhat.cov().A.todense(), H_covfac, rtol=1e-5,
                            msg="The covariance estimated by the probabilistic linear solver does not" +
                                "match the directly computed one.")

    # def test_posterior_covariance_posdef(self):
    #     """Posterior covariances of the output must be positive (semi-) definite."""
    #     # Initialization
    #     A, f = self.poisson_linear_system
    #     
    #     for matblinsolve in self.matblinsolvers:
    #         with self.subTest():
    #             # Solve linear system
    #             u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f)
    #         
    #             # Check positive definiteness
    #             self.assertTrue(np.linalg.eigvals(Ahat.cov().A.todense()) >= 0,
    #                               msg="Covariance of A not positive semi-definite.")
    #             self.assertTrue(np.linalg.eigvals(Ainvhat.cov().A.todense()) >= 0,
    #                               msg="Covariance of Ainv not positive semi-definite.")

    def test_matrixprior(self):
        """Solve random linear system with a matrix-based linear solver."""
        np.random.seed(1)
        # Linear system
        n = 10
        A = np.random.rand(n, n)
        A = A.dot(A.T) + n * np.eye(n)  # Symmetrize and make diagonally dominant
        b = np.random.rand(n, 1)

        # Prior distributions on A
        covA = linops.SymmetricKronecker(A=np.eye(n), B=np.eye(n))
        Ainv0 = prob.RandomVariable(distribution=prob.Normal(mean=np.eye(n), cov=covA))

        for matblinsolve in self.matblinsolvers:
            with self.subTest():
                x, Ahat, Ainvhat, info = matblinsolve(A=A, Ainv0=Ainv0, b=b)
                xnp = np.linalg.solve(A, b).ravel()

                self.assertAllClose(x.mean(), xnp, rtol=1e-4,
                                    msg="Solution does not match np.linalg.solve.")

    def test_searchdir_conjugacy(self):
        """Search directions should remain A-conjugate up to machine precision, i.e. s_i^T A s_j = 0 for i != j."""
        searchdirs = []

        # Define callback function to obtain search directions
        def callback_searchdirs(xk, Ak, Ainvk, sk, yk, alphak, resid):
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
                self.assertAllClose(np.diag(np.diag(inner_prods)), inner_prods, atol=1e-7,
                                    msg="Search directions from solver are not A-conjugate.")

    def test_posterior_mean_CG_equivalency(self):
        """The probabilistic linear solver should recover CG iterates as a posterior mean for specific covariances."""
        pass

    def test_prior_distributions(self):
        """The solver should automatically handle different types of prior information."""
        pass


class NoisyLinearSolverTestCase(unittest.TestCase, NumpyAssertions):
    """Tests the probabilistic linear solver with noise functionality."""

    def setUp(self):
        """Resources for tests."""


    def test_optimal_scale(self):
        """Tests the computation of the optimal scale for the posterior covariance."""
        pass

