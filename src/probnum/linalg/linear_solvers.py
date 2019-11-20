"""Probabilistic numerical methods for solving linear systems"""

import abc
import numpy as np


class ProbabilisticLinearSolver(abc.ABC):
    """
    Probabilistic linear solvers infer solutions to linear systems in a Bayesian framework.

    Probabilistic numerical linear solvers infer solutions to problems of the form

    .. math:: Ax^*=b,

    where :math:`A \\in \\mathbb{R}^{m \\times n}` and :math:`b \\in \\mathbb{R}^{m}`. They output a probability measure
    which quantifies uncertainty in the solution.
    """

    @abc.abstractmethod
    def solve(self, A, b, maxiter=None):
        """
        Solve the given linear system.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------

        """
        pass


def _check_linear_system(A, b):
    """
    Check linear system dimensions.

    Raises an exception if the input arguments are not compatible.

    Parameters
    ----------
    A : array-like
        Matrix.
    b : array-like
        Right-hand side vector.

    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")


class MatrixBasedLinearSolver(ProbabilisticLinearSolver):
    """
    Probabilistic linear solver using prior information on the matrix inverse.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information either
    on the matrix inverse :math:`H=A^{-1}` and outputs a posterior belief over :math:`H`. For a specific prior choice
    this recovers the iterates of the conjugate gradient method. This code implements the method described in [1]_ and
    [2]_.

    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [2] Hennig, P. et al., Probabilistic Numerics, 2020

    Attributes
    ----------
    H_mean : array-like
        Mean of the Gaussian prior on :math:`H=A^{-1}`.
    H_cov_kronfac : array-like
        Kronecker factor of the covariance :math:`W_H \\otimes W_H` of the Gaussian prior on :math:`H=A^{-1}`.
    """

    def __init__(self, H_mean=None, H_cov_kronfac=None):
        # todo: initialize as identity operators (avoids having to specify dimensions at this point)
        self.H_mean = H_mean
        self.H_cov_kronfac = H_cov_kronfac

    def _check_convergence(self, iter, maxiter, resid, resid_tol):
        """

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of the current iteration.
        resid_tol : float
            Residual tolerance.

        Returns
        -------
        is_converged : bool
        """
        # maximum iterations
        if iter == maxiter - 1:
            return True
        # residual below error tolerance
        # todo: add / replace with relative tolerance
        elif np.linalg.norm(resid) < resid_tol:
            return True
        # uncertainty-based
        # todo: based on posterior contraction
        else:
            return False

    def solve(self, A, b, maxiter=None, resid_tol=10 ** -6, reorth=False):
        """
        Solve the given linear system for symmetric, positive-definite :math:`A`.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int, default=len(b)*10
            Maximum number of iterations.
        resid_tol : float
            Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
            terminates.
        reorth : bool, default=False
            Reorthogonalize search directions at each step with all previous ones.

        Returns
        -------

        """
        # todo: allow for linear operator A (via scipy.sparse.linalg.LinearOperator) and replace matrix multiplication
        # with sparse matrix multiplication for efficiency
        # todo: make sure for A-conjugate search directions (i.e. choice of W st. Y'WY diagonal) this is similarly
        #  efficient as CG

        # convert arguments
        # todo: make sure this doesn't de-sparsify the matrix, replace with more general type checking
        A = np.asarray(A)
        b = np.asarray(b)

        # check linear system dimensionality
        _check_linear_system(A=A, b=b)

        # setup
        n = len(b)
        if maxiter is None:
            maxiter = n * 10

        # initialization
        iter = 0
        x = np.matmul(self.H_mean, b)
        resid = np.matmul(A, x) - b

        # iteration
        while True:

            # compute search direction
            search_dir = - np.matmul(self.H_mean, resid)
            if reorth:
                # todo: reorthogonalize to all previous search directions (i.e. perform full inversion of MxM matrix?)
                raise NotImplementedError("Not yet implemented.")

            # perform action and observe
            obs = np.matmul(A, search_dir)

            # compute step size
            step_size = - np.dot(search_dir, resid) / np.dot(search_dir, obs)

            # step and residual update
            x = x + step_size * search_dir
            resid = resid + step_size * obs

            # mean and covariance (rank 2) update
            Ws = np.matmul(self.H_cov_kronfac, search_dir)
            delta = obs - np.matmul(self.H_mean, search_dir)
            u = Ws / np.dot(search_dir, Ws)
            udelta = np.outer(u, delta)
            self.H_mean = udelta + udelta.T - np.dot(delta, search_dir) * np.outer(u, u)
            self.H_cov_kronfac = self.H_cov_kronfac - np.outer(u, Ws)

            # stopping criteria
            if self._check_convergence(iter=iter, maxiter=maxiter, resid=resid, resid_tol=resid_tol):
                break

        return x, self.H_mean, self.H_cov_kronfac


class SolutionBasedConjugateGradients(ProbabilisticLinearSolver):
    """
    Conjugate Gradients using prior information on the solution of the linear system.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information
    on the solution and outputs a posterior belief over :math:`x`. This code implements the
    method described in [1]_.

    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012

    Attributes
    ----------
    x_mean : array-like
        Mean of the Gaussian prior on :math:`x`.
    x_cov : array-like
        Covariance :math:`\\Sigma` of the Gaussian prior on :math:`x`.
    """

    def __init__(self, x_mean=None, x_cov=None):
        self.x_mean = x_mean
        self.x_cov = x_cov

    def solve(self, A, b, maxiter=None):
        """
        Solve the given linear system.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------

        """

        # check linear system dimensionality
        _check_linear_system(A=A, b=b)

        raise NotImplementedError("Not yet implemented.")
