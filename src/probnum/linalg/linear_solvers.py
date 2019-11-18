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


class MatrixBasedConjugateGradients(ProbabilisticLinearSolver):
    """
    Conjugate Gradients using prior information on the matrix inverse.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information either
    on the matrix inverse :math:`H=A^{-1}` and outputs a posterior belief over :math:`H`. This code implements the
    method described in [1]_.

    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260

    Attributes
    ----------
    H_mean : array-like
        Mean of the Gaussian prior on :math:`H=A^{-1}`.
    H_cov_kronfac
        Kronecker factor of the covariance :math:`W_H \\otimes W_H` of the Gaussian prior on :math:`H=A^{-1}`.
    """

    def __init__(self, H_mean, H_cov_kronfac):
        self.H_mean = H_mean
        self.H_cov_kronfac = H_cov_kronfac

    def solve(self, A, b, maxiter=None, resid_tol=10 ** -6, reorth=False):
        """
        Solve the given linear system for symmetric, positive-definite :math:`A`.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        maxiter : int
            Maximum number of iterations.
        resid_tol : float
            Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
            terminates.
        reorth : bool, default=False
            Reorthogonalize search directions at each step with all previous ones.

        Returns
        -------

        """
        # Convert arguments
        A = np.asarray(A)  # make sure this doesn't de-sparsify the matrix, replace with more general type checking
        b = np.asarray(b)

        # Check linear system dimensionality
        _check_linear_system(A=A, b=b)

        # Setup
        n = len(b)
        if maxiter is None:
            maxiter = n * 10

        # Initialization
        iter = 0
        x = np.matmul(self.H_mean, b)
        resid = np.matmul(A, x) - b

        # Iteration
        while True:

            # Compute search direction
            search_dir = - np.matmul(self.H_mean, resid)

            if reorth:
                # Reorthogonalize to all previous search directions
                s = 0
            else:
                # Orthogonalization
                s = 0

            # Perform action and observe
            obs = np.matmul(A, search_dir)

            # Compute step size
            step_size = - np.dot(search_dir, resid) / np.dot(search_dir, obs)

            # Step and residual update
            x = x + step_size * search_dir
            resid = resid + step_size * obs

            # Mean and covariance update

            # Stopping criteria
            if iter == maxiter - 1:
                # Maximum iterations
                break
            if np.linalg.norm(resid) < resid_tol:
                # Residual below error tolerance
                break

                # Uncertainty-based

        raise NotImplementedError("Not yet implemented.")
