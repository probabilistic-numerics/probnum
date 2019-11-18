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
    """

    def __init__(self, prior_mean_inverse, prior_cov_kron_fac):
        """

        Parameters
        ----------
        prior_mean_inverse
        prior_cov_kron_fac
        """

    def solve(self, A, b, maxiter=None, reorth=False):
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
        reorth : bool, default=False
            Reorthogonalize search directions at each step with all previous ones.

        Returns
        -------

        """
        # Convert arguments
        A = np.asarray(A)  # make sure this doesn't de-sparsify the matrix, replace with more general type checking
        b = np.asarray(b)

        # Check for correct dimensions
        _check_linear_system(A=A, b=b)

        # Setup
        n = len(b)
        if maxiter is None:
            maxiter = n * 10

        # Matrix-based Gaussian inference
        iter = 0
        while True:

            # Compute search direction

            # Reorthogonalize to all previous search directions
            if reorth:
                s = 0
            else:
                s = 0

            # Perform action and observe

            # Compute step size

            # Stopping conditions
            if iter == maxiter - 1:
                # Maximum iterations
                break

                # Residual below error tolerance

                # Uncertainty-based

        raise NotImplementedError("Not yet implemented.")
