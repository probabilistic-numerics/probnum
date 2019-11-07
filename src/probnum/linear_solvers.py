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
    def solve(self, A, b, **kwargs):
        """
        Solve the given linear system.

        Parameters
        ----------
        A : array-like
            The matrix of the linear system.
        b : array-like
            The right-hand-side of the linear system.
        kwargs :
            Additional arguments passed on to the probabilistic linear solver.

        Returns
        -------

        """
        pass


class MatrixBasedConjugateGradients(ProbabilisticLinearSolver):
    """
    Conjugate Gradients using prior information on the matrix inverse.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information either
    on the matrix inverse :math:`H=A^{-1}` and outputs a posterior belief over :math:`H`. This code implements the
    method described in [1]_.

    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    """

    def solve(self, A, b, **kwargs):
        # Convert arguments
        A = np.asarray(A)
        b = np.asarray(b)

        # Check for dimension mismatch
        if np.shape(A)[0] != np.shape(b)[0]:
            raise ValueError("Dimension mismatch.")
        raise NotImplementedError("Not yet implemented.")

