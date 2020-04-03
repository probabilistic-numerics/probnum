"""
Solution-based probabilistic linear solvers.

Implementations of solution-based linear solvers which perform inference on the solution of a linear system given linear
observations.
"""

from probnum.linalg.linearsolvers.matrixbased import ProbabilisticLinearSolver


class SolutionBasedSolver(ProbabilisticLinearSolver):
    """
    Solver iteration of BayesCG.

    Implements the solve iteration of the solution-based solver BayesCG [1]_.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012
    """

    def __init__(self, A, b, x):
        self.x = x
        super().__init__(A=A, b=b)

    def solve(self, callback=None, maxiter=None, atol=None, rtol=None):
        raise NotImplementedError
