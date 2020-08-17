"""
Solution-based probabilistic linear solvers.

Implementations of solution-based linear solvers which perform inference on the solution
of a linear system given linear observations.
"""

import warnings
import numpy as np

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
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    """

    def __init__(self, A, b, x0=None):
        self.x0 = x0
        super().__init__(A=A, b=b)

    def has_converged(self, iter, maxiter, resid=None, atol=None, rtol=None):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide
        whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of
            the current iteration.
        atol : float
            Absolute residual tolerance. Stops if
            :math:`\\lVert r_i \\rVert < \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if
            :math:`\\lVert r_i \\rVert < \\text{rtol} \\lVert b \\rVert`.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(
                "Iteration terminated. Solver reached the maximum number of iterations."
            )
            return True, "maxiter"
        # residual below error tolerance
        elif np.linalg.norm(resid) <= atol:
            return True, "resid_atol"
        elif np.linalg.norm(resid) <= rtol * np.linalg.norm(self.b):
            return True, "resid_rtol"
        else:
            return False, ""

    def solve(self, callback=None, maxiter=None, atol=None, rtol=None):
        raise NotImplementedError
