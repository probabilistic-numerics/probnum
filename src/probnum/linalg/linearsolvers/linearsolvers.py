"""
Probabilistic numerical methods for solving linear systems.

This module provides routines to solve linear systems of equations in a Bayesian framework. This means that a prior
distribution over elements of the linear system can be provided and is updated with information collected by the solvers
to return a posterior distribution.
"""

import warnings

import numpy as np
import scipy.sparse

from probnum import prob
from probnum.linalg import linops
from probnum import utils
from probnum.linalg.linearsolvers.matrixbased import GeneralMatrixBasedSolver, NoisySymmetricMatrixBasedSolver, \
    SymmetricMatrixBasedSolver
from probnum.linalg.linearsolvers.solutionbased import SolutionBasedSolver


def problinsolve(A, b, A0=None, Ainv0=None, x0=None, assume_A="sympos", maxiter=None, atol=10 ** -6, rtol=10 ** -6,
                 callback=None, **kwargs):
    """
    Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`. They return a probability measure
    which quantifies uncertainty in the output arising from finite computational resources. This solver can take prior
    information either on the linear operator :math:`A` or its inverse :math:`H=A^{-1}` in
    the form of a random variable ``A0`` or ``Ainv0`` and outputs a posterior belief over :math:`A` or :math:`H`. This
    code implements the method described in [1]_ based on the work in [2]_.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A linear operator (or square matrix). Only matrix-vector products :math:`Av` are used internally.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`. For multiple right hand sides, ``nrhs`` problems are solved
        sequentially with the posteriors over the matrices acting as priors for subsequent solves.
    A0 : RandomVariable, shape=(n, n), optional
        Prior belief over the linear operator :math:`A` provided as a :class:`~probnum.prob.RandomVariable`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    x0 : array-like, shape=(n,) or (n, nrhs), optional
        Initial guess for the solution of the linear system. Will be ignored if ``Ainv`` is given.
    assume_A : str, default="sympos"
        Assumptions on the linear operator, which can influence solver choice or behavior. The available options are
        (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========

    maxiter : int, optional
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the dimension of :math:`A`.
    atol : float, optional
        Absolute residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{atol}`, the
        iteration terminates.
    rtol : float, optional
        Relative residual tolerance. If :math:`\\lVert r_i \\rVert  < \\text{rtol} \\lVert b \\rVert`, the
        iteration terminates.
    callback : function, optional
        User-supplied function called after each iteration of the linear solver. It is called as
        ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid)`` and can be used to return quantities from the iteration. Note that
        depending on the function supplied, this can slow down the solver.
    kwargs : optional
        Keyword arguments passed onto the solver iteration.

    Returns
    -------
    x : RandomVariable, shape=(n,) or (n, nrhs)
        Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.
    A : RandomVariable, shape=(n,n)
        Posterior belief over the linear operator.
    Ainv : RandomVariable, shape=(n,n)
        Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
    info : dict
        Information on convergence of the solver.

    Raises
    ------
    ValueError
        If size mismatches detected or input matrices are not square.
    LinAlgError
        If the matrix ``A`` is singular.
    LinAlgWarning
        If an ill-conditioned input ``A`` is detected.

    Notes
    -----
    For a specific class of priors the probabilistic linear solver recovers the iterates of the conjugate gradient
    method as the posterior mean of the induced distribution on :math:`x=Hb`. The matrix-based view taken here
    recovers the solution-based inference of :func:`bayescg` [3]_.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View, *Statistics and Computing*, 2019

    See Also
    --------
    bayescg : Solve linear systems with prior information on the solution.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> n = 20
    >>> A = np.random.rand(n, n)
    >>> A = 0.5 * (A + A.T) + 5 * np.eye(n)
    >>> b = np.random.rand(n)
    >>> x, A, Ainv, info = problinsolve(A=A, b=b)
    >>> print(info["iter"])
    9
    """

    # Check linear system for type and dimension mismatch
    _check_linear_system(A=A, b=b, A0=A0, Ainv0=Ainv0, x0=x0)

    # Transform linear system components to random variables and linear operators
    A, b, A0, Ainv0, x0 = _preprocess_linear_system(A=A, b=b, A0=A0, Ainv0=Ainv0, x0=x0, assume_A=assume_A)

    # Parameter initialization
    n = A.shape[0]
    nrhs = b.shape[1]
    x = x0
    info = {}

    # Set convergence parameters
    if maxiter is None:
        maxiter = n * 10

    # Iteratively solve for multiple right hand sides (with posteriors as new priors)
    for i in range(nrhs):
        # Select and initialize solver
        linear_solver = _init_solver(A=A, b=utils.as_colvec(b[:, i]), A0=A0, Ainv0=Ainv0, x0=x, assume_A=assume_A)

        # Solve linear system
        x, A0, Ainv0, info = linear_solver.solve(maxiter=maxiter, atol=atol, rtol=rtol, callback=callback, **kwargs)

    # Return Ainv @ b for multiple rhs
    if nrhs > 1:
        x = Ainv0 @ b

    # Check solution and issue warnings (e.g. singular or ill-conditioned matrix)
    _check_solution(info=info)

    return x, A0, Ainv0, info


def bayescg(A, b, x0=None, maxiter=None, atol=None, rtol=None, callback=None):
    """
    Conjugate Gradients using prior information on the solution of the linear system.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information
    on the solution and outputs a posterior belief over :math:`x`. This code implements the
    method described in Cockayne et al. [1]_.

    Note that the solution-based view of BayesCG and the matrix-based view of :meth:`problinsolve` correspond [2]_.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.prob.RandomVariable`. If an array or linear operator are given, a prior distribution is
        chosen automatically.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    x0 : array-like or RandomVariable, shape=(n,) or or (n, nrhs)
        Prior belief over the solution of the linear system.
    maxiter : int
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the dimension of :math:`A`.
    atol : float, optional
        Absolute residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{atol}`, the
        iteration terminates.
    rtol : float, optional
        Relative residual tolerance. If :math:`\\lVert r_i \\rVert  < \\text{rtol} \\lVert b \\rVert`, the
        iteration terminates.
    callback : function, optional
        User-supplied function called after each iteration of the linear solver. It is called as
        ``callback(xk, sk, yk, alphak, resid)`` and can be used to return quantities from the iteration. Note that
        depending on the function supplied, this can slow down the solver.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012
    .. [2] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View, *Statistics and Computing*, 2019

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    """
    # Check linear system for type and dimension mismatch
    _check_linear_system(A=A, b=b, x0=x0)

    # Transform linear system types to random variables and linear operators
    # and set a default prior if not specified
    A, b, _, _, x = _preprocess_linear_system(A=A, b=b, x0=x0)

    # Set default convergence parameters
    n = A.shape[0]
    if maxiter is None:
        maxiter = n * 10

    # Solve linear system
    x, _, _, info = SolutionBasedSolver(A=A, b=b, x=x0).solve(maxiter=maxiter, atol=atol, rtol=rtol)

    # Check solution and issue warnings (e.g. singular or ill-conditioned matrix)
    _check_solution(info=info)

    return x, info


def _check_linear_system(A, b, A0=None, Ainv0=None, x0=None):
    """
    Check linear system compatibility.

    Raises an exception if the input arguments are not of the right type or not compatible.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable
        Linear operator.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : RandomVariable, shape=(n,n)
        Random variable representing the prior belief over the linear operator :math:`A`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs), optional
        Initial guess for the solution of the linear system. Will be ignored if ``Ainv`` is given.

    Raises
    ------
    ValueError
        If type or size mismatches detected or inputs ``A`` and ``Ainv`` are not square.
    """
    # Check types
    linop_types = (
        np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator, prob.RandomVariable)
    vector_types = (np.ndarray, scipy.sparse.spmatrix, prob.RandomVariable)
    if not isinstance(A, linop_types):
        raise ValueError(
            "A must be either an array, a linear operator or a RandomVariable.")
    if not isinstance(b, vector_types):
        raise ValueError("The right hand side must be a (sparse) array.")
    if A0 is not None and not isinstance(A0, prob.RandomVariable):
        raise ValueError(
            "The prior belief over A must be a RandomVariable.")
    if Ainv0 is not None and not isinstance(Ainv0, linop_types):
        raise ValueError(
            "The inverse of A must be either an array, a linear operator or a RandomVariable of either.")
    if x0 is not None and not isinstance(x0, vector_types):
        raise ValueError("The initial guess for the solution must be a (sparse) array.")

    # Prior distribution mismatch
    if ((isinstance(A0, prob.RandomVariable) or isinstance(Ainv0, prob.RandomVariable)) and
            isinstance(x0, prob.RandomVariable)):
        raise ValueError("Cannot specify distributions on the linear operator and the solution simultaneously.")

    # Dimension mismatch
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and b must match.")
    if Ainv0 is not None and A.shape != Ainv0.shape:
        raise ValueError("Dimension mismatch. The dimensions of A and Ainv0 must match.")
    if A0 is not None and A.shape != A0.shape:
        raise ValueError("Dimension mismatch. The dimensions of A and A0 must match.")
    if x0 is not None and A.shape[1] != x0.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and x0 must match.")

    # Square matrices
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A0 is not None and A0.shape[0] != A0.shape[1]:
        raise ValueError("A0 must be square.")
    if Ainv0 is not None and Ainv0.shape[0] != Ainv0.shape[1]:
        raise ValueError("The inverse of A must be square.")


def _preprocess_linear_system(A, b, assume_A, A0=None, Ainv0=None, x0=None):
    """
    Transform the linear system to linear operator and random variable form.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable
        A square matrix, linear operator or random variable representing the prior belief over :math:`A`.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    assume_A : str, default="sympos"
        Assumptions on the linear operator, which can influence solver choice or behavior. The available options are
        (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========

    A0 : RandomVariable, shape=(n,n)
        Random variable representing the prior belief over the linear operator :math:`A`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n)
        Optional. A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if ``Ainv`` is given.

    Returns
    -------
    A : RandomVariable, shape=(n,n)
        Prior belief over the linear operator :math:`A`.
    b : array-like, shape=(n,) or (n, nrhs)
        Right-hand-side of the linear system.
    A0 : RandomVariable, shape=(n,n)
        Prior belief over the linear operator :math:`A`.
    Ainv0 : RandomVariable, shape=(n,n)
        Prior belief over the linear operator inverse :math:`H=A^{-1}`.
    x : array-like or RandomVariable, shape=(n,) or (n, nrhs)
        Prior belief over the solution :math:`x` to the linear system.
    """
    # Choose matrix based view if not clear from arguments
    if (Ainv0 is not None or A0 is not None) and x0 is not None:
        warnings.warn(
            "Cannot use prior information on both the matrix (inverse) and the solution. The latter will be ignored.")
        x = None
    else:
        x = x0

    # Check matrix assumptions for correctness
    assume_A = assume_A.lower()
    _assume_A_tmp = assume_A
    for allowed_str in ["gen", "sym", "pos", "noise"]:
        _assume_A_tmp = _assume_A_tmp.replace(allowed_str, "")
    if _assume_A_tmp != "":
        raise ValueError('\'{}\' contains unrecognized linear operator assumptions.'.format(assume_A))

    # Choose priors for A and Ainv if not specified, based on matrix assumptions in "assume_A"
    if "sym" in assume_A and "pos" in assume_A and "noise" not in assume_A:
        # No priors specified
        if A0 is None and Ainv0 is None:
            dist = prob.Normal(mean=linops.Identity(shape=A.shape[0]),
                               cov=linops.SymmetricKronecker(
                                   linops.Identity(shape=A.shape[0])))
            Ainv0 = prob.RandomVariable(distribution=dist)

            dist = prob.Normal(mean=linops.Identity(shape=A.shape[0]),
                               cov=linops.SymmetricKronecker(
                                   linops.Identity(shape=A.shape[0])))
            A0 = prob.RandomVariable(distribution=dist)
        # Only prior on Ainv specified
        elif A0 is None and Ainv0 is not None:
            try:
                if isinstance(Ainv0, prob.RandomVariable):
                    A0_mean = Ainv0.mean().inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                A0_mean = np.linalg.inv(Ainv0.mean())
            except NotImplementedError:
                A0_mean = linops.Identity(A.shape[0])
                warnings.warn(message="Prior specified only for Ainv. Automatic prior mean inversion not implemented, "
                                      + "falling back to standard normal prior.")
            # hereditary positive definiteness
            A0_covfactor = A

            dist = prob.Normal(mean=A0_mean,
                               cov=linops.SymmetricKronecker(A=A0_covfactor))
            A0 = prob.RandomVariable(distribution=dist)
        # Only prior on A specified
        if A0 is not None and Ainv0 is None:
            try:
                if isinstance(A0, prob.RandomVariable):
                    Ainv0_mean = A0.mean().inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                                      "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                Ainv0_mean = np.linalg.inv(A0.mean())
            except NotImplementedError:
                Ainv0_mean = linops.Identity(A.shape[0])
                warnings.warn(message="Prior specified only for Ainv. " +
                                      "Automatic prior mean inversion failed, falling back to standard normal prior.")
            # (non-symmetric) posterior correspondence
            Ainv0_covfactor = Ainv0_mean

            dist = prob.Normal(mean=Ainv0_mean,
                               cov=linops.SymmetricKronecker(A=Ainv0_covfactor))
            Ainv0 = prob.RandomVariable(distribution=dist)

    elif "sym" in assume_A:
        raise NotImplementedError
    elif "pos" in assume_A:
        raise NotImplementedError
    elif "gen" in assume_A:
        raise NotImplementedError

    # TODO: Implement case where only a pre-conditioner is given as Ainv0
    # TODO: Automatic prior selection based on data scale, matrix trace, etc.

    # Transform linear system to correct dimensions
    b = utils.as_colvec(b)  # (n,) -> (n, 1)
    if x0 is not None:
        x = utils.as_colvec(x0)  # (n,) -> (n, 1)

    assert (not (Ainv0 is None and x is None)), "Neither Ainv nor x are specified."

    return A, b, A0, Ainv0, x


def _init_solver(A, A0, Ainv0, b, x0, assume_A):
    """
    Selects and initializes probabilistic linear solver based on the prior information given.

    Parameters
    ----------
    A : RandomVariable, shape=(n,n)
        Random variable representing the prior belief over the linear operator :math:`A`.
    A0 : RandomVariable, shape=(n,n)
        Random variable representing the prior belief over the linear operator :math:`A`.
    Ainv0 : RandomVariable, shape=(n,n)
        Optional. Random variable representing the prior belief over the inverse :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if ``Ainv`` is given.
    assume_A : str
        Assumptions on the linear operator, which can influence solver choice or behavior. The available options are
        (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========

    Returns
    -------
    linear_solver : ProbabilisticLinearSolver
        A type of probabilistic linear solver implementing the solve method for linear systems.

    """
    # Select solution-based or matrix-based view
    if isinstance(A0, prob.RandomVariable) or isinstance(Ainv0, prob.RandomVariable):
        prior_info_view = "matrix"
    elif isinstance(x0, prob.RandomVariable):
        prior_info_view = "solution"
    else:
        raise ValueError("No prior information on A, Ainv or x specified.")

    # Combine assumptions on A with distribution assumptions
    if prior_info_view == "matrix":
        if isinstance(Ainv0.cov(), linops.SymmetricKronecker):
            if "noise" in assume_A:
                return NoisySymmetricMatrixBasedSolver(A=A, b=b, A_mean=A0.mean(), A_covfactor=A0.cov().A,
                                                       Ainv_mean=Ainv0.mean(), Ainv_covfactor=Ainv0.cov().A)
            else:
                return SymmetricMatrixBasedSolver(A=A, b=b, A_mean=A0.mean(), A_covfactor=A0.cov().A,
                                                  Ainv_mean=Ainv0.mean(), Ainv_covfactor=Ainv0.cov().A)
        elif isinstance(Ainv0.cov(), linops.Kronecker):
            return GeneralMatrixBasedSolver(A=A, b=b)
        else:
            raise NotImplementedError
    elif prior_info_view == "solution":
        return SolutionBasedSolver(A=A, b=b, x=x0)
    else:
        raise NotImplementedError


def _check_solution(info):
    """
    Check the solution of the linear system.

    Raises exceptions or warnings based on the properties of the linear system and the solver iteration.

    Parameters
    ----------
    info : dict
        Convergence information output by a probabilistic linear solver.

    Raises
    ------
    LinAlgError
        If the matrix is singular.
    LinAlgWarning
        If an ill-conditioned input a is detected.
    """

    # Singular matrix
    # TODO: get info from solver
    # Ill-conditioned matrix A
    pass
