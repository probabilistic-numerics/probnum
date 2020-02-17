"""
Probabilistic numerical methods for solving linear systems.

This module provides routines to solve linear systems of equations in a Bayesian framework. This means that a prior
distribution over elements of the linear system can be provided and is updated with information collected by the solvers
to return a posterior distribution.
"""

import warnings
import abc

import numpy as np
import scipy.sparse
import GPy

from probnum import probability
from probnum.linalg import linear_operators
import probnum.utils as utils

__all__ = ["problinsolve"]  # , "bayescg"]


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
    code implements the method described in [1]_ based on the work in [2]_ and [3]_.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View, *Statistics and Computing*, 2019

    Notes
    -----
    For a specific class of priors the probabilistic linear solver recovers the iterates of the conjugate gradient
    method as the posterior mean of the induced distribution on :math:`x=Hb`.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A square matrix or linear operator.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`. For multiple right hand sides, ``nrhs`` problems are solved
        sequentially with the posteriors over the matrices acting as priors for subsequent solves.
    A0 : RandomVariable, shape=(n, n), optional
        Prior belief over the linear operator :math:`A` provided as a :class:`~probnum.probability.RandomVariable`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`. This can be viewed as taking the form of a pre-conditioner. If an array or linear operator is
        given, a prior distribution is chosen automatically.
    x0 : array-like, shape=(n,) or (n, nrhs), optional
        Initial guess for the solution of the linear system. Will be ignored if ``Ainv`` is given.
    assume_A : str, default="sympos"
        Assumptions on the matrix, which can influence solver choice or behavior. The available options are

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         symmetric pos. def.  ``sympos``
        ====================  =========

        If ``A`` or ``Ainv`` are random variables, then the encoded assumptions in the distribution are used
        automatically.
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
    kwargs :
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
    10
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
        linear_solver = _init_solver(A=A, b=utils.as_colvec(b[:, i]), A0=A0, Ainv0=Ainv0, x0=x)

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

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012
    .. [2] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View, *Statistics and Computing*, 2019

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.probability.RandomVariable`. If an array or linear operator are given, a prior distribution is
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
    x, _, _, info = _BayesCG(A=A, b=b, x=x0).solve(maxiter=maxiter, atol=atol, rtol=rtol)

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
        np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator, probability.RandomVariable)
    vector_types = (np.ndarray, scipy.sparse.spmatrix, probability.RandomVariable)
    if not isinstance(A, linop_types):
        raise ValueError(
            "A must be either an array, a linear operator or a RandomVariable.")
    if not isinstance(b, vector_types):
        raise ValueError("The right hand side must be a (sparse) array.")
    if A0 is not None and not isinstance(A0, probability.RandomVariable):
        raise ValueError(
            "The prior belief over A must be a RandomVariable.")
    if Ainv0 is not None and not isinstance(Ainv0, linop_types):
        raise ValueError(
            "The inverse of A must be either an array, a linear operator or a RandomVariable of either.")
    if x0 is not None and not isinstance(x0, vector_types):
        raise ValueError("The initial guess for the solution must be a (sparse) array.")

    # Prior distribution mismatch
    if ((isinstance(A0, probability.RandomVariable) or isinstance(Ainv0, probability.RandomVariable)) and
            isinstance(x0, probability.RandomVariable)):
        raise ValueError("Cannot specify distribution on the linear operator and the solution simultaneously.")

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
        Assumptions on the matrix, which can influence solver choice or behavior. The available options are

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         symmetric pos. def.  ``sympos``
        ====================  =========

        If ``A`` or ``Ainv`` are random variables, then the encoded assumptions in the distribution are used
        automatically.
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

    # Check matrix assumptions
    if assume_A not in ["gen", "sym", "pos", "sympos"]:
        raise ValueError('\'{}\' is not a recognized linear operator assumption.'.format(assume_A))

    # Choose priors for A and Ainv if not specified, based on matrix assumptions in "assume_A"
    if assume_A == "sympos":
        # No priors specified
        if A0 is None and Ainv0 is None:
            dist = probability.Normal(mean=linear_operators.Identity(shape=A.shape[0]),
                                      cov=linear_operators.SymmetricKronecker(
                                          linear_operators.Identity(shape=A.shape[0]),
                                          linear_operators.Identity(shape=A.shape[0])))
            Ainv0 = probability.RandomVariable(distribution=dist)

            dist = probability.Normal(mean=linear_operators.Identity(shape=A.shape[0]),
                                      cov=linear_operators.SymmetricKronecker(
                                          linear_operators.Identity(shape=A.shape[0]),
                                          linear_operators.Identity(shape=A.shape[0])))
            A0 = probability.RandomVariable(distribution=dist)
        # Only prior on Ainv specified
        elif A0 is None and Ainv0 is not None:
            try:
                if isinstance(Ainv0, probability.RandomVariable):
                    A0_mean = Ainv0.mean().inv()
                else:
                    A0_mean = Ainv0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                              "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                A0_mean = np.linalg.inv(Ainv0.mean())
            except NotImplementedError:
                A0_mean = linear_operators.Identity(A.shape[0])
                warnings.warn(message="Prior specified only for Ainv. Automatic prior mean inversion not implemented, "
                                      + "falling back to standard normal prior.")
            # hereditary positive definiteness
            A0_covfactor = A

            dist = probability.Normal(mean=A0_mean,
                                      cov=linear_operators.SymmetricKronecker(A=A0_covfactor, B=A0_covfactor))
            A0 = probability.RandomVariable(distribution=dist)
        # Only prior on A specified
        if A0 is not None and Ainv0 is None:
            try:
                if isinstance(A0, probability.RandomVariable):
                    Ainv0_mean = A0.mean().inv()
                else:
                    Ainv0_mean = A0.inv()
            except AttributeError:
                warnings.warn(message="Prior specified only for Ainv. Inverting prior mean naively. " +
                              "This operation is computationally costly! Specify an inverse prior (mean) instead.")
                Ainv0_mean = np.linalg.inv(A0.mean())
            except NotImplementedError:
                Ainv0_mean = linear_operators.Identity(A.shape[0])
                warnings.warn(message="Prior specified only for Ainv. " +
                                      "Automatic prior mean inversion failed, falling back to standard normal prior.")
            # (non-symmetric) posterior correspondence
            Ainv0_covfactor = Ainv0_mean

            dist = probability.Normal(mean=Ainv0_mean,
                                      cov=linear_operators.SymmetricKronecker(A=Ainv0_covfactor, B=Ainv0_covfactor))
            Ainv0 = probability.RandomVariable(distribution=dist)

    elif assume_A == "sym":
        raise NotImplementedError
    elif assume_A == "pos":
        raise NotImplementedError
    elif assume_A == "gen":
        # TODO: Implement case where only a pre-conditioner is given as Ainv0
        # TODO: Automatic prior selection based on data scale, matrix trace, etc.
        raise NotImplementedError

    # Transform linear system to correct dimensions
    b = utils.as_colvec(b)  # (n,) -> (n, 1)
    if x0 is not None:
        x = utils.as_colvec(x0)  # (n,) -> (n, 1)

    assert (not (Ainv0 is None and x is None)), "Neither Ainv nor x are specified."

    return A, b, A0, Ainv0, x


def _init_solver(A, A0, Ainv0, b, x0):
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

    Returns
    -------
    linear_solver : _ProbabilisticLinearSolver
        A type of probabilistic linear solver implementing the solve method for linear systems.

    """
    # Select solution-based or matrix-based view
    if isinstance(A0, probability.RandomVariable) or isinstance(Ainv0, probability.RandomVariable):
        prior_info_view = "matrix"
    elif isinstance(x0, probability.RandomVariable):
        prior_info_view = "solution"
    else:
        raise ValueError("No prior information on A, Ainv or x specified.")

    # Combine assumptions on A with distribution assumptions
    if prior_info_view == "matrix":
        if isinstance(Ainv0.cov(), linear_operators.SymmetricKronecker):
            return _SymmetricMatrixSolver(A=A, b=b, A_mean=A0.mean(), A_covfactor=A0.cov().A,
                                          Ainv_mean=Ainv0.mean(), Ainv_covfactor=Ainv0.cov().A)
        elif isinstance(Ainv0.cov(), linear_operators.Kronecker):
            return _GeneralMatrixSolver(A=A, b=b)
        else:
            raise NotImplementedError
    elif prior_info_view == "solution":
        return _BayesCG(A=A, b=b, x=x0)
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


class _ProbabilisticLinearSolver(abc.ABC):
    """
    An abstract base class for probabilistic linear solvers.

    This class is designed to be subclassed with new (probabilistic) linear solvers, which implement a ``.solve()``
    method. Objects of this type are instantiated in wrapper functions such as :meth:``problinsolve``.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.probability.RandomVariable`. If an array or linear operator is given, a prior distribution is
        chosen automatically.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def _check_convergence(self, iter, maxiter, resid, atol, rtol):
        """
        Check convergence of a linear solver.

        Evaluates a set of convergence criteria based on its input arguments to decide whether the iteration has converged.

        Parameters
        ----------
        iter : int
            Current iteration of solver.
        maxiter : int
            Maximum number of iterations
        resid : array-like
            Residual vector :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert` of the current iteration.
        atol : float
            Absolute residual tolerance. Stops if :math:`\\lVert r_i \\rVert < \\text{atol}`.
        rtol : float
            Relative residual tolerance. Stops if :math:`\\lVert r_i \\rVert < \\text{rtol} \\lVert b \\rVert`.

        Returns
        -------
        has_converged : bool
            True if the method has converged.
        convergence_criterion : str
            Convergence criterion which caused termination.
        """
        # maximum iterations
        if iter >= maxiter:
            warnings.warn(message="Iteration terminated. Solver reached the maximum number of iterations.")
            return True, "maxiter"
        # residual below error tolerance
        elif np.linalg.norm(resid) <= atol:
            return True, "resid_atol"
        elif np.linalg.norm(resid) <= rtol * np.linalg.norm(self.b):
            return True, "resid_rtol"
        # uncertainty-based
        # todo: based on posterior contraction
        else:
            return False, ""

    def solve(self, callback=None, **kwargs):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It is called as
            ``callback(xk, sk, yk, alphak, resid, **kwargs)`` and can be used to return quantities from the iteration.
            Note that depending on the function supplied, this can slow down the solver.
        kwargs
            Key-word arguments adjusting the behaviour of the ``solve`` iteration. These are usually convergence
            criteria.

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

        """
        raise NotImplementedError


class _GeneralMatrixSolver(_ProbabilisticLinearSolver):
    """
    Solver iteration of the (general) probabilistic linear solver.

    Parameters
    ----------
    """

    def __init__(self, A, b):
        raise NotImplementedError
        # super().__init__(A=A, b=b)

    def solve(self, callback=None, maxiter=None, atol=None):
        raise NotImplementedError


class _SymmetricMatrixSolver(_ProbabilisticLinearSolver):
    """
    Solver iteration of the symmetric probabilistic linear solver.

    Implements the solve iteration of the symmetric matrix-based probabilistic linear solver described in [1]_ and [2]_.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A_mean : array-like or LinearOperator
        Mean of the prior distribution on the linear operator :math:`A`.
    A_covfactor : array-like or LinearOperator
        The Kronecker factor :math:`W_A` of the covariance :math:`\\operatorname{Cov}(A) = W_A \\otimes_s W_A` of
        :math:`A`.
    Ainv_mean : array-like or LinearOperator
        Mean of the prior distribution on the linear operator :math:`A^{-1}`.
    Ainv_covfactor : array-like or LinearOperator
        The Kronecker factor :math:`W_H` of the covariance :math:`\\operatorname{Cov}(H) = W_H \\otimes_s W_H` of
        :math:`H = A^{-1}`.

    Returns
    -------
    A : RandomVariable
        Posterior belief over the linear operator.
    Ainv : RandomVariable
        Posterior belief over the inverse linear operator.
    x : RandomVariable
        Posterior belief over the solution of the linear system.
    info : dict
        Information about convergence and the solution.
    """

    def __init__(self, A, b, A_mean, A_covfactor, Ainv_mean, Ainv_covfactor):
        self.A_mean = A_mean
        self.A_covfactor = A_covfactor
        self.Ainv_mean = Ainv_mean
        self.Ainv_covfactor = Ainv_covfactor
        self.x = Ainv_mean @ b
        self.S = []
        self.Y = []
        self.sy = []
        super().__init__(A=A, b=b)

    def _calibrate_uncertainty(self):
        """
        Calibrate uncertainty based on the Rayleigh coefficients

        A regression model for the log-Rayleigh coefficient is built based on the collected observations. The degrees of
        freedom in the covariance of A and H are set according to the predicted log-Rayleigh coefficient for the
        remaining unexplored dimensions.
        """
        # Transform to arrays
        _sy = np.squeeze(np.array(self.sy))
        _S = np.squeeze(np.array(self.S)).T
        _Y = np.squeeze(np.array(self.Y)).T

        if self.iter_ > 5:  # only calibrate if enough iterations for a regression model have been performed
            # Rayleigh quotient
            iters = np.arange(self.iter_)
            logR = np.log(_sy) - np.log(np.einsum('ij,ij->j', _S, _S))

            # Least-squares fit for y intercept
            x_mean = np.mean(iters)
            y_mean = np.mean(logR)
            beta1 = np.sum((iters - x_mean) * (logR - y_mean)) / np.sum((iters - x_mean) ** 2)
            beta0 = y_mean - beta1 * x_mean

            # Log-Rayleigh quotient regression
            mf = GPy.mappings.linear.Linear(1, 1)
            k = GPy.kern.RBF(input_dim=1, lengthscale=1, variance=1)
            m = GPy.models.GPRegression(iters[:, None], (logR - beta0)[:, None], kernel=k, mean_function=mf)
            m.optimize(messages=False)

            # Predict Rayleigh quotient
            remaining_dims = np.arange(self.iter_, self.A.shape[0])[:, None]
            GP_pred = m.predict(remaining_dims)
            R_pred = np.exp(GP_pred[0].ravel() + beta0)

            # Set scale
            Phi = linear_operators.ScalarMult(shape=self.A.shape, scalar=np.asscalar(np.mean(R_pred)))
            Psi = linear_operators.ScalarMult(shape=self.A.shape, scalar=np.asscalar(np.mean(1 / R_pred)))

        else:
            Phi = None
            Psi = None

        return Phi, Psi

    def _create_output_randvars(self, S=None, Y=None, Phi=None, Psi=None):
        """Return output random variables x, A, Ainv from their means and covariances."""

        _A_covfactor = self.A_covfactor
        _Ainv_covfactor = self.Ainv_covfactor

        # Set degrees of freedom based on uncertainty calibration in unexplored space
        if Phi is not None:
            def _mv(x):
                def _I_S_fun(x):
                    return x - S @ np.linalg.solve(S.T @ S, S.T @ x)

                return _I_S_fun(Phi @ _I_S_fun(x))

            I_S_Phi_I_S_op = linear_operators.LinearOperator(shape=self.A.shape, matvec=_mv)
            _A_covfactor = self.A_covfactor + I_S_Phi_I_S_op

        if Psi is not None:
            def _mv(x):
                def _I_Y_fun(x):
                    return x - Y @ np.linalg.solve(Y.T @ Y, Y.T @ x)

                return _I_Y_fun(Psi @ _I_Y_fun(x))

            I_Y_Psi_I_Y_op = linear_operators.LinearOperator(shape=self.A.shape, matvec=_mv)
            _Ainv_covfactor = self.Ainv_covfactor + I_Y_Psi_I_Y_op

        # Create output random variables
        A = probability.RandomVariable(shape=self.A_mean.shape,
                                       dtype=float,
                                       distribution=probability.Normal(mean=self.A_mean,
                                                                       cov=linear_operators.SymmetricKronecker(
                                                                           A=_A_covfactor)))
        cov_Ainv = linear_operators.SymmetricKronecker(A=_Ainv_covfactor)
        Ainv = probability.RandomVariable(shape=self.Ainv_mean.shape,
                                          dtype=float,
                                          distribution=probability.Normal(mean=self.Ainv_mean, cov=cov_Ainv))
        # Induced distribution on x via Ainv
        # Exp = x = A^-1 b, Cov = 1/2 (W b'Wb + Wbb'W)
        Wb = _Ainv_covfactor @ self.b
        bWb = np.squeeze(Wb.T @ self.b)

        def _mv(x):
            return 0.5 * (bWb * _Ainv_covfactor @ x + Wb @ (Wb.T @ x))

        cov_op = linear_operators.LinearOperator(shape=np.shape(_Ainv_covfactor), dtype=float,
                                                 matvec=_mv, matmat=_mv)

        x = probability.RandomVariable(shape=(self.A_mean.shape[0],),
                                       dtype=float,
                                       distribution=probability.Normal(mean=self.x.ravel(), cov=cov_op))
        return x, A, Ainv

    def _mean_update(self, u, v):
        """Linear operator implementing the symmetric rank 2 mean update (+= uv' + vu')."""

        def mv(x):
            return u @ (v.T @ x) + v @ (u.T @ x)

        def mm(X):
            return u @ (v.T @ X) + v @ (u.T @ X)

        return linear_operators.LinearOperator(shape=self.A_mean.shape, matvec=mv, matmat=mm)

    def _covariance_update(self, u, Ws):
        """Linear operator implementing the symmetric rank 2 covariance update (-= Ws u^T)."""

        def mv(x):
            return u @ (Ws.T @ x)

        def mm(X):
            return u @ (Ws.T @ X)

        return linear_operators.LinearOperator(shape=self.A_mean.shape, matvec=mv, matmat=mm)

    def solve(self, callback=None, maxiter=None, atol=None, rtol=None, calibrate=True):
        # initialization
        self.iter_ = 0
        resid = self.A @ self.x - self.b

        # iteration with stopping criteria
        while True:
            # check convergence
            _has_converged, _conv_crit = self._check_convergence(iter=self.iter_, maxiter=maxiter,
                                                                 resid=resid, atol=atol, rtol=rtol)
            if _has_converged:
                break

            # compute search direction (with implicit reorthogonalization)
            search_dir = - self.Ainv_mean @ resid
            self.S.append(search_dir)

            # perform action and observe
            obs = self.A @ search_dir
            self.Y.append(obs)

            # compute step size
            sy = search_dir.T @ obs
            step_size = - (search_dir.T @ resid) / sy
            self.sy.append(sy)

            # step and residual update
            self.x = self.x + step_size * search_dir
            resid = resid + step_size * obs

            # (symmetric) mean and covariance updates
            Vs = self.A_covfactor @ search_dir
            delta_A = obs - self.A_mean @ search_dir
            u_A = Vs / (search_dir.T @ Vs)
            v_A = delta_A - 0.5 * (search_dir.T @ delta_A) * u_A

            Wy = self.Ainv_covfactor @ obs
            delta_Ainv = search_dir - self.Ainv_mean @ obs
            u_Ainv = Wy / (obs.T @ Wy)
            v_Ainv = delta_Ainv - 0.5 * (obs.T @ delta_Ainv) * u_Ainv

            # rank 2 mean updates (+= uv' + vu')
            # TODO: Operator form may cause stack size issues for too many iterations
            self.A_mean = linear_operators.aslinop(self.A_mean) + self._mean_update(u=u_A, v=v_A)
            self.Ainv_mean = linear_operators.aslinop(self.Ainv_mean) + self._mean_update(u=u_Ainv, v=v_Ainv)

            # rank 1 covariance kronecker factor update (-= u_A(Vs)' and -= u_Ainv(Wy)')
            self.A_covfactor = linear_operators.aslinop(self.A_covfactor) - self._covariance_update(u=u_A, Ws=Vs)
            self.Ainv_covfactor = linear_operators.aslinop(self.Ainv_covfactor) - self._covariance_update(u=u_Ainv,
                                                                                                          Ws=Wy)

            # iteration increment
            self.iter_ += 1

            # callback function used to extract quantities from iteration
            if callback is not None:
                # Phi, Psi = self._calibrate_uncertainty()
                xk, Ak, Ainvk = self._create_output_randvars(S=np.squeeze(np.array(self.S)).T,
                                                             Y=np.squeeze(np.array(self.Y)).T,
                                                             Phi=None,  # Phi,
                                                             Psi=None)  # Psi)
                callback(xk=xk, Ak=Ak, Ainvk=Ainvk, sk=search_dir, yk=obs, alphak=step_size, resid=resid)

        # Calibrate uncertainty
        if calibrate:
            Phi, Psi = self._calibrate_uncertainty()
        else:
            Phi = None
            Psi = None

        # Create output random variables
        x, A, Ainv = self._create_output_randvars(S=np.squeeze(np.array(self.S)).T,
                                                  Y=np.squeeze(np.array(self.Y)).T,
                                                  Phi=Phi,
                                                  Psi=Psi)

        # Log information on solution
        info = {
            "iter": self.iter_,
            "maxiter": maxiter,
            "resid_l2norm": np.linalg.norm(resid, ord=2),
            "conv_crit": _conv_crit,
            "matrix_cond": None  # TODO: matrix condition from solver (see scipy solvers)
        }

        return x, A, Ainv, info


class _BayesCG(_ProbabilisticLinearSolver):
    """
    Solver iteration of BayesCG.

    Implements the solve iteration of BayesCG [1]_.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        The square matrix or linear operator of the linear system.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.

    """

    def __init__(self, A, b, x):
        self.x = x
        super().__init__(A=A, b=b)

    def solve(self, callback=None, maxiter=None, atol=None, rtol=None):
        raise NotImplementedError
