"""
Probabilistic numerical methods for solving linear systems.

This module provides routines to solve linear systems of equations in a Bayesian framework. This means that a prior
distribution over elements of the linear system can be provided and is updated with information collected by the solvers
to return a posterior distribution.
"""

import warnings

import numpy as np
import scipy.sparse

import probnum.probability
import probnum.linalg.linear_operators
import probnum.utils

__all__ = ["problinsolve", "bayescg"]


def problinsolve(A, b, Ainv=None, x0=None, assume_A="sympos", maxiter=None, resid_tol=10 ** -6):
    """
    Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`. They return a probability measure
    which quantifies uncertainty in the output arising from finite computational resources.

    This solver can take prior information either on the linear operator :math:`A` or its inverse :math:`H=A^{-1}` and
    outputs a posterior belief over :math:`A` or :math:`H`. For a specific class of priors this recovers the iterates of
    the conjugate gradient method [1]_ as the posterior mean. This code implements the method described in [2]_, [3]_
    and [4]_.

    .. [1] Hestenes, M. R. and Stiefel E., Methods of Conjugate Gradients for Solving Linear Systems,
            *Journal of Research of the National Bureau of Standards*, 1952, 49 (6): 409
    .. [2] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [3] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [4] Hennig, P. and Osborne M. A., *Probabilistic Numerics. Computation as Machine Learning*, 2020, Cambridge
           University Press

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.probability.RandomVariable`. If an array or linear operator is given, a prior distribution is
        chosen automatically.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    Ainv : array-like or LinearOperator or RandomVariable, shape=(n,n)
        Optional. A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Initial guess for the solution of the linear system. Will be ignored if ``Ainv`` is given.
    assume_A : str, default="sympos"
        Assumptions on the matrix, which can influence solver choice or behavior. The available options are

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         symmetric pos. def.  ``sympos``
        ====================  =========
    maxiter : int
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the dimension of :math:`A`.
    resid_tol : float
        Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
        terminates.

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
        If size mismatches detected or input ``A`` is not square.
    LinAlgError
        If the matrix is singular.
    LinAlgWarning
        If an ill-conditioned input a is detected.

    See Also
    --------
    bayescg : Solve linear systems with prior information on the solution.

    Examples
    --------
    """

    # Check linear system for type and dimension mismatch
    _check_linear_system(A=A, b=b, Ainv=Ainv, x0=x0)

    # Transform linear system types to random variables and linear operators
    # and set a default prior if not specified
    A, b, Ainv, x = _preprocess_linear_system(A=A, b=b, Ainv=Ainv, x0=x0)

    # Select solver
    # TODO: include also random variable covariance type in this selection? (symmkron => sympos)
    if assume_A in ('sym', 'sympos'):
        if isinstance(Ainv, probnum.probability.RandomVariable):
            solve_iter = _problinsolve_symm_iter
        elif isinstance(x0, probnum.probability.RandomVariable):
            solve_iter = _bayescg_iter
        else:
            raise ValueError("No prior information specified on Ainv or x.")
    elif assume_A in ('gen', 'pos'):
        if isinstance(Ainv, probnum.probability.RandomVariable):
            solve_iter = _problinsolve_gen_iter
        elif isinstance(x0, probnum.probability.RandomVariable):
            solve_iter = _bayescg_iter
        else:
            raise ValueError("No prior information specified on Ainv or x.")
    else:
        raise ValueError('\'{}\' is not a recognized linear operator assumption.'.format(assume_A))

    # TODO: make kronecker structure explicit in solver selection and implement via Kronecker linear operator

    # Set default parameters
    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10

    # Solve linear system
    # TODO: refactor into returning random variables and unify signatures of solve_iter
    A, Ainv_mean, Ainv_cov_kronfac, x, info = solve_iter(A=A, Ainv_mean=Ainv.mean(), Ainv_cov_kronfac=Ainv.cov().A, b=b,
                                                         maxiter=maxiter, resid_tol=resid_tol)

    # Check solution and issue warnings (e.g. singular or ill-conditioned matrix)
    _check_solution(info=info)

    return x, A, Ainv, info


def bayescg(A, b, x0=None):
    """
    Conjugate Gradients using prior information on the solution of the linear system.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver takes prior information
    on the solution and outputs a posterior belief over :math:`x`. This code implements the
    method described in [1]_. Note that the solution-based view of BayesCG and the matrix-based view of
    :meth:`problinsolve` correspond (see [2]_).

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

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    """
    # Check linear system for type and dimension mismatch
    _check_linear_system(A=A, b=b, Ainv=None, x0=x0)

    raise NotImplementedError


def _check_linear_system(A, b, Ainv=None, x0=None):
    """
    Check linear system compatibility.

    Raises an exception if the input arguments are not of the right type or not compatible.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable
        Linear operator.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    Ainv : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
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
    np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator, probnum.probability.RandomVariable)
    vector_types = (np.ndarray, scipy.sparse.spmatrix, probnum.probability.RandomVariable)
    if not isinstance(A, linop_types):
        raise ValueError(
            "A must be either an array, a linear operator or a RandomVariable of either.")
    if not isinstance(b, vector_types):
        raise ValueError("The right hand side must be a (sparse) array.")
    if Ainv is not None and not isinstance(Ainv, linop_types):
        raise ValueError(
            "The inverse of A must be either an array, a linear operator or a RandomVariable of either.")
    if x0 is not None and not isinstance(x0, vector_types):
        raise ValueError("The initial guess for the solution must be a (sparse) array.")

    # Dimension mismatch
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and b must match.")
    if Ainv is not None and A.shape == Ainv.shape:
        raise ValueError("Dimension mismatch. The dimensions of A and Ainv must match.")
    if x0 is not None and A.shape[1] != x0.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and x0 must match.")

    # Square matrices
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if Ainv is not None and Ainv.shape[0] != Ainv.shape[1]:
        raise ValueError("The inverse of A must be square.")


def _preprocess_linear_system(A, b, Ainv=None, x0=None):
    """
    Transform the linear system to linear operator and random variable form.

    .

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable
        Linear operator.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    Ainv : array-like or LinearOperator or RandomVariable, shape=(n,n)
        Optional. A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Initial guess for the solution of the linear system. Will be ignored if ``Ainv`` is given.

    Returns
    -------
    A : RandomVariable, shape=(n,n)
        Posterior belief over the linear operator.
    b : array-like, shape=(n,) or (n, nrhs)
        Right-hand-side of the linear system.
    Ainv : RandomVariable, shape=(n,n)
        Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
    x : array-like or RandomVariable, shape=(n,) or (n, nrhs)
        Approximate solution :math:`x` to the linear system.

    Raises
    ------

    """
    # Choose matrix based view if not clear from arguments
    if Ainv is not None and x0 is not None:
        warnings.warn(
            "Cannot use prior information on both the matrix inverse and the solution. The latter will be ignored.")
        x = None
    else:
        x = x0

    # Todo Automatic prior selection based on data scale, etc.?

    # Transform linear system to correct dimensions
    if isinstance(A, probnum.linalg.linear_operators.LinearOperator):
        A = A
    else:
        A = probnum.utils.atleast_2d(A)
    b = probnum.utils.atleast_1d(b)
    if Ainv is not None and not isinstance(Ainv, probnum.linalg.linear_operators.LinearOperator):
        Ainv = probnum.utils.atleast_2d(Ainv)
    if x0 is not None:
        x = probnum.utils.atleast_1d(x0)

    # TODO create random variables and linear operators

    assert (not (Ainv is None and x is None)), "Neither Ainv nor x are specified."

    return A, b, Ainv, x


def _check_solution(info):
    """

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
    raise NotImplementedError
    # Singular matrix

    # Ill-conditioned matrix A


def _problinsolve_gen_iter():
    raise NotImplementedError


def _mean_update_operator(self, u, v, shape):
    """
    Implements the rank 2 update term for the posterior mean of :math:`H`.

    Parameters
    ----------
    u : array-like
        Update vector :math:`u_i=\\frac{W_iy_i}{y_i^{\\top}W_iy_i}`
    v : array-like
        Update vector :math:`v_i=\\Delta - \\frac{1}{2} u_iy_i^{\\top}\\Delta`
    shape : tuple
        Shape of the resulting update operator.

    Returns
    -------
    update : probnum.linalg.linear_operators.LinearOperator
    """

    def mv(x):
        return np.dot(v, x) * u + np.dot(u, x) * v

    def mm(M):
        return np.outer(u, M @ v) + np.outer(v, u @ M)

    return probnum.linalg.linear_operators.LinearOperator(
        shape=shape,
        matvec=mv,
        rmatvec=mv,
        matmat=mm,
        dtype=u.dtype
    )


def _cov_kron_fac_update_operator(self, u, Wy, shape):
    """
    Implements the rank 1 update term for the posterior covariance Kronecker factor :math:`W`.

    Parameters
    ----------
    u : array-like
        Update vector :math:`u_i=\\frac{W_iy_i}{y_i^{\\top}W_iy_i}`
    Wy : array-like
        Update vector :math:`W_iy_i`
    shape : tuple
        Shape of the resulting update operator.

    Returns
    -------
    update : LinearOperator
    """

    def mv(x):
        return np.dot(Wy, x) * u

    def mm(M):
        return np.outer(u, M @ Wy)

    return probnum.linalg.linear_operators.LinearOperator(
        shape=shape,
        matvec=mv,
        rmatvec=mv,
        matmat=mm,
        dtype=u.dtype
    )


def _problinsolve_symm_iter(A, Ainv_mean, Ainv_cov_kronfac, b, maxiter, resid_tol):
    # initialization
    iter_ = 0
    n = Ainv_mean.shape[0]
    x = Ainv_mean @ b
    resid = A @ x - b

    # iteration with stopping criteria
    # todo: extract iteration and make into iterator
    while True:
        # check convergence
        _has_converged, _conv_crit = _check_convergence(iter=iter_, maxiter=maxiter, resid=resid, resid_tol=resid_tol)
        if _has_converged:
            break

        # compute search direction (with implicit reorthogonalization)
        search_dir = - Ainv_mean @ resid

        # perform action and observe
        obs = A @ search_dir

        # compute step size
        step_size = - np.dot(search_dir, resid) / np.dot(search_dir, obs)

        # todo: scale search_dir and obs by step-size to fulfill theory on conjugate directions?

        # step and residual update
        x = x + step_size * search_dir
        resid = resid + step_size * obs

        # (symmetric) mean and covariance updates
        Wy = Ainv_cov_kronfac @ obs
        delta = search_dir - Ainv_mean @ obs
        u = Wy / np.dot(obs, Wy)
        v = delta - 0.5 * np.dot(obs, delta) * u

        # rank 2 mean update (+= uv' + vu')
        # todo: only use linear operators if necessary (only create update operators if H is a LinearOperator)
        Ainv_mean = Ainv_mean + _mean_update_operator(u=u, v=v, shape=(n, n))

        # rank 1 covariance kronecker factor update (-= u(Wy)')
        Ainv_cov_kronfac = Ainv_cov_kronfac - _cov_kron_fac_update_operator(u=u, Wy=Wy, shape=(n, n))

        # iteration increment
        iter_ += 1

    # Log information on solution
    # TODO: matrix condition from solver (see scipy solvers)
    info = {
        "iter": iter_,
        "maxiter": maxiter,
        "resid": resid,
        "conv_crit": _conv_crit,
        "matrix_cond": None
    }

    return A, Ainv_mean, Ainv_cov_kronfac, x, info


def _bayescg_iter():
    raise NotImplementedError


def _check_convergence(iter, maxiter, resid, resid_tol):
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
    has_converged : bool
        True if the method has converged.
    convergence_criterion : str
        Convergence criterion which caused termination.
    """
    # maximum iterations
    if iter >= maxiter:
        return True, "maxiter"
    # residual below error tolerance
    # todo: add / replace with relative tolerance
    elif np.linalg.norm(resid) < resid_tol:
        return True, "resid"
    # uncertainty-based
    # todo: based on posterior contraction
    else:
        return False, ""
