"""Probabilistic numerical methods for solving linear systems."""

import abc
import warnings
import numpy as np
import scipy
import scipy.sparse
import probnum.probability as probability
import probnum.linalg.linear_operators as linear_operators
import probnum.utils as utils


def problinsolve(a, b, ainv=None, x0=None, assume_a="sympos", maxiter=None, resid_tol=10 ** -6):
    """
    Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`. They return a probability measure
    which quantifies uncertainty in the output arising from finite computational resources.

    This solver can take prior information either on the linear operator :math:`A` or its inverse :math:`H=A^{-1}` and
    outputs a posterior belief over :math:`A` or :math:`H`. For a specific class of priors this recovers the iterates of
    the conjugate gradient method as the posterior mean. This code implements the method described in [1]_, [2]_ and
    [3]_.

    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260
    .. [3] Hennig, P. and Osborne M. A., *Probabilistic Numerics. Computation as Machine Learning*, 2020, Cambridge
           University Press

    Parameters
    ----------
    a : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.probability.RandomVariable`. If an array or linear operator is given, a prior distribution is
        chosen automatically.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    ainv : array-like or LinearOperator or RandomVariable, shape=(n,n)
        Optional. A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Initial guess for the solution of the linear system. Will be ignored if ``ainv`` is given.
    assume_a : str, default='sympos'
        Assumptions on the matrix, which can influence solver choice or behavior. The available options are

        ====================  =========
         generic matrix       'gen'
         symmetric            'sym'
         positive definite    'pos'
         symmetric pos. def.  'sympos'
        ====================  =========
    maxiter : int
        Maximum number of iterations.
    resid_tol : float
        Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
        terminates.

    Returns
    -------
        x : RandomVariable, shape=(n,) or (n, nrhs)
            Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.
        a : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator.
        ainv : RandomVariable, shape=(n,n)
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info : dict
            Information on convergence of the solver.

    Raises
    ------
    ValueError
        If size mismatches detected or input ``a`` is not square.
    LinAlgError
        If the matrix is singular.
    LinAlgWarning
        If an ill-conditioned input a is detected.
    """

    # Check linear system
    _check_linear_system(a=a, b=b, ainv=ainv, x0=x0)
    if ainv is not None and x0 is not None:
        warnings.warn(
            "Cannot use prior information on both the matrix inverse and on a guess of the solution." +
            "The latter will be ignored")
        x0 = None

    # Transform array(s) to correct dimensions
    # Todo: write util functions who do this for random variables
    if isinstance(a, linear_operators.LinearOperator):
        a1 = a
    else:
        a1 = np.atleast_2d(a)
    b1 = utils.atleast_1d(b)
    if ainv is not None and not isinstance(ainv, linear_operators.LinearOperator):
        ainv1 = np.atleast_2d(ainv)
    if x0 is not None:
        x = utils.atleast_1d(x0)

    # Check assumptions on linear operator A
    if assume_a not in ('gen', 'sym', 'pos', 'sympos'):
        raise ValueError('\'{}\' is not a recognized linear operator assumption.'.format(assume_a))

    # Choose prior if none is given

    # Select solver

    # Solve linear system

    # Check solution and issue warnings (e.g. singular or ill-conditioned matrix)
    _check_solution(info=info)

    # Log information on solution

    return x, a1, ainv1, info


def _check_linear_system(a, b, ainv=None, x0=None):
    """
    Check linear system compatibility.

    Raises an exception if the input arguments are not of the right type or not compatible.

    Parameters
    ----------
    a : array-like or LinearOperator or RandomVariable
        Linear operator.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    ainv : array-like or LinearOperator or RandomVariable, shape=(n,n)
        Optional. A square matrix, linear operator or random variable representing the prior belief over the inverse
        :math:`H=A^{-1}`.
    x0 : array-like, shape=(n,) or (n, nrhs)
        Optional. Initial guess for the solution of the linear system. Will be ignored if ``ainv`` is given.

    Raises
    ------
    ValueError
        If type or size mismatches detected or inputs ``a`` and ``ainv`` are not square.
    """
    # Check types
    linop_types = (np.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator, probability.RandomVariable)
    vector_types = (np.ndarray, scipy.sparse.spmatrix, probability.RandomVariable)
    if not isinstance(a, linop_types):
        raise ValueError("A must be either an array, a linear operator or a RandomVariable of either.")
    if not isinstance(b, vector_types):
        raise ValueError("The right hand side must be a (sparse) array.")
    if ainv is not None and not isinstance(ainv, linop_types):
        raise ValueError("The inverse of A must be either an array, a linear operator or a RandomVariable of either.")
    if x0 is not None and not isinstance(x0, vector_types):
        raise ValueError("The initial guess for the solution must be a (sparse) array.")

    # Dimension mismatch
    if a.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and b must match.")
    if a.shape == ainv.shape:
        raise ValueError("Dimension mismatch. The dimensions of A and Ainv must match.")
    if x0 is not None and a.shape[1] != x0.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and x0 must match.")

    # Square matrices
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix A must be square.")
    if ainv.shape[0] != ainv.shape[1]:
        raise ValueError("The inverse of A must be square.")


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
    # Singular matrix

    # Ill-conditioned matrix A


def bayescg(a, b, x0):
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
    a : array-like or LinearOperator or RandomVariable, shape=(n,n)
        A square matrix or linear operator. A prior distribution can be provided as a
        :class:`~probnum.probability.RandomVariable`. If an array or linear operator are given, a prior distribution is
        chosen automatically.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    x0 : array-like or RandomVariable, shape=(n,) or or (n, nrhs)
        Prior belief over the solution of the linear system.
    """
    raise NotImplementedError


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
        # todo: refactor as dimension of linear operator is not known at this point
        if H_mean is not None:
            self.H_mean = scipy.sparse.linalg.aslinearoperator(H_mean)
        else:
            self.H_mean = None
        if H_cov_kronfac is not None:
            self.H_cov_kronfac = scipy.sparse.linalg.aslinearoperator(H_cov_kronfac)
        else:
            self.H_cov_kronfac = None

    def _has_converged(self, iter, maxiter, resid, resid_tol):
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
        """
        # maximum iterations
        if iter >= maxiter:
            return True
        # residual below error tolerance
        # todo: add / replace with relative tolerance
        elif np.linalg.norm(resid) < resid_tol:
            return True
        # uncertainty-based
        # todo: based on posterior contraction
        else:
            return False

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
        update : LinearOperator
        """

        def mv(x):
            return np.dot(v, x) * u + np.dot(u, x) * v

        def mm(M):
            return np.outer(u, np.matmul(M, v)) + np.outer(v, np.matmul(u, M))

        return scipy.sparse.linalg.LinearOperator(
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
            return np.outer(u, np.matmul(M, Wy))

        return scipy.sparse.linalg.LinearOperator(
            shape=shape,
            matvec=mv,
            rmatvec=mv,
            matmat=mm,
            dtype=u.dtype
        )

    def solve(self, A, b, maxiter=None, resid_tol=10 ** -6):
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

        Returns
        -------
        x : array-like
            Approximate solution :math:`Ax \\approx b` to the linear system.
        """

        # make linear system
        A = scipy.sparse.linalg.interface.aslinearoperator(A)
        b = np.asanyarray(b)

        # check linear system dimensionality
        _check_linear_system(A=A, b=b)

        # setup
        n = len(b)
        if maxiter is None:
            maxiter = n * 10
        if self.H_mean is None:
            # todo: use / implement better operator classes (PyLops?) that can perform .T, .todense() for H_mean as well
            self.H_mean = scipy.sparse.linalg.interface.IdentityOperator(shape=(n, n))
        if self.H_cov_kronfac is None:
            self.H_cov_kronfac = scipy.sparse.linalg.interface.IdentityOperator(shape=(n, n))

        # initialization
        iter = 0
        x = self.H_mean.matvec(b)
        resid = A.matvec(x) - b

        # iteration with stopping criteria
        # todo: extract iteration and make into iterator
        while not self._has_converged(iter=iter, maxiter=maxiter, resid=resid, resid_tol=resid_tol):
            # compute search direction (with implicit reorthogonalization)
            search_dir = - self.H_mean.matvec(resid)

            # perform action and observe
            obs = A.matvec(search_dir)

            # compute step size
            step_size = - np.dot(search_dir, resid) / np.dot(search_dir, obs)

            # todo: scale search_dir and obs by step-size to fulfill theory on conjugate directions?

            # step and residual update
            x = x + step_size * search_dir
            resid = resid + step_size * obs

            # (symmetric) mean and covariance updates
            Wy = self.H_cov_kronfac.matvec(obs)
            delta = search_dir - self.H_mean.matvec(obs)
            u = Wy / np.dot(obs, Wy)
            v = delta - 0.5 * np.dot(obs, delta) * u

            # rank 2 mean update operator (+= uv' + vu')
            # todo: speedup: implement full update as operator and do not rely on +?
            self.H_mean = self.H_mean + self._mean_update_operator(u=u, v=v, shape=(n, n))

            # rank 1 covariance kronecker factor update operator (-= u(Wy)')
            self.H_cov_kronfac = self.H_cov_kronfac - self._cov_kron_fac_update_operator(u=u, Wy=Wy, shape=(n, n))

            # iteration increment
            iter += 1

        # information about convergence
        info = {
            "n_iter": iter,
            "resid_norm": np.linalg.norm(resid)
        }
        print(info)
        # todo: return solution, some general distribution class and dict with
        #  convergence info (iter, resid, convergence criterion)
        return x, info
