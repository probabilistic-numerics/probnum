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

from probnum import probability
from probnum.linalg import linear_operators
import probnum.utils as utils

__all__ = ["problinsolve", "bayescg"]


def problinsolve(A, b, A0=None, Ainv0=None, x0=None, assume_A="sympos", maxiter=None, resid_tol=10 ** -6,
                 callback=None):
    """
    Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`. They return a probability measure
    which quantifies uncertainty in the output arising from finite computational resources. This solver can take prior
    information either on the linear operator :math:`A` or its inverse :math:`H=A^{-1}` in
    the form of a random variable ``A0`` or ``Ainv0`` and outputs a posterior belief over :math:`A` or :math:`H`. This
    code implements the method described in [1]_ and [2]_.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on Optimization*, 2015, 25, 234-260

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
    A0 : RandomVariable, shape=(n,n), optional
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
    resid_tol : float, optional
        Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
        terminates.
    callback : function, optional
        User-supplied function called after each iteration of the linear solver. It is called as
        ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid)`` and can be used to return quantities from the iteration. Note that
        depending on the function supplied, this can slow down the solver.

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
    # TODO: move this into the solver iteration itself (compute with matrices)
    for i in range(nrhs):
        # Select and initialize solver
        linear_solver = _init_solver(A=A, b=utils.as_colvec(b[:, i]), A0=A0, Ainv0=Ainv0, x0=x)

        # Solve linear system
        x, A0, Ainv0, info = linear_solver.solve(maxiter=maxiter, resid_tol=resid_tol, callback=callback)

    # Return Ainv @ b for multiple rhs
    if nrhs > 1:
        x = Ainv0 @ b

    # Check solution and issue warnings (e.g. singular or ill-conditioned matrix)
    _check_solution(info=info)

    return x, A0, Ainv0, info


def bayescg(A, b, x0=None, maxiter=None, resid_tol=None, callback=None):
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
    resid_tol : float
        Residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b \\rVert < \\text{tol}`, the iteration
        terminates.
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
    x, _, _, info = _BayesCG(A=A, b=b, x=x0).solve(maxiter=maxiter, resid_tol=resid_tol)

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

    # Choose prior if none specified, based on matrix assumptions in "assume_A"
    # TODO: Automatic prior selection based on data scale, matrix trace, etc.?
    # TODO: Implement case where only a pre-conditioner is given as Ainv0
    if A0 is None and Ainv0 is None:
        dist = probability.Normal(mean=linear_operators.Identity(shape=A.shape[0]),
                                  cov=linear_operators.SymmetricKronecker(linear_operators.Identity(shape=A.shape[0]),
                                                                          linear_operators.Identity(shape=A.shape[0])))
        Ainv0 = probability.RandomVariable(distribution=dist)

    # Translate A0 prior into Ainv0 prior or vice versa
    # TODO: Implement theory from paper
    if A0 is None:
        dist = probability.Normal(mean=linear_operators.Identity(shape=A.shape[0]),
                                  cov=linear_operators.SymmetricKronecker(linear_operators.Identity(shape=A.shape[0]),
                                                                          linear_operators.Identity(shape=A.shape[0])))
        A0 = probability.RandomVariable(distribution=dist)  # TODO: Remove me

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


def _check_convergence(iter, maxiter, resid, resid_tol):
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
        warnings.warn(message="Iteration terminated. Solver reached the maximum number of iterations.")
        return True, "maxiter"
    # residual below error tolerance
    # todo: add / replace with relative tolerance
    elif np.linalg.norm(resid) < resid_tol:
        return True, "resid"
    # uncertainty-based
    # todo: based on posterior contraction
    else:
        return False, ""


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

    def solve(self, callback=None, **convergence_args):
        """
        Solve the linear system :math:`Ax=b`.

        Parameters
        ----------
        callback : function, optional
            User-supplied function called after each iteration of the linear solver. It is called as
            ``callback(xk, sk, yk, alphak, resid, **kwargs)`` and can be used to return quantities from the iteration.
            Note that depending on the function supplied, this can slow down the solver.
        convergence_args
            Arguments specifying when the solver has converged to a solution.

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

    def solve(self, callback=None, maxiter=None, resid_tol=None):
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
        Mean of the prior distribution on the linear operator :math:`A`.
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
        # TODO: call class function here which derives missing prior mean and covariance via posterior correspondence
        super().__init__(A=A, b=b)

    def _create_output_randvars(self):
        """Return output random variables x, A, Ainv from their means and covariances."""
        # Create output random variables
        A = probability.RandomVariable(shape=self.A_mean.shape,
                                       dtype=float,
                                       distribution=probability.Normal(mean=self.A_mean,
                                                                       cov=linear_operators.SymmetricKronecker(
                                                                           A=self.A_covfactor)))
        cov_Ainv = linear_operators.SymmetricKronecker(A=self.Ainv_covfactor)
        Ainv = probability.RandomVariable(shape=self.Ainv_mean.shape,
                                          dtype=float,
                                          distribution=probability.Normal(mean=self.Ainv_mean, cov=cov_Ainv))
        # Induced distribution on x via Ainv (see Hennig2020)
        # E = A^-1 b, Cov = 1/2 (W b'Wb + Wbb'W)
        Wb = self.Ainv_covfactor @ self.b
        # TODO: do we want todense() here or just not a linear operator?
        x = probability.RandomVariable(shape=(self.A_mean.shape[0],),
                                       dtype=float,
                                       distribution=probability.Normal(mean=self.x.ravel(),
                                                                       cov=0.5 * (self.Ainv_covfactor.todense() * (
                                                                               Wb.T @ self.b) + Wb @ Wb.T)))
        return x, A, Ainv

    def solve(self, callback=None, maxiter=None, resid_tol=None):
        # initialization
        iter_ = 0
        resid = self.A @ self.x - self.b

        # iteration with stopping criteria
        while True:
            # check convergence
            _has_converged, _conv_crit = _check_convergence(iter=iter_, maxiter=maxiter,
                                                            resid=resid, resid_tol=resid_tol)
            if _has_converged:
                break

            # compute search direction (with implicit reorthogonalization)
            search_dir = - self.Ainv_mean @ resid

            # perform action and observe
            obs = self.A @ search_dir

            # compute step size
            step_size = - (search_dir.T @ resid) / (search_dir.T @ obs)
            # todo: scale search_dir and obs by step-size to fulfill theory on conjugate directions?

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
            # TODO: should we really perform these updates in operator form? Yes, cannot build full matrices
            #  for example in deep learning. BUT: Ensure speed of iteration is fast.
            uvT_A = u_A @ v_A.T  # TODO: handle this via a product_operator implementing matvec(x) = u * v @ x
            uvT_Ainv = u_Ainv @ v_Ainv.T
            self.A_mean = linear_operators.aslinop(self.A_mean) + linear_operators.MatrixMult(uvT_A + uvT_A.T)
            self.Ainv_mean = linear_operators.aslinop(self.Ainv_mean) + linear_operators.MatrixMult(
                uvT_Ainv + uvT_Ainv.T)

            # rank 1 covariance kronecker factor update (-= u_A(Vs)' and -= u_Ainv(Wy)')
            self.A_covfactor = linear_operators.aslinop(self.A_covfactor) - linear_operators.MatrixMult(Vs @ u_A.T)
            self.Ainv_covfactor = linear_operators.aslinop(self.Ainv_covfactor) - linear_operators.MatrixMult(
                Wy @ u_Ainv.T)

            # callback function used to extract quantities from iteration
            if callback is not None:
                xk, Ak, Ainvk = self._create_output_randvars()
                callback(xk=xk, Ak=Ak, Ainvk=Ainvk, sk=search_dir, yk=obs, alphak=step_size, resid=resid)

            # iteration increment
            iter_ += 1

        # Create output random variables
        x, A, Ainv = self._create_output_randvars()

        # Log information on solution
        # TODO: matrix condition from solver (see scipy solvers)
        info = {
            "iter": iter_,
            "maxiter": maxiter,
            "resid_l2norm": np.linalg.norm(resid, ord=2),
            "conv_crit": _conv_crit,
            "matrix_cond": None
        }

        return x, A, Ainv, info


class _BayesCG(_ProbabilisticLinearSolver):
    """
    Solver iteration of BayesCG.

    Implements the solve iteration of BayesCG [1]_ [2]_.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian Analysis*, 2019, 14, 937-1012
    .. [2] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View, *Statistics and Computing*, 2019

    Parameters
    ----------


    """

    def __init__(self, A, b, x):
        self.x = x
        super().__init__(A=A, b=b)

    def solve(self, maxiter, resid_tol):
        raise NotImplementedError
