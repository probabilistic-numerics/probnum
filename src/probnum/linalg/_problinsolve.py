"""Probabilistic numerical methods for solving linear systems.

This module provides routines to solve linear systems of equations in a
Bayesian framework. This means that a prior distribution over elements
of the linear system can be provided and is updated with information
collected by the solvers to return a posterior distribution.
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import probnum  # pylint: disable=unused-import
from probnum import linops, randvars, utils
from probnum.linalg.solvers.matrixbased import SymmetricMatrixBasedSolver
from probnum.typing import LinearOperatorArgType

# pylint: disable=too-many-branches


def problinsolve(
    A: Union[
        LinearOperatorArgType,
        "randvars.RandomVariable[LinearOperatorArgType]",
    ],
    b: Union[np.ndarray, "randvars.RandomVariable[np.ndarray]"],
    A0: Optional[
        Union[
            LinearOperatorArgType,
            "randvars.RandomVariable[LinearOperatorArgType]",
        ]
    ] = None,
    Ainv0: Optional[
        Union[
            LinearOperatorArgType,
            "randvars.RandomVariable[LinearOperatorArgType]",
        ]
    ] = None,
    x0: Optional[Union[np.ndarray, "randvars.RandomVariable[np.ndarray]"]] = None,
    assume_A: str = "sympos",
    maxiter: Optional[int] = None,
    atol: float = 10 ** -6,
    rtol: float = 10 ** -6,
    callback: Optional[Callable] = None,
    **kwargs
) -> Tuple[
    "randvars.RandomVariable[np.ndarray]",
    "randvars.RandomVariable[linops.LinearOperator]",
    "randvars.RandomVariable[linops.LinearOperator]",
    Dict,
]:
    r"""Solve the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \in \mathbb{R}^{n \times n}` and :math:`b \in \mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources or stochastic input. This solver can take prior
    information either on the linear operator :math:`A` or its inverse :math:`H=A^{
    -1}` in the form of a random variable ``A0`` or ``Ainv0`` and outputs a posterior
    belief about :math:`A` or :math:`H`. This code implements the method described in
    Wenger et al. [1]_ based on the work in Hennig et al. [2]_.

    Parameters
    ----------
    A :
        *shape=(n, n)* -- A square linear operator (or matrix). Only matrix-vector
        products :math:`v \mapsto Av` are used internally.
    b :
        *shape=(n, ) or (n, nrhs)* -- Right-hand side vector, matrix or random
        variable in :math:`A x = b`.
    A0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief about the linear operator :math:`A`.
    Ainv0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief about the inverse :math:`H=A^{-1}`. This can be
        viewed as a preconditioner.
    x0 :
        *shape=(n, ) or (n, nrhs)* -- Prior belief for the solution of the linear
        system. Will be ignored if ``Ainv0`` is given.
    assume_A :
        Assumptions on the linear operator which can influence solver choice and
        behavior. The available options are (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========

    maxiter :
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol :
        Absolute convergence tolerance.
    rtol :
        Relative convergence tolerance.
    callback :
        User-supplied function called after each iteration of the linear solver. It is
        called as ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs)`` and can
        be used to return quantities from the iteration. Note that depending on the
        function supplied, this can slow down the solver considerably.
    kwargs : optional
        Optional keyword arguments passed onto the solver iteration.

    Returns
    -------
    x :
        Approximate solution :math:`x` to the linear system. Shape of the return matches
        the shape of ``b``.
    A :
        Posterior belief over the linear operator.
    Ainv :
        Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
    info :
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
    For a specific class of priors the posterior mean of :math:`x_k=Hb` coincides with
    the iterates of the conjugate gradient method. The matrix-based view taken here
    recovers the solution-based inference of :func:`bayescg` [3]_.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020
    .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
       Optimization*, 2015, 25, 234-260
    .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
       *Statistics and Computing*, 2019

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

    # Check matrix assumptions for correctness
    assume_A = assume_A.lower()
    _assume_A_tmp = assume_A
    for allowed_str in ["gen", "sym", "pos", "noise"]:
        _assume_A_tmp = _assume_A_tmp.replace(allowed_str, "")
    if _assume_A_tmp != "":
        raise ValueError(
            "Assumption '{}' contains unrecognized linear operator properties.".format(
                assume_A
            )
        )

    # Transform the linear system to an appropriate form
    A, b, x0 = _preprocess_linear_system(A=A, b=b, x0=x0)

    # Parameter initialization
    n = A.shape[0]
    nrhs = b.shape[1]
    x = x0
    info = {}

    # Set convergence parameters
    if maxiter is None:
        maxiter = n * 10

    if nrhs > 1:
        # Iteratively solve for multiple right hand sides (with posteriors as new
        # priors)
        for i in range(nrhs):
            if i > 0:
                x = None  # Only use prior information on Ainv for multiple rhs
            # Select and initialize solver
            linear_solver = _init_solver(
                A=A,
                b=utils.as_colvec(b[:, i]),
                A0=A0,
                Ainv0=Ainv0,
                x0=x,
                assume_A=assume_A,
            )

            # Solve linear system
            x, A0, Ainv0, info = linear_solver.solve(
                maxiter=maxiter, atol=atol, rtol=rtol, callback=callback, **kwargs
            )

        # Return Ainv @ b for multiple rhs
        x = Ainv0 @ b
    else:
        # Single right hand side
        linear_solver = _init_solver(
            A=A, b=b, A0=A0, Ainv0=Ainv0, x0=x, assume_A=assume_A
        )

        # Solve linear system
        x, A0, Ainv0, info = linear_solver.solve(
            maxiter=maxiter, atol=atol, rtol=rtol, callback=callback, **kwargs
        )

    # Check result and issue warnings (e.g. singular or ill-conditioned matrix)
    _postprocess(info=info, A=A)

    return x, A0, Ainv0, info


def bayescg(A, b, x0=None, maxiter=None, atol=None, rtol=None, callback=None):
    """Conjugate Gradients using prior information on the solution of the linear system.

    In the setting where :math:`A` is a symmetric positive-definite matrix, this solver
    takes prior information on the solution and outputs a posterior belief over
    :math:`x`. This code implements the method described in Cockayne et al. [1]_.

    Note that the solution-based view of BayesCG and the matrix-based view of
    :meth:`problinsolve` correspond [2]_.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A square linear operator (or matrix). Only matrix-vector products :math:`Av` are
        used internally.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    x0 : array-like or RandomVariable, shape=(n,) or or (n, nrhs)
        Prior belief over the solution of the linear system.
    maxiter : int
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol : float, optional
        Absolute residual tolerance. If :math:`\\lVert r_i \\rVert = \\lVert Ax_i - b
        \\rVert < \\text{atol}`, the iteration terminates.
    rtol : float, optional
        Relative residual tolerance. If :math:`\\lVert r_i \\rVert  < \\text{rtol}
        \\lVert b \\rVert`, the iteration terminates.
    callback : function, optional
        User-supplied function called after each iteration of the linear solver. It is
        called as ``callback(xk, sk, yk, alphak, resid, **kwargs)`` and can be used to
        return quantities from the iteration. Note that depending on the function
        supplied, this can slow down the solver.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    .. [2] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
       *Statistics and Computing*, 2019

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    """
    raise NotImplementedError


def _check_linear_system(A, b, A0=None, Ainv0=None, x0=None):
    """Check linear system compatibility.

    Raises an exception if the input arguments are not of the right type or not
    compatible.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A square linear operator (or matrix). Only matrix-vector products :math:`Av` are
        used internally.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.

    Raises
    ------
    ValueError
        If type or size mismatches detected or inputs ``A`` and ``Ainv`` are not square.
    """
    # Check types
    linop_types = (
        np.ndarray,
        scipy.sparse.spmatrix,
        scipy.sparse.linalg.LinearOperator,
        linops.LinearOperator,
        randvars.RandomVariable,
    )
    vector_types = (np.ndarray, scipy.sparse.spmatrix, randvars.RandomVariable)
    if not isinstance(A, linop_types):
        raise ValueError(
            "A must be either an array, a linear operator or a random variable."
        )
    if not isinstance(b, vector_types):
        raise ValueError(
            "The right hand side must be a (sparse) array or a random variable."
        )
    if A0 is not None and not isinstance(A0, randvars.RandomVariable):
        raise ValueError("The prior belief over A must be a random variable.")
    if Ainv0 is not None and not isinstance(Ainv0, linop_types):
        raise ValueError(
            "The inverse of A must be either an array, a linear operator or "
            "a random variable of either."
        )
    if x0 is not None and not isinstance(x0, vector_types):
        raise ValueError("The initial guess for the solution must be a (sparse) array.")

    # Prior distribution mismatch
    if (
        isinstance(A0, randvars.RandomVariable)
        or isinstance(Ainv0, randvars.RandomVariable)
    ) and isinstance(x0, randvars.RandomVariable):
        raise ValueError(
            "Cannot specify distributions on the linear operator and the solution "
            "simultaneously."
        )

    # Dimension mismatch
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch. The dimensions of A and b must match.")
    if Ainv0 is not None and A.shape != Ainv0.shape:
        raise ValueError(
            "Dimension mismatch. The dimensions of A and Ainv0 must match."
        )
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


def _preprocess_linear_system(A, b, x0=None):
    """Transform the linear system to an appropriate form.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A square linear operator (or matrix). Only matrix-vector products :math:`Av` are
        used internally.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.

    Returns
    -------
    A : RandomVariable, shape=(n,n)
        Prior belief over the linear operator :math:`A`.
    b : array-like, shape=(n,) or (n, nrhs)
        Right-hand-side of the linear system.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.
    """
    # Transform linear system to correct dimensions
    if not isinstance(b, randvars.RandomVariable):
        b = utils.as_colvec(b)  # (n,) -> (n, 1)
    if x0 is not None:
        x0 = utils.as_colvec(x0)  # (n,) -> (n, 1)

    return A, b, x0


def _init_solver(A, b, A0, Ainv0, x0, assume_A):
    """Selects and initializes an appropriate instance of the probabilistic linear
    solver based on the system properties and prior information given.

    Parameters
    ----------
    A : array-like or LinearOperator, shape=(n,n)
        A square linear operator (or matrix). Only matrix-vector products :math:`Av` are
        used internally.
    b : array_like, shape=(n,) or (n, nrhs)
        Right-hand side vector or matrix in :math:`A x = b`.
    A0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`.
    Ainv0 : array-like or LinearOperator or RandomVariable, shape=(n,n), optional
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`.
    x0 : array-like, or RandomVariable, shape=(n,) or (n, nrhs)
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.
    assume_A : str
        Assumptions on the linear operator, which can influence solver choice or
        behavior. The available options are (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========

    Returns
    -------
    linear_solver : ProbabilisticLinearSolver
        A type of probabilistic linear solver implementing the solve method for linear
        systems.
    """
    # Choose matrix based view if not clear from arguments
    if (Ainv0 is not None or A0 is not None) and isinstance(
        x0, randvars.RandomVariable
    ):
        warnings.warn(
            "Cannot use prior uncertainty on both the matrix (inverse) and the "
            "solution. The latter will be ignored."
        )
        x0 = x0.mean

    # Extract information from priors
    # System matrix is symmetric
    if isinstance(A0, randvars.RandomVariable):
        if isinstance(A0.cov, linops.SymmetricKronecker) and "sym" not in assume_A:
            assume_A += "sym"
    if isinstance(Ainv0, randvars.RandomVariable):
        if isinstance(Ainv0.cov, linops.SymmetricKronecker) and "sym" not in assume_A:
            assume_A += "sym"
    # System matrix is NOT stochastic
    if (
        not isinstance(A, randvars.RandomVariable)
        and not isinstance(A, scipy.sparse.linalg.LinearOperator)
        and "noise" in assume_A
    ):
        warnings.warn(
            "A is assumed to be noisy, but is neither a random variable nor a "
            "linear operator. Use exact probabilistic linear solver instead."
        )

    # Solution-based view
    if isinstance(x0, randvars.RandomVariable):
        raise NotImplementedError
    # Matrix-based view
    else:
        if "sym" in assume_A and "pos" in assume_A:
            if "noise" in assume_A:
                raise NotImplementedError
            else:
                return SymmetricMatrixBasedSolver(A=A, b=b, x0=x0, A0=A0, Ainv0=Ainv0)
        elif "sym" not in assume_A and "pos" in assume_A:
            raise NotImplementedError
        else:
            raise NotImplementedError


def _postprocess(info, A):
    """Postprocess the linear system and its solution.

    Raises exceptions or warnings based on the properties of the linear system and the
    solver iteration.

    Parameters
    ----------
    info : dict
        Convergence information output by a probabilistic linear solver.
    A : array-like or LinearOperator, shape=(n,n)
        A square linear operator (or matrix).

    Raises
    ------
    LinAlgError
        If the matrix ``A`` is singular.
    LinAlgWarning
        If an ill-conditioned system matrix ``A`` is detected.
    """
    rel_cond = info["rel_cond"]

    # Get the correct machine epsilon for the precision used.
    # if A.dtype.char in 'fF':  # single precision
    #     lamch = scipy.linalg.get_lapack_funcs('lamch', dtype='f')
    # else:
    #     lamch = scipy.linalg.get_lapack_funcs('lamch', dtype='d')
    # machine_eps = lamch('E')
    machine_eps = 10 ** -16

    # Singular matrix
    # # TODO: get info from solver
    # if False:
    #     raise scipy.linalg.LinAlgError("The system matrix A is singular.")
    # Ill-conditioned matrix A
    if rel_cond is not None and 1 / rel_cond < machine_eps:
        warnings.warn(
            (
                "Ill-conditioned matrix detected (estimated rcond={:.6g}). "
                "Results are likely inaccurate."
            ).format(rel_cond),
            scipy.linalg.LinAlgWarning,
            stacklevel=3,
        )
