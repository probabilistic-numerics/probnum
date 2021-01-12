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

import probnum.random_variables as rvs
from probnum import linops, utils
from probnum.linalg.linearsolvers import LinearSolverState, ProbabilisticLinearSolver
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    NoisyLinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem

# Type aliases
SquareLinOp = Union[
    np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator, "rvs.RandomVariable"
]
RandomVecMat = Union[np.ndarray, "rvs.RandomVariable"]


def problinsolve(
    A: SquareLinOp,
    b: RandomVecMat,
    A0: Optional[SquareLinOp] = None,
    Ainv0: Optional[SquareLinOp] = None,
    x0: Optional[RandomVecMat] = None,
    assume_A: str = "sympos",
    maxiter: Optional[int] = None,
    atol: float = 10 ** -6,
    rtol: float = 10 ** -6,
    **kwargs
) -> Tuple[
    "rvs.RandomVariable",
    "rvs.RandomVariable",
    "rvs.RandomVariable",
    "rvs.RandomVariable",
    LinearSolverState,
]:
    """Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources or stochastic input. This solver can take prior
    information either on the linear operator :math:`A` or its inverse :math:`H=A^{
    -1}` in the form of a random variable ``A0`` or ``Ainv0`` and outputs a posterior
    belief over :math:`A` or :math:`H`. This code implements the method described in
    Wenger et al. [1]_ based on the work in Hennig et al. [2]_.

    Parameters
    ----------
    A :
        *shape=(n, n)* -- A square linear operator (or matrix). Only matrix-vector
        products :math:`v \\mapsto Av` are used internally.
    b :
        *shape=(n, ) or (n, nrhs)* -- Right-hand side vector, matrix or random
        variable in :math:`A x = b`. For multiple right hand sides, ``nrhs`` problems
        are solved sequentially with the posteriors over the matrices acting as priors
        for subsequent solves. If the right-hand-side is assumed to be noisy, every
        iteration of the solver samples a realization from ``b``.
    A0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief over the linear operator :math:`A`. If an array or
        linear operator is given, a prior distribution is chosen automatically.
    Ainv0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief over the inverse :math:`H=A^{-1}`. This can be
        viewed as taking the form of a pre-conditioner. If an array or linear operator
        is given, a prior distribution is chosen automatically.
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
    kwargs : optional
        Optional keyword arguments passed onto the solver iteration.

    Returns
    -------
    x :
        Approximate solution :math:`x` to the linear system. Shape of the return
        matches the shape of ``b``.
    A :
        Posterior belief over the linear operator.
    Ainv :
        Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
    b :
        Posterior belief over the rhs.
    state :
        State of the linear solver at convergence.

    Raises
    ------
    ValueError
        If size mismatches detected or input matrices are not square.

    Notes
    -----
    For a specific class of priors the posterior mean of :math:`x_k` coincides with
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
    >>> import probnum as pn
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> n = 20
    >>> A = random_spd_matrix(dim=n, random_state=1)
    >>> b = np.random.rand(n)
    >>> x, _, _, _, info = problinsolve(A=A, b=b)
    >>> info.iteration
    11
    >>> info.residual
    """
    linsys = LinearSystem(A=A, b=b)
    belief = _init_prior_belief(A0=A0, Ainv0=Ainv0, x0=x0, assume_A=assume_A)
    linear_solver = _init_solver(
        linsys=linsys, belief=belief, atol=atol, rtol=rtol, maxiter=maxiter
    )
    belief, solver_state = linear_solver.solve(problem=linsys)

    return belief.x, belief.A, belief.Ainv, belief.b, solver_state


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
        called as ``callback(x, s, y, alpha, resid, **kwargs)`` and can be used to
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


def _init_prior_belief(
    A0: Optional[SquareLinOp] = None,
    Ainv0: Optional[SquareLinOp] = None,
    x0: Optional[RandomVecMat] = None,
    assume_A: str = "sympos",
) -> LinearSystemBelief:
    """Initialize a prior belief over the linear system.

    Automatically chooses an appropriate prior belief over the linear system components
    based on the arguments given.

    Parameters
    ----------
    A0 :
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`.
    Ainv0 :
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`.
    x0 :
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.
    assume_A :
        Assumptions on the linear operator which can influence solver choice and
        behavior. The available options are (combinations of)

        ====================  =========
         generic matrix       ``gen``
         symmetric            ``sym``
         positive definite    ``pos``
         (additive) noise     ``noise``
        ====================  =========


    Raises
    ------
    ValueError
        If type or size mismatches detected or inputs ``A`` and ``Ainv`` are not square.
    """
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

    raise NotImplementedError


def _init_solver(
    linsys: LinearSystem,
    belief: LinearSystemBelief,
) -> ProbabilisticLinearSolver:
    """Initialize a custom probabilistic linear solver.

    Selects and initializes an appropriate instance of the probabilistic linear
    solver based on the system properties and prior information given.

    Parameters
    ----------
    linsys : Linear system to solve.
    belief : (Prior) belief over the quantities of interest of the linear system.
    """
    # Choose matrix based view if not clear from arguments
    if (Ainv0 is not None or A0 is not None) and isinstance(x0, rvs.RandomVariable):
        warnings.warn(
            "Cannot use prior uncertainty on both the matrix (inverse) and the "
            "solution. The latter will be ignored."
        )
        x0 = x0.mean

    # Extract information from priors
    # System matrix is symmetric
    if isinstance(A0, rvs.RandomVariable):
        if isinstance(A0.cov, linops.SymmetricKronecker) and "sym" not in assume_A:
            assume_A += "sym"
    if isinstance(Ainv0, rvs.RandomVariable):
        if isinstance(Ainv0.cov, linops.SymmetricKronecker) and "sym" not in assume_A:
            assume_A += "sym"
    # System matrix is NOT stochastic
    if (
        not isinstance(A, rvs.RandomVariable)
        and not isinstance(A, scipy.sparse.linalg.LinearOperator)
        and "noise" in assume_A
    ):
        warnings.warn(
            "A is assumed to be noisy, but is neither a random variable nor a "
            "linear operator. Use exact probabilistic linear solver instead."
        )

    # Solution-based view
    if isinstance(x0, rvs.RandomVariable):
        raise NotImplementedError
    # Matrix-based view
    else:
        if "sym" in assume_A and "pos" in assume_A:
            if "noise" in assume_A:
                return NoisySymmetricMatrixBasedSolver(
                    A=A, b=b, x0=x0, A0=A0, Ainv0=Ainv0
                )
            else:
                return SymmetricMatrixBasedSolver(A=A, b=b, x0=x0, A0=A0, Ainv0=Ainv0)
        elif "sym" not in assume_A and "pos" in assume_A:
            return AsymmetricMatrixBasedSolver(A=A, b=b, x0=x0)
        else:
            raise NotImplementedError
