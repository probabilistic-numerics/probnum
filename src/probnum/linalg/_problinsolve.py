"""Interface for the probabilistic linear solver."""

from typing import Optional, Tuple

import probnum.random_variables as rvs
from probnum.linalg.solvers import LinearSolverState, ProbabilisticLinearSolver
from probnum.problems import LinearSystem, NoisyLinearSystem
from probnum.type import MatrixArgType


def problinsolve(
    A: MatrixArgType,
    b: MatrixArgType,
    A0: Optional[MatrixArgType] = None,
    Ainv0: Optional[MatrixArgType] = None,
    x0: MatrixArgType = None,
    assume_linsys: str = "sympos",
    maxiter: Optional[int] = None,
    atol: float = 10 ** -6,
    rtol: float = 10 ** -6,
) -> Tuple[
    rvs.RandomVariable,
    rvs.RandomVariable,
    rvs.RandomVariable,
    rvs.RandomVariable,
    LinearSolverState,
]:
    """Solve the linear system :math:`A x = b` in a Bayesian framework.

    Probabilistic linear solvers infer solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`.
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
        products :math:`v \\mapsto Av` are used internally.
    b :
        *shape=(n, ) or (n, nrhs)* -- Right-hand side vector, matrix or random
        variable in :math:`A x = b`. For multiple right hand sides, ``nrhs`` problems
        are solved sequentially with the posteriors over the matrices acting as priors
        for subsequent solves. If the right-hand-side is assumed to be noisy, every
        iteration of the solver samples a realization from ``b``.
    A0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief about the linear operator :math:`A`.
    Ainv0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief about the inverse :math:`H=A^{-1}`. This can be
        viewed as a pre-conditioner.
    x0 :
        *shape=(n, ) or (n, nrhs)* -- Prior belief for the solution of the linear
        system. Will be ignored if ``Ainv0`` is given.
    assume_linsys :
        Assumptions on the linear system which can influence solver choice and
        behavior. The available options are (combinations of)

        =========================  =========
         generic matrix            ``gen``
         symmetric matrix          ``sym``
         positive definite matrix  ``pos``
         (additive) noise          ``noise``
        =========================  =========

    maxiter :
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol :
        Absolute convergence tolerance.
    rtol :
        Relative convergence tolerance.

    Returns
    -------
    x :
        Approximate solution :math:`x` to the linear system. Shape of the return
        matches the shape of ``b``.
    A :
        Posterior belief about the linear operator.
    Ainv :
        Posterior belief about the linear operator inverse :math:`H=A^{-1}`.
    b :
        Posterior belief about the rhs.
    solver_state :
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
    >>> rng = np.random.default_rng(42)
    >>> n = 20
    >>> A = random_spd_matrix(dim=n, random_state=rng)
    >>> b = rng.uniform(size=n)
    >>> x, _, _, _, solver_state = problinsolve(A=A, b=b)
    >>> solver_state.iteration
    10
    >>> np.linalg.norm(solver_state.residual)
    1.0691148648343433e-06
    """
    # pylint: disable=invalid-name,too-many-arguments
    if "noise" in assume_linsys:
        linsys = NoisyLinearSystem.from_randvars(A=rvs.asrandvar(A), b=rvs.asrandvar(b))
    else:
        linsys = LinearSystem(A=A, b=b)

    linear_solver = ProbabilisticLinearSolver.from_problem(
        problem=linsys,
        assume_linsys=assume_linsys,
        A0=A0,
        Ainv0=Ainv0,
        x0=x0,
        atol=atol,
        rtol=rtol,
        maxiter=maxiter,
    )
    belief, solver_state = linear_solver.solve(problem=linsys)

    return belief.x, belief.A, belief.Ainv, belief.b, solver_state
