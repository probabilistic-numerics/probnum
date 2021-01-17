"""Interface for the probabilistic linear solver."""

from typing import Optional, Tuple

import scipy.sparse

import probnum
import probnum.random_variables as rvs
from probnum import linops
from probnum.linalg.linearsolvers import (
    LinearSolverState,
    ProbabilisticLinearSolver,
    beliefs,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.problems import LinearSystem
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
        representing the prior belief over the linear operator :math:`A`.
    Ainv0 :
        *shape=(n, n)* -- A square matrix, linear operator or random variable
        representing the prior belief over the inverse :math:`H=A^{-1}`. This can be
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
        Posterior belief over the linear operator.
    Ainv :
        Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
    b :
        Posterior belief over the rhs.
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
    linsys = LinearSystem(A=A, b=b)
    prior_belief = _init_prior_belief(
        linsys=linsys, A0=A0, Ainv0=Ainv0, x0=x0, assume_linsys=assume_linsys
    )
    linear_solver = _init_solver(
        prior_belief=prior_belief, atol=atol, rtol=rtol, maxiter=maxiter
    )
    belief, solver_state = linear_solver.solve(problem=linsys)

    return belief.x, belief.A, belief.Ainv, belief.b, solver_state


def _init_prior_belief(
    linsys: LinearSystem,
    A0: Optional[MatrixArgType] = None,
    Ainv0: Optional[MatrixArgType] = None,
    x0: MatrixArgType = None,
    assume_linsys: str = "sympos",
) -> beliefs.LinearSystemBelief:
    """Initialize a prior belief over the linear system.

    Automatically chooses an appropriate prior belief over the linear system components
    based on the arguments given.

    Parameters
    ----------
    linsys :
        Linear system to solve.
    A0 :
        A square matrix, linear operator or random variable representing the prior
        belief over the linear operator :math:`A`.
    Ainv0 :
        A square matrix, linear operator or random variable representing the prior
        belief over the inverse :math:`H=A^{-1}`.
    x0 :
        Optional. Prior belief for the solution of the linear system. Will be ignored if
        ``Ainv0`` is given.
    assume_linsys :
        Assumptions on the linear system which can influence solver choice and
        behavior. The available options are (combinations of)

        =========================  =========
         generic matrix            ``gen``
         symmetric matrix          ``sym``
         positive definite matrix  ``pos``
         (additive) noise          ``noise``
        =========================  =========


    Raises
    ------
    ValueError
        If type or size mismatches detected or inputs ``A`` and ``Ainv`` are not square.
    """
    # Check matrix assumptions for correctness
    assume_linsys = assume_linsys.lower()
    _assume_A_tmp = assume_linsys
    for allowed_str in ["gen", "sym", "pos", "noise"]:
        _assume_A_tmp = _assume_A_tmp.replace(allowed_str, "")
    if _assume_A_tmp != "":
        raise ValueError(
            "Assumption '{}' contains unrecognized linear operator properties.".format(
                assume_linsys
            )
        )

    # Choose matrix based view if not clear from arguments
    if (Ainv0 is not None or A0 is not None) and isinstance(x0, rvs.RandomVariable):
        x0 = None

    # Extract information from system and priors
    # System matrix is symmetric
    if isinstance(A0, rvs.RandomVariable):
        if isinstance(A0.cov, linops.SymmetricKronecker) and "sym" not in assume_linsys:
            assume_linsys += "sym"
    if isinstance(Ainv0, rvs.RandomVariable):
        if (
            isinstance(Ainv0.cov, linops.SymmetricKronecker)
            and "sym" not in assume_linsys
        ):
            assume_linsys += "sym"
    # System matrix or right hand side is stochastic
    if (
        isinstance(linsys.A, rvs.RandomVariable)
        or isinstance(linsys.b, rvs.RandomVariable)
        and "noise" not in assume_linsys
    ):
        assume_linsys += "noise"

    # Choose belief class
    belief_class = beliefs.LinearSystemBelief
    if "sym" in assume_linsys and "pos" in assume_linsys:
        if "noise" in assume_linsys:
            belief_class = beliefs.NoisyLinearSystemBelief
        else:
            belief_class = beliefs.WeakMeanCorrespondenceBelief
    elif "sym" in assume_linsys and "pos" not in assume_linsys:
        belief_class = beliefs.SymmetricLinearSystemBelief

    # Instantiate belief from available prior information
    if x0 is None and A0 is not None and Ainv0 is not None:
        return belief_class.from_matrices(A0=A0, Ainv0=Ainv0, problem=linsys)
    elif Ainv0 is not None:
        return belief_class.from_inverse(Ainv0=Ainv0, problem=linsys)
    elif A0 is not None:
        return belief_class.from_matrix(A0=A0, problem=linsys)
    elif x0 is not None:
        return belief_class.from_solution(x0=x0, problem=linsys)
    else:
        return belief_class.from_scalar(scalar=1.0, problem=linsys)


def _init_solver(
    prior_belief: beliefs.LinearSystemBelief,
    maxiter: int,
    atol: float,
    rtol: float,
) -> ProbabilisticLinearSolver:
    """Initialize a custom probabilistic linear solver.

    Selects and initializes an appropriate instance of the probabilistic linear
    solver based on the prior information given.

    Parameters
    ----------
    prior_belief :
        Prior belief over the quantities of interest of the linear system.
    maxiter :
        Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
        dimension of :math:`A`.
    atol :
        Absolute convergence tolerance.
    rtol :
        Relative convergence tolerance.
    """

    observation_op = observation_ops.MatVecObservation()
    stopping_criteria = [stop_criteria.MaxIterations(maxiter=maxiter)]
    if isinstance(
        prior_belief,
        (beliefs.SymmetricLinearSystemBelief, beliefs.WeakMeanCorrespondenceBelief),
    ):
        policy = policies.ConjugateDirections()
        stopping_criteria.append(stop_criteria.Residual(atol=atol, rtol=rtol))
    elif isinstance(prior_belief, beliefs.NoisyLinearSystemBelief):
        policy = policies.ExploreExploit()
        stopping_criteria.append(
            stop_criteria.PosteriorContraction(atol=atol, rtol=rtol)
        )
    elif isinstance(prior_belief, beliefs.LinearSystemBelief):
        policy = policies.ConjugateDirections()
        stopping_criteria.append(stop_criteria.Residual(atol=atol, rtol=rtol))
    else:
        raise ValueError("Unknown prior belief class.")

    return ProbabilisticLinearSolver(
        prior=prior_belief,
        policy=policy,
        observation_op=observation_op,
        stopping_criteria=stopping_criteria,
    )
