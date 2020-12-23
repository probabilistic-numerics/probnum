"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

import dataclasses
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum._probabilistic_numerical_method import (
    PNMethodState,
    ProbabilisticNumericalMethod,
)
from probnum.problems import LinearSystem

from .policies import Policy
from .stop_criteria import StoppingCriterion

# pylint: disable="invalid-name"


@dataclasses.dataclass
class LinearSystemBelief:
    r"""Belief over quantities of interest of a linear system.

    Random variables :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})` modelling
    the solution :math:`x`, the system matrix :math:`A`, its inverse :math:`H=A^{-1}`
    and the right hand side :math:`b` of a linear system :math:`Ax=b`.

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the inverse of the system matrix.
    b :
        Belief over the right hand side
    """
    x: rvs.RandomVariable
    A: rvs.RandomVariable
    Ainv: rvs.RandomVariable
    b: rvs.RandomVariable

    def __post_init__(self):
        # Check and normalize shapes
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1, 1))
        if self.x.ndim == 1:
            self.x = self.x.reshape((-1, 1))
        if self.x.ndim != 2:
            raise ValueError("Belief over solution must be two-dimensional.")
        if self.A.ndim != 2 or self.Ainv.ndim != 2 or self.b.ndim != 2:
            raise ValueError("Beliefs over system components must be two-dimensional.")

        # Check shape mismatch
        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if self.A.shape[0] != self.b.shape[0]:
            raise dim_mismatch_error(self.A, self.b, "A", "b")

        if self.A.shape[0] != self.x.shape[0]:
            raise dim_mismatch_error(self.A, self.x, "A", "x")

        if self.x.shape[1] != self.b.shape[1]:
            raise dim_mismatch_error(self.x, self.b, "x", "b")

        if self.A.shape != self.Ainv.shape:
            raise dim_mismatch_error(self.A, self.Ainv, "A", "Ainv")

    # TODO: add different classmethods here to construct standard beliefs, i.e. from
    #  deterministic arguments (preconditioner), from a prior on the solution,
    #  from just an inverse prior, etc.

    def from_x0(self, problem: LinearSystem, x0: np.ndarray) -> LinearSystem:
        """Create matrix prior means from an initial guess for the solution of the
        linear system.

        Constructs a matrix-variate prior mean for :math:`H` from ``x0`` and ``b`` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`.

        Parameters
        ----------
        problem :
            Linear system to solve.
        x0 :
            Initial guess for the solution of the linear system.

        Returns
        -------
        A0_mean :
            Mean of the matrix-variate prior distribution on the system matrix
            :math:`A`.
        Ainv0_mean :
            Mean of the matrix-variate prior distribution on the inverse of the system
            matrix :math:`H = A^{-1}`.
        """
        # Check inner product between x0 and b; if negative or zero, choose better
        # initialization

        bx0 = np.squeeze(problem.b.T @ x0)
        bb = np.linalg.norm(problem.b) ** 2
        if bx0 < 0:
            x0 = -x0
            bx0 = -bx0
            print("Better initialization found, setting x0 = - x0.")
        elif bx0 == 0:
            if np.all(problem.b == np.zeros_like(problem.b)):
                print("Right-hand-side is zero. Initializing with solution x0 = 0.")
                x0 = problem.b
            else:
                print("Better initialization found, setting x0 = (b'b/b'Ab) * b.")
                bAb = np.squeeze(problem.b.T @ (problem.A @ problem.b))
                x0 = bb / bAb * problem.b
                bx0 = bb ** 2 / bAb

        # Construct prior mean of A and H
        alpha = 0.5 * bx0 / bb

        def _mv(v):
            return (x0 - alpha * problem.b) * (x0 - alpha * problem.b).T @ v

        def _mm(M):
            return (x0 - alpha * problem.b) @ (x0 - alpha * problem.b).T @ M

        Ainv0_mean = linops.ScalarMult(
            scalar=alpha, shape=problem.A.shape
        ) + 2 / bx0 * linops.LinearOperator(
            matvec=_mv, matmat=_mm, shape=problem.A.shape
        )
        A0_mean = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape) - 1 / (
            alpha * np.squeeze((x0 - alpha * problem.b).T @ x0)
        ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=problem.A.shape)

        # TODO: what covariance should be returned for this prior mean?


@dataclasses.dataclass
class LinearSolverState(PNMethodState):
    r"""State of a probabilistic linear solver.

    The solver state contains miscellaneous quantities computed during an iteration
    of a probabilistic linear solver. The solver state is passed between the
    different components of the solver and may be used by them.

    For example the residual :math:`r_i = Ax_i - b` can (depending on the prior) be
    updated more efficiently than in :math:`\mathcal{O}(n^2)` and is therefore part
    of the solver state and passed to the stopping criteria.

    Parameters
    ----------
    actions
        Performed actions :math:`s_i`.
    observations
        Collected observations :math:`y_i = A s_i`.
    iteration
        Current iteration :math:`i` of the solver.
    residual
        Residual :math:`r_i = Ax_i - b` of the current solution.
    log_rayleigh_quotients
        Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(s_i^\top
        s_i)`.
    step_sizes
        Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer taking
        steps :math:`x_{i+1} = x_i + \alpha_i s_i`.
    has_converged
        Has the solver converged?
    stopping_criterion
        Stopping criterion which caused termination of the solver.

    Examples
    --------

    """
    iteration: int = 0
    residual: Optional[Union[np.ndarray, rvs.RandomVariable]] = None
    log_rayleigh_quotients: Optional[List[float]] = None
    step_sizes: Optional[List[float]] = None
    has_converged: bool = False
    stopping_criterion: Optional[List[StoppingCriterion]] = None


class ProbabilisticLinearSolver(ProbabilisticNumericalMethod):
    """Compose a custom probabilistic linear solver.

    Class implementing probabilistic linear solvers. Such (iterative) solvers infer
    solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources or stochastic input. This class unifies and
    generalizes probabilistic linear solvers as described in the literature [1]_ [2]_
    [3]_ [4]_.

    Parameters
    ----------
    prior :
        Prior belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    policy :
        Policy defining actions taken by the solver.
    observe :
        Observation process defining how information about the linear system is
        obtained.
    update_belief :
        Operator updating the belief over the quantities of interest :math:`(x, A,
        A^{-1})` of the linear system.
    stopping_criteria :
        Stopping criteria determining when the solver has converged.
    optimize_hyperparams :
        Function optimizing hyperparameters of the solver.

    References
    ----------
    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
       Optimization*, 2015, 25, 234-260
    .. [2] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
       *Statistics and Computing*, 2019
    .. [4] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    bayescg : Solve linear systems with prior information on the solution.

    Examples
    --------
    Create a custom probabilistic linear solver from pre-defined components.

    >>> from probnum.linalg.linearsolvers import ProbabilisticLinearSolver
    >>> #pls = ProbabilisticLinearSolver()

    Define a linear system.

    >>> import numpy as np
    >>> import probnum as pn
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> n = 20
    >>> A = random_spd_matrix(dim=n, random_state=1)
    >>> b = np.random.rand(n)

    Solve the linear system using the custom solver.

    >>> #sol, info = pls(A, b)
    """

    def __init__(
        self,
        prior: LinearSystemBelief,
        policy: Policy,
        observe,
        update_belief,
        stopping_criteria=Optional[List[StoppingCriterion]],
        optimize_hyperparams=None,
    ):
        # pylint: disable="too-many-arguments"
        self.policy = policy
        self.observe = observe
        self.update_belief = update_belief
        self.stopping_criteria = stopping_criteria
        self.optimize_hyperparams = optimize_hyperparams
        super().__init__(
            prior=prior,
        )

    def _init_belief_and_solver_state(
        self, problem: LinearSystem
    ) -> Tuple[LinearSystemBelief, LinearSolverState]:
        """Initialize the belief and solver state.

        Constructs the initial belief and solver state depending on the given
        components. This has the benefit that only needed quantities are computed
        during the solver iteration.

        Parameters
        ----------
        problem :
            Linear system to solve.
        """
        belief = self.prior
        solver_state = LinearSolverState(
            actions=[],
            observations=[],
            iteration=0,
            residual=problem.A @ self.prior.x.mean - problem.b,
            log_rayleigh_quotients=[],
            step_sizes=[],
            has_converged=False,
            stopping_criterion=None,
        )
        return belief, solver_state

    def has_converged(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        solver_state: LinearSolverState,
    ) -> Tuple[bool, LinearSolverState]:
        """Check whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        has_converged :
            True if the method has converged.
        solver_state :
            Updated state of the solver.
        """
        if solver_state.has_converged:
            return True, solver_state

        # Check stopping criteria
        for stopping_criterion in self.stopping_criteria:
            _has_converged = stopping_criterion(problem, belief, solver_state)
            if _has_converged:
                solver_state.has_converged = True
                solver_state.stopping_criterion = stopping_criterion.__class__.__name__
                return True, solver_state
        return False, solver_state

    def solve_iterator(
        self,
        problem: LinearSystem,
        belief: LinearSystemBelief,
        solver_state: LinearSolverState,
    ) -> Generator[Tuple[LinearSystemBelief, LinearSolverState], None, None]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        belief :
            Updated belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of
            the linear system.
        solver_state :
            Updated state of the linear solver.
        """

        while True:
            # Compute action via policy
            action, solver_state = self.policy(
                problem=problem, belief=belief, solver_state=solver_state
            )

            # Make an observation of the linear system
            observation, solver_state = self.observe(
                problem=problem, action=action, solver_state=solver_state
            )

            # Update the belief over the system matrix, its inverse and/or the solution
            belief, solver_state = self.update_belief(
                problem=problem,
                belief=belief,
                action=action,
                observation=observation,
                solver_state=solver_state,
            )

            yield belief, solver_state

    def solve(
        self,
        problem: LinearSystem,
    ) -> Tuple[LinearSystemBelief, LinearSolverState]:
        """Solve the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.

        Returns
        -------
        belief : Posterior belief `(x, A, Ainv, b)` over the solution :math:`x`,
                 the system matrix :math:`A`, its inverse :math:`H=A^{-1}` and the
                 right hand side :math:`b`.
        solver_state : State of the solver at convergence.
        """
        # Setup
        belief, solver_state = self._init_belief_and_solver_state(problem=problem)

        # Evaluate stopping criteria for the prior
        _has_converged, solver_state = self.has_converged(
            problem=problem, belief=belief, solver_state=solver_state
        )

        # Solver iteration
        solve_iterator = self.solve_iterator(
            problem=problem, belief=belief, solver_state=solver_state
        )

        for (belief, solver_state) in solve_iterator:

            # Evaluate stopping criteria and update solver state
            _has_converged, solver_state = self.has_converged(
                problem=problem,
                belief=belief,
                solver_state=solver_state,
            )

        # Belief over solution, system matrix, its inverse and the right hand side
        belief = self._infer_belief_system_components(belief=belief)

        return belief, solver_state

    def _infer_belief_system_components(
        self, belief: LinearSystemBelief
    ) -> LinearSystemBelief:
        """Compute the belief over all quantities of interest of the linear system."""
        raise NotImplementedError
