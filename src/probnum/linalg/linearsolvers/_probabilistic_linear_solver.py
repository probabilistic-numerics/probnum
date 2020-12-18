"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

from typing import Generator, List, Optional, Tuple

import probnum.random_variables as rvs
from probnum._probabilistic_numerical_method import ProbabilisticNumericalMethod
from probnum.problems import LinearSystem

from ._linear_solver_state import LinearSolverState
from ._policies import LinearSolverPolicy
from ._stopping_criteria import StoppingCriterion

# pylint: disable="invalid-name"


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
        Prior belief over the quantities of interest :math:`(x, A, A^{-1})` of the
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
        prior: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
        policy: LinearSolverPolicy,
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

    def _init_solver_state(self, problem: LinearSystem) -> LinearSolverState:
        """Initialize the solver state.

        Constructs the initial solver state depending on the given components. This
        has the benefit that only needed quantities are computed during the solver
        iteration.

        Parameters
        ----------
        problem :
            Linear system to solve.
        """
        # TODO: determine here depending on components what initialization to use.
        return LinearSolverState(
            belief=self.prior,
            actions=[],
            observations=[],
            iteration=0,
            residual=problem.A @ self.prior[0].mean - problem.b,
            rayleigh_quotients=[],
            has_converged=False,
            stopping_criterion=None,
        )

    def has_converged(
        self, problem: LinearSystem, solver_state: LinearSolverState
    ) -> Tuple[bool, LinearSolverState]:
        """Check whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
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
            _has_converged = stopping_criterion(problem, solver_state)
            if _has_converged:
                solver_state.has_converged = True
                solver_state.stopping_criterion = stopping_criterion.__class__.__name__
                return True, solver_state
        return False, solver_state

    def solve_iterator(
        self, problem: LinearSystem, solver_state: LinearSolverState
    ) -> Generator[LinearSolverState, None, None]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time.

        Parameters
        ----------
        problem :
            Linear system to solve.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        solver_state :
            Updated state of the linear solver.
        """

        while True:
            # Compute action via policy
            action = self.policy(problem, solver_state)
            solver_state.actions.append(action)

            # Make an observation of the linear system
            observation = self.observe(problem, action)
            solver_state.observations.append(observation)

            # Update the belief over the system matrix, its inverse and/or the solution
            solver_state.belief = self.update_belief(solver_state, action, observation)

            yield solver_state

    def solve(
        self,
        problem: LinearSystem,
    ) -> Tuple[
        Tuple[
            rvs.RandomVariable,
            rvs.RandomVariable,
            rvs.RandomVariable,
            rvs.RandomVariable,
        ],
        LinearSolverState,
    ]:
        """Solve the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.

        Returns
        -------
        """
        # Setup
        solver_state = self._init_solver_state(problem=problem)

        # Evaluate stopping criteria for the prior
        _has_converged, solver_state = self.has_converged(
            problem=problem, solver_state=solver_state
        )

        # Solver iteration
        solve_iterator = self.solve_iterator(problem=problem, solver_state=solver_state)

        for sost in solve_iterator:

            # Evaluate stopping criteria and update solver state
            _has_converged, solver_state = self.has_converged(
                problem=problem, solver_state=sost
            )

        # Belief over solution, system matrix, its inverse and the right hand side
        x, A, Ainv, b = self._infer_belief_system_components(solver_state=solver_state)

        return (x, A, Ainv, b), solver_state

    def _infer_belief_system_components(self, solver_state: LinearSolverState):
        """Compute the belief over all components of the linear system."""
        raise NotImplementedError
