"""Probabilistic linear solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

from typing import Generator, Tuple

import numpy as np

from probnum import ProbabilisticNumericalMethod, problems
from probnum.linalg.solvers import (
    belief_updates,
    beliefs,
    information_ops,
    policies,
    stopping_criteria,
)

from ._state import LinearSolverState


class ProbabilisticLinearSolver(
    ProbabilisticNumericalMethod[problems.LinearSystem, beliefs.LinearSystemBelief]
):
    r"""Compose a custom probabilistic linear solver.

    Class implementing probabilistic linear solvers. Such (iterative) solvers infer
    solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \in \mathbb{R}^{n \times n}` and :math:`b \in \mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources or stochastic input. This class unifies and
    generalizes probabilistic linear solvers as described in the literature. [1]_ [2]_
    [3]_ [4]_

    Parameters
    ----------
    policy
        Policy returning actions taken by the solver.
    information_op
        Information operator defining how information about the linear system is
        obtained given an action.
    belief_update
        Belief update defining how to update the QoI beliefs given new observations.
    stopping_criterion
        Stopping criterion determining when a desired terminal condition is met.

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
    ~probnum.linalg.problinsolve : Solve linear systems in a Bayesian framework.
    ~probnum.linalg.bayescg : Solve linear systems with prior information on the solution.

    Examples
    --------
    """

    def __init__(
        self,
        policy: policies.LinearSolverPolicy,
        information_op: information_ops.LinearSolverInformationOp,
        belief_update: belief_updates.LinearSolverBeliefUpdate,
        stopping_criterion: stopping_criteria.LinearSolverStoppingCriterion,
    ):
        self.policy = policy
        self.information_op = information_op
        self.belief_update = belief_update
        super().__init__(stopping_criterion=stopping_criterion)

    @classmethod
    def from_problem(
        cls, problem: problems.LinearSystem
    ) -> "ProbabilisticLinearSolver":
        """Construct a probabilistic linear solver from a linear system to be solved.

        Parameters
        ----------
        problem
            Linear system to be solved.
        """
        raise NotImplementedError

    def solve_iterator(
        self,
        prior: beliefs.LinearSystemBelief,
        problem: problems.LinearSystem,
        rng: np.random.Generator,
    ) -> Generator[LinearSolverState, None, None]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time
        and exposes the internal solver state.

        Parameters
        ----------
        prior
            Prior belief about the quantities of interest :math:`(x, A, A^{-1}, b)` of the linear system.
        problem
            Linear system to be solved.
        rng
            Random number generator.

        Yields
        ------
        solver_state
            State of the probabilistic linear solver.
        """
        solver_state = LinearSolverState(problem=problem, prior=prior, rng=rng)

        while True:

            yield solver_state

            # Compute action via policy
            action = self.policy(solver_state=solver_state)
            solver_state.action = action

            # Make observation via information operator
            observation = self.information_op(solver_state=solver_state)
            solver_state.observation = observation

            # Update the belief over the quantity of interest
            # updated_belief = self.belief_update(solver_state=solver_state)
            # TODO: assign updated belief to solver_state

            solver_state.next_step()

    def solve(
        self,
        prior: beliefs.LinearSystemBelief,
        problem: problems.LinearSystem,
        rng: np.random.Generator,
    ) -> Tuple[beliefs.LinearSystemBelief, LinearSolverState]:
        r"""Solve the linear system.

        Parameters
        ----------
        prior
            Prior belief about the quantities of interest :math:`(x, A, A^{-1}, b)` of the linear system.
        problem
            Linear system to be solved.
        rng
            Random number generator.

        Returns
        -------
        belief
            Posterior belief :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})`
            over the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state
            Final state of the solver.
        """
        solver_state = None

        for solver_state in self.solve_iterator(prior=prior, problem=problem, rng=rng):

            if self.stopping_criterion(solver_state=solver_state):
                break

        return solver_state.belief, solver_state
