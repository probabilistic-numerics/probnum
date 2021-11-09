"""Probabilistic linear solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

from typing import Generator, Tuple

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
        prior: beliefs.LinearSystemBelief,
        policy: policies.LinearSolverPolicy,
        information_op: information_ops.LinearSolverInformationOp,
        belief_update: belief_updates.LinearSolverBeliefUpdate,
        stopping_criterion: stopping_criteria.LinearSolverStoppingCriterion,
    ):
        self.policy = policy
        self.information_op = information_op
        self.belief_update = belief_update
        super().__init__(prior=prior, stopping_criterion=stopping_criterion)

    @classmethod
    def from_problem(cls) -> "ProbabilisticLinearSolver":
        """Construct a probabilistic linear solver from a linear system to be solved.

        Parameters
        ----------
        problem :
            Linear system.
        """
        raise NotImplementedError

    def solve_iterator(self) -> Generator[LinearSolverState]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time
        and exposes the internal solver state.
        """
        raise NotImplementedError

    def solve(
        self, problem: problems.LinearSystem
    ) -> Tuple[beliefs.LinearSystemBelief, LinearSolverState]:
        r"""Solve the linear system.

        Parameters
        ----------
        problem :
            Linear system.

        Returns
        -------
        belief :
            Posterior belief :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})`
            over the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Final state of the solver.
        """
        solver_state = None

        for solver_state in self.solve_iterator(problem):
            pass

        return solver_state.belief, solver_state
