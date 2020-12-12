"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax_* = b`.
"""

from typing import Tuple, Union

import numpy as np

import probnum.random_variables as rvs
from probnum import ProbabilisticNumericalMethod
from probnum.type import IntArgType


class ProbabilisticLinearSolver(ProbabilisticNumericalMethod):
    """Compose a custom probabilistic linear solver.

    Class implementing probabilistic linear solvers. Such (iterative) solvers infer
    solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources. This class unifies and generalizes the methods
    described in in Hennig et al. [1]_, Cockayne et al. [2]_, Bartels et al. [3]_ and
    Wenger et al. [4]_.

    Parameters
    ----------
    prior :
    policy :
    observe :
    update_belief :
    stopping_criteria :
    optimize_hyperparams :

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
    >>> from probnum.linalg import ProbabilisticLinearSolver
    """

    def __init__(
        self,
        prior,
        policy,
        observe,
        update_belief,
        stopping_criteria=None,
        optimize_hyperparams=None,
    ):
        # pylint: disable="too-many-arguments"
        self.belief = prior
        self.policy = policy
        self.observe = observe
        self.update_belief = update_belief
        self.stopping_criteria = stopping_criteria
        self.optimize_hyperparams = optimize_hyperparams
        super().__init__(
            prior=prior,
        )

    def has_converged(
        self, problem: "LinearSystem", iteration: IntArgType
    ) -> Tuple[bool, Union[str, None]]:
        """Check whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
        iteration :
            Number of iterations of the solver performed up to this point.

        Returns
        -------
        has_converged :
            True if the method has converged.
        convergence_criterion :
            Convergence criterion which caused termination.
        """
        for stopping_criterion in self.stopping_criteria:
            _has_converged, convergence_criterion = stopping_criterion(
                problem, self.belief, iteration
            )
            if _has_converged:
                return True, convergence_criterion
        return False, None

    def solve_iterator(
        self, problem: "LinearSystem"
    ) -> Tuple[np.array, np.array, rvs.RandomVariable]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time.

        Parameters
        ----------
        problem :
            Linear system to solve.

        Returns
        -------
        action :
            Action to probe the problem.
        observation :
            Observation of the problem for the given ``action``.
        fun_params :
            Belief over the parameters of the objective function.
        """
        while True:
            # Compute action via policy
            action = self.policy(problem, self.belief)

            # Make an observation of the linear system
            observation = self.observe(problem, action)

            # Update the belief over the system matrix, its inverse or the solution
            self.belief = self.update_belief(self.belief, action, observation)

            yield action, observation, self.belief

    def solve(self, problem: "LinearSystem"):
        raise NotImplementedError
