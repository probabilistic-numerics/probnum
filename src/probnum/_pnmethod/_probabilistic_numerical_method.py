"""Probabilistic Numerical Methods."""

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

from ._stopping_criterion import StoppingCriterion

ProblemType = TypeVar("ProblemType")
BeliefType = TypeVar("BeliefType")
StateType = TypeVar("StateType")


class ProbabilisticNumericalMethod(ABC, Generic[ProblemType, BeliefType]):
    """Probabilistic numerical methods.

    An abstract base class defining the implementation of a probabilistic numerical
    method [1]_ [2]_. A PN method solves a numerical problem by treating it as a
    probabilistic inference task.

    Parameters
    ----------
    stopping_criterion
        Stopping criterion determining when a desired terminal condition is met.

    References
    ----------
    .. [1] Hennig, P., Osborne, Mike A. and Girolami M., Probabilistic numerics and
       uncertainty in computations. *Proceedings of the Royal Society of London A:
       Mathematical, Physical and Engineering Sciences*, 471(2179), 2015.
    .. [2] Cockayne, J., Oates, C., Sullivan Tim J. and Girolami M., Bayesian
       probabilistic numerical methods. *SIAM Review*, 61(4):756–789, 2019

    See Also
    --------
    ~probnum.linalg.solvers.ProbabilisticLinearSolver : Compose a custom
        probabilistic linear solver.

    Notes
    -----
    All PN methods should subclass this base class. Typically convenience functions
    (such as :meth:`~probnum.linalg.problinsolve`) will instantiate an object of a
    derived subclass.
    """

    def __init__(self, stopping_criterion: StoppingCriterion):
        self.stopping_criterion = stopping_criterion

    @abstractmethod
    def solve(
        self, prior: BeliefType, problem: ProblemType, **kwargs
    ) -> Tuple[BeliefType, StateType]:
        """Solve the given numerical problem.

        Parameters
        ----------
        prior :
            Prior knowledge about quantities of interest of the numerical problem.
        problem :
            Numerical problem to be solved.
        """
        raise NotImplementedError
