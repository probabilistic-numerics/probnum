"""Probabilistic Numerical Methods."""

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

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
    prior :
        Prior knowledge about quantities of interest for the numerical problem.

    References
    ----------
    .. [1] Hennig, P., Osborne, Mike A. and Girolami M., Probabilistic numerics and
       uncertainty in computations. *Proceedings of the Royal Society of London A:
       Mathematical, Physical and Engineering Sciences*, 471(2179), 2015.
    .. [2] Cockayne, J., Oates, C., Sullivan Tim J. and Girolami M., Bayesian
       probabilistic numerical methods. *SIAM Review*, 61(4):756–789, 2019

    Notes
    -----
    All PN methods should subclass this base class. Typically convenience functions
    (such as :meth:`~probnum.linalg.problinsolve`) will instantiate an object of a
    derived subclass.
    """

    def __init__(self, prior: BeliefType):
        self.prior = prior

    @abstractmethod
    def solve(
        self,
        problem: ProblemType,
    ) -> Tuple[BeliefType, StateType]:
        """Solve the given numerical problem.

        Parameters
        ----------
        problem :
            Numerical problem to be solved.
        """
        raise NotImplementedError
