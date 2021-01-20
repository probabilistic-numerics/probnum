"""Probabilistic Numerical Methods."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar

ProblemType = TypeVar("ProblemType")
BeliefType = TypeVar("BeliefType")
ProblemDataType = TypeVar("ProblemDataType")


@dataclasses.dataclass
class PNMethodHyperparams(ABC):
    """Hyperparameters of a probabilistic numerical method."""

    pass


@dataclasses.dataclass
class PNMethodInfo(ABC):
    """Information about the solve performed by the probabilistic numerical method."""

    pass


@dataclasses.dataclass
class PNMethodState(ABC, Generic[BeliefType]):
    """State of a probabilistic numerical method.

    The state is passed between different components of the
    algorithm and can be used to efficiently reuse already computed quantities.

    Parameters
    ----------
    info
        Information about the solve.
    problem :
        Numerical problem to be solved.
    belief :
        Belief over the quantities of interest.
    data :
        Data collected about the numerical problem.
    """

    info: PNMethodInfo
    problem: ProblemType
    belief: BeliefType
    data: ProblemDataType


class ProbabilisticNumericalMethod(ABC, Generic[ProblemType, BeliefType]):
    """Probabilistic numerical methods.

    An abstract base class defining the implementation of a probabilistic numerical
    method [1]_ [2]_. A PN method solves a numerical problem by treating it as a
    probabilistic inference task.

    All PN methods should subclass this base class. Typically convenience functions
    (such as :meth:`~probnum.linalg.problinsolve`) will instantiate an object of a
    derived subclass.

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

    See Also
    --------
    ~probnum.linalg.solvers.ProbabilisticLinearSolver : Compose a custom
        probabilistic linear solver.
    """

    def __init__(self, prior: BeliefType):
        self.prior = prior

    @abstractmethod
    def solve(
        self,
        problem: ProblemType,
    ) -> Tuple[BeliefType, PNMethodState]:
        """Solve the given numerical problem.

        Parameters
        ----------
        problem :
            Numerical problem to be solved.
        """
        raise NotImplementedError
