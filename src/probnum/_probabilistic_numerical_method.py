"""Probabilistic Numerical Methods."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar

ProblemType = TypeVar("ProblemType")
BeliefType = TypeVar("BeliefType")
ActionType = TypeVar("ActionType")
ObservationType = TypeVar("ObservationType")


@dataclasses.dataclass
class PNMethodData(ABC, Generic[ActionType, ObservationType]):
    """Data about the numerical problem collected by a probabilistic numerical method.

    Parameters
    ----------
    actions
        Performed actions.
    observations
        Collected observations of the problem.
    """

    actions: Optional[List[ActionType]] = None
    observations: Optional[List[ObservationType]] = None


@dataclasses.dataclass
class PNMethodHyperparams(ABC):
    """Hyperparameters of a probabilistic numerical method."""

    pass


@dataclasses.dataclass
class PNMethodState(ABC, Generic[BeliefType]):
    """State of a probabilistic numerical method.

    The state of a PN method contains the belief about the quantities of
    interest of the numerical problem -- such as the solution -- and the data
    collected by the method. The state is passed between different components of the
    algorithm and can be used to reuse miscellaneous computed quantities.

    Parameters
    ----------
    belief
        Belief about the quantities of interest of the numerical problem.
    data
        Collected data from the numerical problem.
    """

    belief: Optional[BeliefType] = None
    data: Optional[PNMethodData] = None


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
        """Solve the given numerical problem."""
        raise NotImplementedError
