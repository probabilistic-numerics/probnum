"""Probabilistic Numerical Method."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, TypeVar, Union

ProblemType = TypeVar("ProblemType")


class ProbabilisticNumericalMethod(ABC):
    """Probabilistic numerical methods.

    An abstract base class defining the implementation of a probabilistic numerical
    method [1,2]_. A PN method solves a numerical problem by treating it as a
    probabilistic inference task.

    All PN methods should subclass this base class. Typically convenience functions
    (such as :meth:`~probnum.linalg.problinsolve` will instantiate an object of a
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
    ProbabilisticLinearSolver : Probabilistic numerical method solving linear systems.
    """

    def __init__(
        self,
        prior: Tuple[Union["probnum.RandomVariable", "probnum.RandomProcess"], ...],
    ):
        self.prior = prior

    @abstractmethod
    def solve(
        self, problem: ProblemType
    ) -> Tuple[
        Tuple[Union["probnum.RandomVariable", "probnum.RandomProcess"], ...], Dict
    ]:
        """Solve the given numerical problem."""
        raise NotImplementedError
