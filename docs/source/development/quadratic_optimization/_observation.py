from typing import Callable
from probnum.type import FloatArgType

import numpy as np
from probnum import utils as _utils


class QuadOptObservation:
    """
    Make an observation of a quadratic optimization problem.

    Parameters
    ----------
    observe :
        Callable returning an observation for a given action.
    """

    def __init__(self, observe: Callable[[FloatArgType], FloatArgType]):
        self._observe = observe

    def __call__(self, action: FloatArgType) -> np.float_:
        """
        Return an observation about the problem.

        Parameters
        ----------
        action :
            Action probing the problem.
        """
        observation = self._observe(action)
        try:
            return _utils.as_numpy_scalar(observation, dtype=np.floating)
        except TypeError as exc:
            raise TypeError(
                "The given argument `p` can not be cast to a `np.floating` object."
            ) from exc


class FunctionEvaluation(QuadOptObservation):
    """
    Observe a (noisy) function evaluation of the quadratic objective.
    """

    def __init__(self, fun: Callable[[FloatArgType], FloatArgType]):
        self._fun = fun
        super().__init__(observe=self._eval_fun)

    def _eval_fun(self, action: FloatArgType) -> FloatArgType:
        """
        Evaluate the objective function.

        Parameters
        ----------
        action :
            Input to the objective function.
        """
        return self._fun(action)
