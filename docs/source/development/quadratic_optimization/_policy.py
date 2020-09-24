from typing import Callable, Optional
from probnum.type import FloatArgType, RandomStateType

import numpy as np
import probnum as pn
import probnum.random_variables as rvs


class QuadOptPolicy:
    """
    Policy of a PN method solving a quadratic optimization problem.

    Parameters
    ----------
    policy :
        Callable returning an action to query the problem and obtain an observation.
    """

    def __init__(
        self,
        policy: Callable[
            [
                Callable[[FloatArgType], FloatArgType],
                pn.RandomVariable,
                Optional[RandomStateType],
            ],
            float,
        ],
    ):
        self._policy = policy

    def __call__(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        random_state: Optional[RandomStateType] = None,
    ) -> float:
        """
        Return action from policy.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        random_state :
            Random state of the policy. If None (or np.random), the global np.random state
            is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        return self._policy(fun, fun_params0, random_state)


class StochasticPolicy(QuadOptPolicy):
    """
    Policy exploring around the estimate of the minimum based on the certainty about the
    parameters.
    """

    def __init__(self):
        super().__init__(policy=self._stochastic_policy)

    def _stochastic_policy(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        random_state: Optional[RandomStateType] = None,
    ) -> float:
        """
        Policy exploring around the estimate of the minimum based on the certainty about the
        parameters.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        random_state :
            Random state of the policy. If None (or np.random), the global np.random state
            is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        a0, b0, c0 = fun_params0
        return (
            -b0.mean / a0.mean
            + rvs.Normal(
                0, np.trace(fun_params0.cov), random_state=random_state
            ).sample()
        )


class DeterministicPolicy(QuadOptPolicy):
    """
    Policy returning the nonzero integers in sequence.
    """

    def __init__(self):
        self._nonzero_integer_gen = self.__nonzero_integer_gen()
        super().__init__(policy=self._deterministic_policy)

    def __nonzero_integer_gen(self):
        num = 0
        while True:
            yield num
            num += 1

    def _deterministic_policy(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        random_state: Optional[RandomStateType] = None,
    ) -> float:
        """
        Policy returning the nonzero integers in sequence.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        random_state :
            Random state of the policy. If None (or np.random), the global np.random state
            is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        return next(self._nonzero_integer_gen)
