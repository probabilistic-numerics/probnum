import collections.abc
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

import probnum as pn
import probnum.utils as _utils
from probnum import linops, randvars
from probnum.typing import FloatArgType, IntArgType

from .belief_updates import gaussian_belief_update
from .observation_operators import function_evaluation
from .policies import explore_exploit_policy, stochastic_policy
from .stopping_criteria import maximum_iterations, parameter_uncertainty

# Type aliases for quadratic optimization
QuadOptPolicyType = Callable[
    [
        Callable[[FloatArgType], FloatArgType],
        randvars.RandomVariable,
    ],
    FloatArgType,
]
QuadOptObservationOperatorType = Callable[
    [Callable[[FloatArgType], FloatArgType], FloatArgType], FloatArgType
]
QuadOptBeliefUpdateType = Callable[
    [
        randvars.RandomVariable,
        FloatArgType,
        FloatArgType,
    ],
    randvars.RandomVariable,
]
QuadOptStoppingCriterionType = Callable[
    [Callable[[FloatArgType], FloatArgType], randvars.RandomVariable, IntArgType],
    Tuple[bool, Union[str, None]],
]


class ProbabilisticQuadraticOptimizer:
    """Probabilistic Quadratic Optimization in 1D.

    PN method solving unconstrained one-dimensional (noisy) quadratic
    optimization problems only needing access to function evaluations.

    Parameters
    ----------
    fun_params_prior :
        Prior belief over the parameters of the latent quadratic function.
    policy :
        Callable returning a new action to probe the problem.
    observation_operator :
        Callable implementing the observation process of the problem.
    belief_update :
        Belief update function updating the belief over the parameters of the quadratic given
        an action and observation of the problem.
    stopping_criteria :
        Stopping criteria to determine when to stop the optimization.

    See Also
    --------
    probsolve_qp : Solve 1D (noisy) quadratic optimization problems.

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> from probnum import randvars
    >>> from quadopt_example.policies import stochastic_policy
    >>> from quadopt_example.observation_operators import function_evaluation
    >>> from quadopt_example.belief_updates import gaussian_belief_update
    >>> from quadopt_example.stopping_criteria import maximum_iterations
    >>>
    >>>
    >>> # Custom stopping criterion based on residual
    >>> def residual(
    >>>     fun,
    >>>     fun_params0,
    >>>     current_iter,
    >>>     abstol=10 ** -6,
    >>>     reltol=10 ** -6,
    >>> ):
    >>>     a, b, c = fun_params0.mean
    >>>     resid = np.abs(fun(1.0) - (0.5 * a + b + c))
    >>>     if resid < abstol:
    >>>         return True, "residual_abstol"
    >>>     elif resid < np.abs(fun(1.0)) * reltol:
    >>>         return True, "residual_reltol"
    >>>     else:
    >>>         return False, None
    >>>
    >>> # Compose custom PN method
    >>> quadopt = ProbabilisticQuadraticOptimizer(
    >>>     fun_params_prior=randvars.Normal(np.zeros(3), np.eye(3)),
    >>>     policy=stochastic_policy,
    >>>     observation_operator=function_evaluation,
    >>>     belief_update=partial(gaussian_belief_update, noise_cov=np.zeros(3)),
    >>>     stopping_criteria=[residual, partial(maximum_iterations, maxiter=10)],
    >>> )
    >>> # Objective function
    >>> f = lambda x: 2.0 * x ** 2 - 0.75 * x + 0.2
    >>>
    >>> quadopt.optimize(f)
    (0.2000000000000014,
     <() Normal with dtype=float64>,
     <(3,) Normal with dtype=float64>,
     {'iter': 3, 'conv_crit': 'residual_abstol'})
    """

    def __init__(
        self,
        fun_params_prior: randvars.RandomVariable,
        policy: QuadOptPolicyType,
        observation_operator: QuadOptObservationOperatorType,
        belief_update: QuadOptBeliefUpdateType,
        stopping_criteria: Union[
            QuadOptStoppingCriterionType, Iterable[QuadOptStoppingCriterionType]
        ],
    ):
        # Optimizer components
        self.fun_params = fun_params_prior
        self.policy = policy
        self.observation_operator = observation_operator
        self.belief_update = belief_update

        if not isinstance(stopping_criteria, collections.abc.Iterable):
            self.stopping_criteria = [stopping_criteria]
        else:
            self.stopping_criteria = stopping_criteria

    def has_converged(
        self, fun: Callable[[FloatArgType], FloatArgType], iteration: IntArgType
    ) -> Tuple[bool, Union[str, None]]:
        """Check whether the optimizer has converged.

        Parameters
        ----------
        fun :
            Quadratic objective function to optimize.
        iteration :
            Number of iterations of the solver performed up to this point.
        """
        for stopping_criterion in self.stopping_criteria:
            _has_converged, convergence_criterion = stopping_criterion(
                fun, self.fun_params, iteration
            )
            if _has_converged:
                return True, convergence_criterion
        return False, None

    def optim_iterator(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
    ) -> Tuple[float, float, randvars.RandomVariable]:
        """Generator implementing the optimization iteration.

        This function allows stepping through the optimization
        process one step at a time.

        Parameters
        ----------
        fun :
            Quadratic objective function to optimize.

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
            action = self.policy(fun, self.fun_params)

            # Make an observation
            observation = self.observation_operator(fun, action)

            # Belief update
            self.fun_params = self.belief_update(self.fun_params, action, observation)

            yield action, observation, self.fun_params

    def optimize(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        callback: Optional[
            Callable[[float, float, randvars.RandomVariable], None]
        ] = None,
    ) -> Tuple[float, randvars.RandomVariable, randvars.RandomVariable, Dict]:
        """Optimize the quadratic objective function.

        Parameters
        ----------
        fun :
            Quadratic objective function to optimize.
        callback :
            Callback function returning intermediate quantities of the
            optimization loop. Note that depending on the function
            supplied, this can slow down the solver considerably.

        Returns
        -------
        x_opt :
            Estimated minimum of ``fun``.
        fun_opt :
            Belief over the optimal value of the objective function.
        fun_params :
            Belief over the parameters of the objective function.
        info :
            Additional information about the optimization, e.g. convergence.
        """
        # Setup
        _has_converged = False
        iteration = 0
        optimization_iterator = self.optim_iterator(fun=fun)

        # Evaluate stopping criteria
        _has_converged, conv_crit = self.has_converged(fun=fun, iteration=iteration)

        while not _has_converged:

            # Perform one iteration of the optimizer
            action, observation, _ = next(optimization_iterator)

            # Callback function
            if callback is not None:
                callback(action, observation, self.fun_params)

            iteration += 1

            # Evaluate stopping criteria
            _has_converged, conv_crit = self.has_converged(fun=fun, iteration=iteration)

        # Belief over optimal function value and optimum
        x_opt, fun_opt = self.belief_optimum()

        # Information (e.g. on convergence)
        info = {"iter": iteration, "conv_crit": conv_crit}

        return x_opt, fun_opt, self.fun_params, info

    def belief_optimum(self) -> Tuple[float, randvars.RandomVariable]:
        """Compute the belief over the optimum and optimal function value.

        Returns
        -------
        x_opt :
            Estimated minimum of ``fun``.
        fun_opt :
            Belief over the optimal value of the objective function.
        """
        x_opt = -self.fun_params.mean[1] / self.fun_params.mean[0]
        fun_opt = np.array([0.5 * x_opt ** 2, x_opt, 1]).T @ self.fun_params
        return x_opt, fun_opt
