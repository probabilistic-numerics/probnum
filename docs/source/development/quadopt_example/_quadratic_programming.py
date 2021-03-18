import collections.abc
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

import probnum as pn
import probnum.linalg.linops as linops
import probnum.randvars as rvs
import probnum.utils as _utils
from probnum.type import FloatArgType, IntArgType, RandomStateArgType

from .belief_updates import gaussian_belief_update
from .observation_operators import function_evaluation
from .policies import explore_exploit_policy, stochastic_policy
from .stopping_criteria import maximum_iterations, parameter_uncertainty


def probsolve_qp(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: Optional[Union[np.ndarray, pn.RandomVariable]] = None,
    assume_fun: Optional[str] = None,
    tol: FloatArgType = 10 ** -5,
    maxiter: IntArgType = 10 ** 4,
    noise_cov: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
    callback: Optional[
        Callable[[FloatArgType, FloatArgType, pn.RandomVariable], None]
    ] = None,
    random_state: RandomStateArgType = None,
) -> Tuple[float, pn.RandomVariable, pn.RandomVariable, Dict]:
    """Probabilistic 1D Quadratic Optimization.

    PN method solving unconstrained one-dimensional (noisy) quadratic
    optimization problems only needing access to function evaluations.

    Parameters
    ----------
    fun :
        Quadratic objective function to optimize.
    fun_params0 :
        *(shape=(3, ) or (3, 1))* -- Prior on the parameters of the
        objective function or initial guess for the parameters.
    assume_fun :
        Type of probabilistic numerical method to use. The available
        options are

        =====================  =============
         automatic selection   ``None``
         exact observations    ``"exact"``
         noisy observations    ``"noise"``
        =====================  =============

        If ``None`` the type of method is inferred from the problem
        ``fun`` and prior ``fun_params0``.
    tol :
        Convergence tolerance.
    maxiter :
        Maximum number of iterations.
    noise_cov :
        *(shape=(3, 3))* -- Covariance of the additive noise on the parameters
        of the noisy objective function.
    callback :
        Callback function returning intermediate quantities of the
        optimization loop. Note that depending on the function
        supplied, this can slow down the solver considerably.
    random_state :
        Random state of the solver. If None (or ``np.random``), the global
        ``np.random`` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    Returns
    -------
    x_opt :
        Estimated minimum of the objective function.
    fun_opt :
        Belief over the optimal value of the objective function.
    fun_params :
        Belief over the parameters of the objective function.
    info :
        Additional information about the optimization, e.g. convergence.

    Examples
    --------
    >>> f = lambda x: 2.0 * x ** 2 - 0.75 * x + 0.2
    >>> x_opt, fun_opt, fun_params_opt, info = probsolve_qp(f)
    >>> print(info["iter"])
    3
    """

    # Choose a variant of the PN method
    if assume_fun is None:
        # Infer PN variant to use based on the problem
        if noise_cov is not None or fun(1.0) != fun(1.0):
            assume_fun = "noise"
        else:
            assume_fun = "exact"

    # Select appropriate prior based on the problem
    fun_params0 = _choose_prior(fun_params0=fun_params0)

    # Create a local instance of the random number generator if none is provided
    random_state = _utils.as_random_state(random_state)

    if assume_fun == "exact":
        # Exact 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=partial(stochastic_policy, random_state=random_state),
            observation_operator=function_evaluation,
            belief_update=partial(gaussian_belief_update, noise_cov=np.zeros(3)),
            stopping_criteria=[
                partial(parameter_uncertainty, abstol=tol, reltol=tol),
                partial(maximum_iterations, maxiter=maxiter),
            ],
        )
    elif assume_fun == "noise":
        # Noisy 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=partial(explore_exploit_policy, random_state=random_state),
            observation_operator=function_evaluation,
            belief_update=partial(gaussian_belief_update, noise_cov=noise_cov),
            stopping_criteria=[
                partial(parameter_uncertainty, abstol=tol, reltol=tol),
                partial(maximum_iterations, maxiter=maxiter),
            ],
        )
    else:
        raise ValueError(f'Unknown assumption on function evaluations: "{assume_fun}".')

    # Run optimization iteration
    x_opt0, fun_opt0, fun_params0, info = probquadopt.optimize(
        fun=fun, callback=callback
    )

    # Return output with information (e.g. on convergence)
    info["assume_fun"] = assume_fun
    return x_opt0, fun_opt0, fun_params0, info


def _choose_prior(
    fun_params0: Union[pn.RandomVariable, np.ndarray, None]
) -> pn.RandomVariable:
    """Initialize the prior distribution over the parameters.

    Sets up a prior distribution if no prior or only a point estimate for the parameters
    of the latent quadratic function is given.

    Parameters
    ----------
    fun_params0
        Random variable encoding the prior distribution over the parameters.
    """
    if isinstance(fun_params0, pn.RandomVariable):
        return fun_params0
    elif isinstance(fun_params0, np.ndarray):
        return rvs.Normal(mean=fun_params0, cov=np.eye(3))
    elif fun_params0 is None:
        return rvs.Normal(mean=np.ones(3), cov=np.eye(3))
    else:
        raise ValueError(
            "Could not initialize a prior distribution from the given prior "
            + f"information '{fun_params0}'."
        )


# Type aliases for quadratic optimization
QuadOptPolicyType = Callable[
    [
        Callable[[FloatArgType], FloatArgType],
        pn.RandomVariable,
    ],
    FloatArgType,
]
QuadOptObservationOperatorType = Callable[
    [Callable[[FloatArgType], FloatArgType], FloatArgType], FloatArgType
]
QuadOptBeliefUpdateType = Callable[
    [
        pn.RandomVariable,
        FloatArgType,
        FloatArgType,
    ],
    pn.RandomVariable,
]
QuadOptStoppingCriterionType = Callable[
    [Callable[[FloatArgType], FloatArgType], pn.RandomVariable, IntArgType],
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
    >>> import probnum.randvars as rvs
    >>> from quadopt_example.policies import stochastic_policy
    >>> from quadopt_example.observation_operators import function_evaluation
    >>> from quadopt_example.belief_updates import gaussian_belief_update
    >>> from quadopt_example.stopping_criteria import maximum_iterations
    >>>
    >>> np.random.seed(42)
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
    >>>     fun_params_prior=rvs.Normal(np.zeros(3), np.eye(3)),
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
        fun_params_prior: pn.RandomVariable,
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
    ) -> Tuple[float, float, pn.RandomVariable]:
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
        callback: Optional[Callable[[float, float, pn.RandomVariable], None]] = None,
    ) -> Tuple[float, pn.RandomVariable, pn.RandomVariable, Dict]:
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

    def belief_optimum(self) -> Tuple[float, pn.RandomVariable]:
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
