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
from .probabilistic_quadratic_optimizer import ProbabilisticQuadraticOptimizer
from .stopping_criteria import maximum_iterations, parameter_uncertainty


def probsolve_qp(
    rng: np.random.Generator,
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: Optional[Union[np.ndarray, randvars.RandomVariable]] = None,
    assume_fun: Optional[str] = None,
    tol: FloatArgType = 10 ** -5,
    maxiter: IntArgType = 10 ** 4,
    noise_cov: Optional[Union[np.ndarray, linops.LinearOperator]] = None,
    callback: Optional[
        Callable[[FloatArgType, FloatArgType, randvars.RandomVariable], None]
    ] = None,
) -> Tuple[float, randvars.RandomVariable, randvars.RandomVariable, Dict]:
    """Probabilistic 1D Quadratic Optimization.

    PN method solving unconstrained one-dimensional (noisy) quadratic
    optimization problems only needing access to function evaluations.

    Parameters
    ----------
    rng :
        Random number generator.
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

    if assume_fun == "exact":
        # Exact 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=partial(stochastic_policy, rng=rng),
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
            policy=partial(explore_exploit_policy, rng=rng),
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
    fun_params0: Union[randvars.RandomVariable, np.ndarray, None]
) -> randvars.RandomVariable:
    """Initialize the prior distribution over the parameters.

    Sets up a prior distribution if no prior or only a point estimate for the parameters
    of the latent quadratic function is given.

    Parameters
    ----------
    fun_params0
        Random variable encoding the prior distribution over the parameters.
    """
    if isinstance(fun_params0, randvars.RandomVariable):
        return fun_params0
    elif isinstance(fun_params0, np.ndarray):
        return randvars.Normal(mean=fun_params0, cov=np.eye(3))
    elif fun_params0 is None:
        return randvars.Normal(mean=np.ones(3), cov=np.eye(3))
    else:
        raise ValueError(
            "Could not initialize a prior distribution from the given prior "
            + f"information '{fun_params0}'."
        )
