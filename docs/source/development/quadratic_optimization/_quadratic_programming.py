from typing import Optional, Union, Callable, Dict, Tuple, Iterable
from probnum.type import FloatArgType, IntArgType, RandomStateType

import numpy as np
import probnum as pn
from ._stopping_criterion import QuadOptStoppingCriterion
from ._policy import QuadOptPolicy
from ._observation import QuadOptObservation
from ._belief_update import QuadOptBeliefUpdate


def probsolve_qp(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: Optional[Union[np.ndarray, pn.RandomVariable]] = None,
    method: Optional[str] = None,
    tol: FloatArgType = 10 ** -6,
    maxiter: IntArgType = 10 ** 5,
    callback: Optional[
        Callable[[FloatArgType, FloatArgType, pn.RandomVariable], None]
    ] = None,
    random_state: Optional[RandomStateType] = None,
) -> Tuple[float, pn.RandomVariable, pn.RandomVariable, Dict]:
    """
    Probabilistic 1D Quadratic Optimization.

    PN method solving unconstrained one-dimensional (noisy) quadratic
    optimization problems only needing access to function evaluations.

    Parameters
    ----------
    fun :
        Quadratic objective function to optimize.
    fun_params0 :
        *(shape=(3, ) or (3, 1))* -- Prior on the parameters of the
        objective function or initial guess for the parameters.
    method :
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
        Estimated minimum of ``fun``.
    fun_opt :
        Belief over the optimal value of the objective function.
    fun_params :
        Belief over the parameters of the objective function.
    info :
        Additional information about the optimization, e.g. convergence.
    """

    # Choose a variant of the PN method
    if method is None:
        # Infer PN variant to use based on the problem
        if fun(1.0) != fun(1.0):
            method = "noise"
        else:
            method = "exact"

    if method == "exact":
        # Exact 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=deterministic_policy,
            stopping_criteria=[
                partial(residual_stopping_criterion, tol=tol),
                partial(max_iterations, maxiter=maxiter),
            ],
        )
    elif method == "noise":
        # Noisy 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=thompson_sampling,
            stopping_criteria=[
                partial(probabilistic_stopping_criterion, tol=tol),
                partial(max_iterations, maxiter=maxiter),
            ],
        )

    # Run optimization iteration
    x_opt0, fun_opt0, fun_params0, info = probquadopt.optimize(
        fun=fun, callback=callback
    )

    # Return output with information (e.g. on convergence)
    return x_opt0, fun_opt0, fun_params0, info


class ProbabilisticQuadraticOptimizer:
    """
    Probabilistic Quadratic Optimization in 1D



    Parameters
    ----------
    fun_params_prior :

    policy :

    observe :

    stopping_criteria :


    """

    def __init__(
        self,
        fun_params_prior: pn.RandomVariable,
        policy: QuadOptPolicy,
        observation: QuadOptObservation,
        belief_update: QuadOptBeliefUpdate,
        stopping_criteria: Iterable[QuadOptStoppingCriterion],
    ):
        raise NotImplementedError

    def optimize(
        self,
        fun: Callable,
        callback: Callable,
    ) -> Tuple[float, pn.RandomVariable, pn.RandomVariable, Dict]:
        raise NotImplementedError
