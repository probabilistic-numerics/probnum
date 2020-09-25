from typing import Optional, Union, Callable, Dict, Tuple, Iterable
from probnum.type import FloatArgType, IntArgType, RandomStateType

import numpy as np
import probnum as pn
from ._policy import QuadOptPolicy, DeterministicPolicy, StochasticPolicy
from ._observation import QuadOptObservation, FunctionEvaluation
from ._belief_update import QuadOptBeliefUpdate, GaussianBeliefUpdate
from ._stopping_criterion import (
    QuadOptStoppingCriterion,
    ParameterUncertainty,
    MaximumIterations,
)


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
            policy=DeterministicPolicy(),
            observation=FunctionEvaluation(fun=fun),
            belief_update=GaussianBeliefUpdate(noise_cov=np.zeros(3)),
            stopping_criteria=[
                ParameterUncertainty(abstol=tol, reltol=tol),
                MaximumIterations(maxiter=maxiter),
            ],
        )
    elif method == "noise":
        # Noisy 1D quadratic optimization
        probquadopt = ProbabilisticQuadraticOptimizer(
            fun_params_prior=fun_params0,
            policy=StochasticPolicy(),
            observation=FunctionEvaluation(fun=fun),
            belief_update=GaussianBeliefUpdate(noise_cov=noise_cov),
            stopping_criteria=[
                ParameterUncertainty(abstol=tol, reltol=tol),
                MaximumIterations(maxiter=maxiter),
            ],
        )
    else:
        raise ValueError(f'Unknown PN method variant "{method}".')

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

    """

    def __init__(
        self,
        fun_params_prior: pn.RandomVariable,
        policy: QuadOptPolicy,
        observation: QuadOptObservation,
        belief_update: QuadOptBeliefUpdate,
        stopping_criteria: Iterable[QuadOptStoppingCriterion],
    ):
        self.fun_params_prior = fun_params_prior
        self.fun_params = self.fun_params_prior
        self.policy = policy
        self.make_observation = observation
        # TODO naming of observation
        self.belief_update = belief_update
        self.stopping_criteria = stopping_criteria
        self.iteration = 0

    def has_converged(
        self, fun: Callable[[FloatArgType], FloatArgType]
    ) -> Tuple[bool, Union[str, None]]:
        """
        Check whether the optimizer has converged.
        """
        for stopping_criterion in self.stopping_criteria:
            _has_converged, convergence_criterion = stopping_criterion(
                fun, self.fun_params, self.iteration
            )
            if _has_converged:
                return _has_converged, convergence_criterion

    def optim_iterator(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
    ) -> Tuple[float, float, pn.RandomVariable]:
        """
        Generator implementing the optimization iteration.

        This function allows stepping through the optimization
        process one step at a time.
        """
        # Compute action via policy
        action = self.policy(fun)

        # Make an observation
        observation = self.make_observation(action)

        # Belief update
        self.fun_params = self.belief_update(self.fun_params, action, observation)

        yield action, observation, self.fun_params

    def optimize(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        callback: Callable[[float, float, pn.RandomVariable], None],
    ) -> Tuple[float, pn.RandomVariable, pn.RandomVariable, Dict]:
        """
        Optimize the quadratic objective function.

        Parameters
        ----------
        fun :
            Quadratic objective function to optimize.
        callback :
            Callback function returning intermediate quantities of the
            optimization loop. Note that depending on the function
            supplied, this can slow down the solver considerably.
        """
        # Setup
        _has_converged = False

        while True:
            # Evaluate stopping criteria
            _has_converged, conv_crit = self.has_converged(fun=fun)
            if _has_converged:
                break

            # Perform one iteration of the optimizer
            action, observation, _ = next(self.optim_iterator(fun=fun))

            # Callback function
            callback(action, observation, self.fun_params)

            self.iteration += 1

        # Belief over optimal function value and optimum
        x_opt, fun_opt = self.belief_optimum()

        # Information on convergence
        info = {"iter": self.iteration, "conv_crit": conv_crit}

        return x_opt, fun_opt, self.fun_params, info

    def belief_optimum(self) -> Tuple[float, pn.RandomVariable]:
        """
        Compute the belief over the optimum and optimal function value.
        """
        x_opt = -self.fun_params.mean[1] / self.fun_params.mean[0]
        fun_opt = np.array([0.5 * x_opt ** 2, x_opt, 1]).T @ self.fun_params
        return x_opt, fun_opt
