"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

import dataclasses
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum import ProbabilisticNumericalMethod
from probnum.linalg.solvers import (
    belief_updates,
    beliefs,
    hyperparam_optim,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.linalg.solvers._state import LinearSolverInfo, LinearSolverState
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

# pylint: disable="invalid-name"


class ProbabilisticLinearSolver(
    ProbabilisticNumericalMethod[LinearSystem, beliefs.LinearSystemBelief]
):
    """Compose a custom probabilistic linear solver.

    Class implementing probabilistic linear solvers. Such (iterative) solvers infer
    solutions to problems of the form

    .. math:: Ax=b,

    where :math:`A \\in \\mathbb{R}^{n \\times n}` and :math:`b \\in \\mathbb{R}^{n}`.
    They return a probability measure which quantifies uncertainty in the output arising
    from finite computational resources or stochastic input. This class unifies and
    generalizes probabilistic linear solvers as described in the literature. [1]_ [2]_
    [3]_ [4]_

    Parameters
    ----------
    prior :
        Prior belief about the quantities of interest :math:`(x, A, A^{-1}, b)` of the
        linear system.
    policy :
        Policy defining actions taken by the solver.
    observation_op :
        Observation process defining how information about the linear system is
        obtained.
    stopping_criteria :
        Stopping criteria determining when the solver has converged.
    hyperparameter_optim :
        Hyperparameter optimization method or boolean flag whether to optimize
        hyperparameters or not.

    References
    ----------
    .. [1] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
       Optimization*, 2015, 25, 234-260
    .. [2] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
       *Statistics and Computing*, 2019
    .. [4] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020

    See Also
    --------
    problinsolve : Solve linear systems in a Bayesian framework.
    bayescg : Solve linear systems with prior information on the solution.

    Examples
    --------
    Define a linear system.

    >>> import numpy as np
    >>> import probnum as pn
    >>> from probnum.problems import LinearSystem
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> n = 10
    >>> A = random_spd_matrix(dim=n, random_state=1)
    >>> linsys = LinearSystem.from_matrix(A=A, random_state=1)

    Create a custom probabilistic linear solver from pre-defined components.

    >>> from probnum.linalg.solvers import ProbabilisticLinearSolver
    >>> from probnum.linalg.solvers.beliefs import LinearSystemBelief
    >>> from probnum.linalg.solvers.policies import ConjugateDirections
    >>> from probnum.linalg.solvers.observation_ops import MatVecObservation
    >>> from probnum.linalg.solvers.stop_criteria import MaxIterations, Residual
    >>> # Custom probabilistic iterative solver
    >>> pls = ProbabilisticLinearSolver(
    ... prior=LinearSystemBelief.from_solution(np.zeros_like(linsys.b), problem=linsys),
    ... policy=ConjugateDirections(),
    ... observation_op=MatVecObservation(),
    ... stopping_criteria=[MaxIterations(), Residual()],
    ... )

    Solve the linear system using the custom solver.

    >>> belief, info = pls.solve(linsys)
    >>> np.linalg.norm(linsys.A @ belief.x.mean - linsys.b)
    2.738786219876204e-05
    """

    def __init__(
        self,
        prior: beliefs.LinearSystemBelief,
        policy: policies.Policy,
        observation_op: observation_ops.ObservationOperator,
        optimize_hyperparams: bool = True,
        belief_update: Optional[belief_updates.LinearSolverBeliefUpdate] = None,
        stopping_criteria: Optional[
            List[stop_criteria.StoppingCriterion]
        ] = stop_criteria.MaxIterations(),
    ):
        # pylint: disable="too-many-arguments"
        self.policy = policy
        self.observation_op = observation_op
        if belief_update is not None:
            self.belief_update = belief_update
        else:
            raise NotImplementedError  # TODO
        self.optimize_hyperparams = optimize_hyperparams
        self.stopping_criteria = stopping_criteria
        super().__init__(
            prior=prior,
        )

    # def _init_belief_update(
    #     self,
    #     belief: beliefs.LinearSystemBelief,
    #     observation_op: observation_ops.ObservationOperator,
    #     hyperparameter_optim: Union[hyperparam_optim.HyperparameterOptimization, bool],
    # ) -> Tuple[
    #     belief_updates.LinearSolverBeliefUpdate,
    #     Optional[hyperparam_optim.HyperparameterOptimization],
    # ]:
    #     """Choose a belief update for the given belief and observation operator."""
    #     if isinstance(belief, beliefs.NoisySymmetricNormalLinearSystemBelief):
    #         if hyperparameter_optim is True:
    #             hyperparameter_optim = hyperparam_optim.OptimalNoiseScale()
    #     if isinstance(belief, beliefs.SymmetricNormalLinearSystemBelief):
    #         if isinstance(observation_op, observation_ops.MatVecObservation):
    #             belief_update_type = belief_updates.SymmetricNormalLinearObsBeliefUpdate
    #
    #             return belief_update_type, hyperparameter_optim
    #     elif isinstance(belief, beliefs.WeakMeanCorrespondenceBelief):
    #         if isinstance(observation_op, observation_ops.MatVecObservation):
    #             belief_update_type = belief_updates.WeakMeanCorrLinearObsBeliefUpdate
    #             if hyperparameter_optim is True:
    #                 hyperparameter_optim = hyperparam_optim.OptimalNoiseScale()
    #
    #             return belief_update_type, hyperparameter_optim
    #
    #     raise NotImplementedError

    @classmethod
    def from_problem(
        cls,
        problem: LinearSystem,
        assume_linsys: str = "sympos",
        A0: Optional[MatrixArgType] = None,
        Ainv0: Optional[MatrixArgType] = None,
        x0: MatrixArgType = None,
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
    ) -> "ProbabilisticLinearSolver":
        """Initialize a probabilistic linear solver from a linear system and available
        prior information.

        Constructs an appropriate prior belief about the linear system from the given
        problem and composes an instance of a probabilistic linear solver taking into
        account the given prior information.

        Parameters
        ----------
        problem :
            Linear system to solve.
        assume_linsys :
            Assumptions on the linear system which can influence solver choice and
            behavior. The available options are (combinations of)

            =========================  =========
             generic matrix            ``gen``
             symmetric matrix          ``sym``
             positive definite matrix  ``pos``
             (additive) noise          ``noise``
            =========================  =========

        A0 :
            A square matrix, linear operator or random variable representing the prior
            belief about the linear operator :math:`A`.
        Ainv0 :
            A square matrix, linear operator or random variable representing the prior
            belief about the inverse :math:`H=A^{-1}`.
        x0 :
            Optional. Prior belief for the solution of the linear system. Will be
            ignored if ``Ainv0`` is given.

        Raises
        ------
        ValueError
            If type or size mismatches detected or inputs ``A`` and ``Ainv`` are not
            square.
        """
        # Check matrix assumptions for correctness
        assume_linsys = assume_linsys.lower()
        _assume_A_tmp = assume_linsys
        for allowed_str in ["gen", "sym", "pos", "noise"]:
            _assume_A_tmp = _assume_A_tmp.replace(allowed_str, "")
        if _assume_A_tmp != "":
            raise ValueError(
                "Assumption '{}' contains unrecognized linear operator properties.".format(
                    assume_linsys
                )
            )

        # Choose matrix based view if not clear from arguments
        if (Ainv0 is not None or A0 is not None) and isinstance(x0, rvs.RandomVariable):
            x0 = None

        # Extract information from system and priors
        # System matrix is symmetric
        if isinstance(A0, rvs.RandomVariable):
            if (
                isinstance(A0.cov, linops.SymmetricKronecker)
                and "sym" not in assume_linsys
            ):
                assume_linsys += "sym"
        if isinstance(Ainv0, rvs.RandomVariable):
            if (
                isinstance(Ainv0.cov, linops.SymmetricKronecker)
                and "sym" not in assume_linsys
            ):
                assume_linsys += "sym"
        # System matrix or right hand side is stochastic
        if (
            isinstance(problem.A, rvs.RandomVariable)
            or isinstance(problem.b, rvs.RandomVariable)
            and "noise" not in assume_linsys
        ):
            assume_linsys += "noise"

        # Choose belief class
        belief_class = beliefs.LinearSystemBelief
        if "sym" in assume_linsys and "pos" in assume_linsys:
            if "noise" in assume_linsys:
                belief_class = beliefs.NoisySymmetricNormalLinearSystemBelief
            else:
                belief_class = beliefs.WeakMeanCorrespondenceBelief
        elif "sym" in assume_linsys and "pos" not in assume_linsys:
            belief_class = beliefs.SymmetricNormalLinearSystemBelief

        # Instantiate a prior belief from available prior information
        if x0 is None and A0 is not None and Ainv0 is not None:
            prior = belief_class.from_matrices(A0=A0, Ainv0=Ainv0, problem=problem)
        elif Ainv0 is not None:
            prior = belief_class.from_inverse(Ainv0=Ainv0, problem=problem)
        elif A0 is not None:
            prior = belief_class.from_matrix(A0=A0, problem=problem)
        elif x0 is not None:
            prior = belief_class.from_solution(x0=x0, problem=problem)
        else:
            prior = belief_class.from_scalar(scalar=1.0, problem=problem)

        return cls.from_prior(prior=prior, maxiter=maxiter, atol=atol, rtol=rtol)

    @classmethod
    def from_prior(
        cls,
        prior: beliefs.LinearSystemBelief,
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
    ) -> "ProbabilisticLinearSolver":
        """Initialize a custom probabilistic linear solver from a prior belief.

        Composes and initializes an appropriate instance of the probabilistic linear
        solver based on the prior information given.

        Parameters
        ----------
        prior :
            Prior belief about the quantities of interest of the linear system.
        maxiter :
            Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is
            the dimension of :math:`A`.
        atol :
            Absolute convergence tolerance.
        rtol :
            Relative convergence tolerance.

        Raises
        ------
        ValueError
            If an unknown or incompatible prior belief class is passed.
        """

        observation_op = observation_ops.MatVecObservation()
        stopping_criteria = [stop_criteria.MaxIterations(maxiter=maxiter)]
        if isinstance(
            prior,
            (
                beliefs.SymmetricNormalLinearSystemBelief,
                beliefs.WeakMeanCorrespondenceBelief,
            ),
        ):
            policy = policies.ConjugateDirections()
            stopping_criteria.append(stop_criteria.Residual(atol=atol, rtol=rtol))
        elif isinstance(prior, beliefs.NoisySymmetricNormalLinearSystemBelief):
            policy = policies.ExploreExploit()
            stopping_criteria.append(
                stop_criteria.PosteriorContraction(atol=atol, rtol=rtol)
            )
        elif isinstance(prior, beliefs.LinearSystemBelief):
            policy = policies.ConjugateDirections()
            stopping_criteria.append(stop_criteria.Residual(atol=atol, rtol=rtol))
        else:
            raise ValueError("Unknown or incompatible prior belief class.")

        return cls(
            prior=prior,
            policy=policy,
            observation_op=observation_op,
            stopping_criteria=stopping_criteria,
        )

    def has_converged(
        self,
        problem: LinearSystem,
        belief: beliefs.LinearSystemBelief,
        solver_state: LinearSolverState,
    ) -> Tuple[bool, LinearSolverState]:
        """Check whether the solver has converged.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        has_converged :
            True if the method has converged.
        solver_state :
            Updated state of the solver.
        """
        if solver_state.has_converged:
            return True, solver_state

        # Check stopping criteria
        for stopping_criterion in self.stopping_criteria:
            _has_converged = stopping_criterion(problem, belief, solver_state)
            if _has_converged:
                solver_state.info = LinearSolverInfo(
                    has_converged=True,
                    stopping_criterion=stopping_criterion.__class__.__name__,
                )
                return True, solver_state
        return False, solver_state

    def solve_iterator(
        self,
        problem: LinearSystem,
        belief: Optional[beliefs.LinearSystemBelief] = None,
        solver_state: Optional[LinearSolverState] = None,
    ) -> Generator[
        Tuple[beliefs.LinearSystemBelief, np.ndarray, np.ndarray, LinearSolverState],
        None,
        None,
    ]:
        """Generator implementing the solver iteration.

        This function allows stepping through the solver iteration one step at a time
        and exposes the internal solver state.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief :
            Belief over the quantities of interest :math:`(x, A, A^{-1}, b)` of the
            linear system.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        belief :
            Updated belief about the quantities of interest :math:`(x, A, A^{-1}, b)` of
            the linear system.
        action :
            Action taken by the solver given by its policy.
        observation :
            Observation of the linear system for the corresponding action.
        solver_state :
            Updated state of the linear solver.
        """
        # Setup
        if belief is None:
            if solver_state is not None:
                belief = solver_state.belief
            else:
                belief = self.prior

        if solver_state is None:
            solver_state = LinearSolverState(
                problem=problem,
                prior=self.prior,
                belief=belief,
                # Todo
            )

        solver_state.belief = belief

        # Evaluate stopping criteria for the prior
        _has_converged, solver_state = self.has_converged(
            problem=problem, belief=belief, solver_state=solver_state
        )

        while True:

            # Compute action via policy
            action = self.policy(
                problem=solver_state.problem,
                belief=solver_state.belief,
                solver_state=solver_state,
            )

            # Make an observation of the linear system
            observation = self.observation_op(
                problem=solver_state.problem,
                action=action,
                solver_state=solver_state,
            )

            # Lazily update and cache data-dependent state components
            solver_state = LinearSolverState.from_new_data(
                action=action, observation=observation, prev_state=solver_state
            )

            # Optimize hyperparameters
            if self.optimize_hyperparams:

                hyperparams = belief.hyperparams.optimize(
                    problem=solver_state.problem,
                    actions=solver_state.data.actions,
                    observations=solver_state.data.observations,
                    solver_state=solver_state,
                )
            else:
                hyperparams = belief.hyperparams

            # Update the belief over the quantities of interest
            belief, solver_state = belief.update(
                problem=problem,
                action=action,
                observation=observation,
                hyperparams=hyperparams,
                solver_state=solver_state,
            )

            # Evaluate stopping criteria
            _has_converged, solver_state = self.has_converged(
                problem=problem,
                belief=belief,
                solver_state=solver_state,
            )

            solver_state.info.iteration += 1

            yield belief, action, observation, solver_state

            if _has_converged:
                break

    def solve(
        self,
        problem: LinearSystem,
    ) -> Tuple[beliefs.LinearSystemBelief, LinearSolverState]:
        """Solve the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.

        Returns
        -------
        belief : Posterior belief `(x, A, Ainv, b)` over the solution :math:`x`,
                 the system matrix :math:`A`, its inverse :math:`H=A^{-1}` and the
                 right hand side :math:`b`.
        solver_state : State of the solver at convergence.
        """

        # Solver iteration
        belief = None
        solver_state = None

        for (belief, _, _, solver_state) in self.solve_iterator(problem):
            pass

        return belief, solver_state
