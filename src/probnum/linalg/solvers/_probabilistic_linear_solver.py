"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

from typing import Generator, List, Optional, Tuple

from probnum import linops, randvars
from probnum._probabilistic_numerical_method import ProbabilisticNumericalMethod
from probnum.linalg.solvers import (
    belief_updates,
    beliefs,
    hyperparam_optim,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.linalg.solvers._state import LinearSolverState
from probnum.linalg.solvers.data import LinearSolverAction, LinearSolverObservation
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

# pylint: disable="invalid-name"

__all__ = ["ProbabilisticLinearSolver"]


class ProbabilisticLinearSolver(
    ProbabilisticNumericalMethod[
        LinearSystem, beliefs.LinearSystemBelief
    ]  # pylint: disable="unsubscriptable-object"
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
    hyperparam_optim_method :
        Hyperparameter optimization method.
    belief_update :
        Belief update defining how to update the QoI beliefs given new observations.
        # TODO: hyperparam_optim_method should be an argument of BeliefUpdate()
    stopping_criteria :
        Stopping criteria determining when the solver has converged.


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
    >>> from probnum.linalg.solvers.beliefs import SymmetricNormalLinearSystemBelief
    >>> from probnum.linalg.solvers.policies import ConjugateDirections
    >>> from probnum.linalg.solvers.observation_ops import MatVec
    >>> from probnum.linalg.solvers.stop_criteria import MaxIterations, ResidualNorm
    >>> # Composition of a custom probabilistic linear solver
    >>> pls = ProbabilisticLinearSolver(
    ...     prior=SymmetricNormalLinearSystemBelief.from_solution(np.zeros_like(linsys.b),
    ...                                            problem=linsys),
    ...     policy=ConjugateDirections(),
    ...     observation_op=MatVec(),
    ...     stopping_criteria=[MaxIterations(), ResidualNorm()],
    ... )

    Solve the linear system using the custom solver.

    >>> belief, solver_state = pls.solve(linsys)
    >>> np.linalg.norm(linsys.A @ belief.x.mean - linsys.b)
    2.738786219837142e-05
    """

    def __init__(
        self,
        prior: beliefs.LinearSystemBelief,  # TODO: move to solve method
        observation_op: observation_ops.ObservationOp,
        policy: Optional[policies.Policy] = None,
        hyperparam_optim_method: Optional[
            hyperparam_optim.HyperparameterOptimization
        ] = None,
        belief_update: Optional[belief_updates.LinearSolverBeliefUpdate] = None,
        stopping_criteria: Optional[List[stop_criteria.StoppingCriterion]] = None,
    ):
        # pylint: disable="too-many-arguments"
        self.observation_op = observation_op
        if policy is None:
            self.policy = self._select_default_policy(
                prior=prior, observation_op=observation_op
            )
        else:
            self.policy = policy
        if belief_update is None:
            self.belief_update = self._select_default_belief_update(
                prior=prior, observation_op=observation_op
            )
        else:
            self.belief_update = belief_update
        self.hyperparam_optim_method = hyperparam_optim_method
        if stopping_criteria is None:
            self.stopping_criteria = self._select_default_stopping_criteria(
                prior=prior, observation_op=observation_op
            )
        else:
            self.stopping_criteria = stopping_criteria
        super().__init__(
            prior=prior,
        )

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
        if (Ainv0 is not None or A0 is not None) and isinstance(
            x0, randvars.RandomVariable
        ):
            x0 = None

        # Extract information from system and priors
        # System matrix is symmetric
        if isinstance(A0, randvars.RandomVariable):
            if (
                isinstance(A0.cov, linops.SymmetricKronecker)
                and "sym" not in assume_linsys
            ):
                assume_linsys += "sym"
        if isinstance(Ainv0, randvars.RandomVariable):
            if (
                isinstance(Ainv0.cov, linops.SymmetricKronecker)
                and "sym" not in assume_linsys
            ):
                assume_linsys += "sym"
        # System matrix or right hand side is stochastic
        if (
            isinstance(problem.A, randvars.RandomVariable)
            or isinstance(problem.b, randvars.RandomVariable)
            and "noise" not in assume_linsys
        ):
            assume_linsys += "noise"

        # Select belief class
        belief_class = beliefs.LinearSystemBelief
        observation_op = observation_ops.MatVec()
        if "sym" in assume_linsys and "pos" in assume_linsys:
            if "noise" in assume_linsys:
                observation_op = observation_ops.SampleMatVec()
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

        return cls.from_prior_observation_op(
            prior=prior,
            observation_op=observation_op,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )

    @classmethod
    def from_prior_observation_op(
        cls,
        prior: beliefs.LinearSystemBelief,
        observation_op: observation_ops.ObservationOp = observation_ops.MatVec(),
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
    ) -> "ProbabilisticLinearSolver":
        """Initialize a custom probabilistic linear solver from a prior belief and a
        observation process.

        Composes and initializes an appropriate instance of the probabilistic linear
        solver based on the prior information given and the observation process.

        Parameters
        ----------
        prior :
            Prior belief about the quantities of interest of the linear system.
        observation_op :
            Observation operator defining how information about the linear system is
            obtained.
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

        policy = cls._select_default_policy(prior=prior, observation_op=observation_op)
        belief_update = cls._select_default_belief_update(
            prior=prior, observation_op=observation_op
        )
        stopping_criteria = cls._select_default_stopping_criteria(
            prior=prior,
            observation_op=observation_op,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )
        return cls(
            prior=prior,
            policy=policy,
            observation_op=observation_op,
            belief_update=belief_update,
            stopping_criteria=stopping_criteria,
        )

    @staticmethod
    def _select_default_policy(
        prior: beliefs.LinearSystemBelief,
        observation_op: observation_ops.ObservationOp,
    ) -> policies.Policy:
        """Select a default policy from a prior belief and observation operator.

        Parameters
        ----------
        prior :
            Prior belief about the quantities of interest of the linear system.
        observation_op :
            Observation operator defining how information about the linear system is
            obtained.
        """
        if isinstance(observation_op, observation_ops.MatVec):
            if isinstance(prior, beliefs.WeakMeanCorrespondenceBelief):
                return policies.ConjugateDirections()
            elif isinstance(prior, beliefs.SymmetricNormalLinearSystemBelief):
                return policies.ConjugateDirections()
        elif isinstance(observation_op, observation_ops.SampleMatVec):
            if isinstance(prior, beliefs.NoisySymmetricNormalLinearSystemBelief):
                return policies.ConjugateDirections()
        raise ValueError(
            "No default policy available for this prior and observation "
            "operator combination."
        )

    @staticmethod
    def _select_default_belief_update(
        prior: beliefs.LinearSystemBelief,
        observation_op: observation_ops.ObservationOp,
    ) -> belief_updates.LinearSolverBeliefUpdate:
        """Select a default belief update from a prior belief and observation operator.

        Parameters
        ----------
        prior :
            Prior belief about the quantities of interest of the linear system.
        observation_op :
            Observation operator defining how information about the linear system is
            obtained.
        """
        if isinstance(observation_op, observation_ops.MatVec):
            if isinstance(prior, beliefs.WeakMeanCorrespondenceBelief):
                return belief_updates.WeakMeanCorrLinearObsBeliefUpdate(prior=prior)
            elif isinstance(prior, beliefs.SymmetricNormalLinearSystemBelief):
                return belief_updates.SymmetricNormalLinearObsBeliefUpdate(prior=prior)
        elif isinstance(observation_op, observation_ops.SampleMatVec):
            if isinstance(prior, beliefs.NoisySymmetricNormalLinearSystemBelief):
                return belief_updates.SymmetricNormalLinearObsBeliefUpdate(prior=prior)
        raise ValueError(
            "No default belief update available for this prior and observation "
            "operator combination."
        )

    @staticmethod
    def _select_default_stopping_criteria(
        prior: beliefs.LinearSystemBelief,
        observation_op: observation_ops.ObservationOp,
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
    ) -> List[stop_criteria.StoppingCriterion]:
        """Select default stopping criteria from a prior belief and observation
        operator.

        Parameters
        ----------
        prior :
            Prior belief about the quantities of interest of the linear system.
        observation_op :
            Observation operator defining how information about the linear system is
            obtained.
        maxiter :
            Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is
            the dimension of :math:`A`.
        atol :
            Absolute convergence tolerance.
        rtol :
            Relative convergence tolerance.
        """
        stopping_criteria = [stop_criteria.MaxIterations(maxiter=maxiter)]
        if isinstance(observation_op, observation_ops.MatVec):
            if isinstance(prior, beliefs.WeakMeanCorrespondenceBelief):
                stopping_criteria.append(
                    stop_criteria.ResidualNorm(atol=atol, rtol=rtol)
                )
                return stopping_criteria
            elif isinstance(prior, beliefs.SymmetricNormalLinearSystemBelief):
                stopping_criteria.append(
                    stop_criteria.ResidualNorm(atol=atol, rtol=rtol)
                )
                return stopping_criteria
        elif isinstance(observation_op, observation_ops.SampleMatVec):
            if isinstance(prior, beliefs.NoisySymmetricNormalLinearSystemBelief):
                stopping_criteria.append(
                    stop_criteria.PosteriorContraction(atol=atol, rtol=rtol)
                )
                return stopping_criteria
        raise ValueError(
            "No default stopping criteria available for this prior and observation "
            "operator combination."
        )

    def has_converged(
        self,
        problem: LinearSystem,
        belief: beliefs.LinearSystemBelief,
        solver_state: LinearSolverState,
    ) -> bool:
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
        """
        if solver_state.info.has_converged:
            return True

        # Check stopping criteria
        for stopping_criterion in self.stopping_criteria:
            _has_converged = stopping_criterion(problem, belief, solver_state)
            if _has_converged:
                solver_state.info.has_converged = True
                solver_state.info.stopping_criterion = (
                    stopping_criterion.__class__.__name__
                )
                return True
        return False

    def solve_iterator(
        self,
        problem: LinearSystem,
        prior: Optional[beliefs.LinearSystemBelief] = None,
        # solver_state: Optional[LinearSolverState] = None, # TODO: solver state should not be an argument here
    ) -> Generator[
        Tuple[
            beliefs.LinearSystemBelief,
            LinearSolverAction,
            LinearSolverObservation,
            LinearSolverState,
        ],
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
                cache=self.belief_update.cache_type(
                    problem=problem, prior=self.prior, belief=belief
                ),
            )
        else:
            solver_state.belief = belief

        # Evaluate stopping criteria for the prior
        _has_converged = self.has_converged(
            problem=problem, belief=belief, solver_state=solver_state
        )

        yield belief, None, None, solver_state  # TODO: This does one unnecessary iteration in the loop, move into while loop instead.

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

            # # Optimize hyperparameters
            # if self.hyperparam_optim_method is not None:

            #     hyperparams = self.hyperparam_optim_method(
            #         problem=solver_state.problem,  # TODO: All solver components should only take a SolverState, which always has a problem, belief and data, from which potentially cached properties are generated.
            #         belief=solver_state.belief,
            #         data=solver_state.data,
            #         solver_state=solver_state,
            #     )
            # else:
            #     hyperparams = self.prior.hyperparams
            # TODO this just gets called automatically in .belief_update()

            # Update the belief over the quantities of interest
            belief, solver_state = self.belief_update(
                # problem=problem,
                # belief=belief,
                # action=action,
                # observation=observation,
                # hyperparams=hyperparams,
                solver_state=solver_state,
            )

            solver_state.info.iteration += 1
            # TODO: solver_state.next_step()

            # Evaluate stopping criteria
            _has_converged = self.has_converged(
                problem=solver_state.problem,
                belief=solver_state.belief,
                solver_state=solver_state,
            )

            yield belief, action, observation, solver_state

            if _has_converged:
                break

    def solve(
        self,
        problem: LinearSystem,
        # prior: Optional[LinearSystemBelief] = None, TODO: generate prior via LinearSystemBelief.from_problem(problem)
    ) -> Tuple[beliefs.LinearSystemBelief, LinearSolverState]:
        r"""Solve the linear system.

        Parameters
        ----------
        problem :
            Linear system to solve.

        Returns
        -------
        belief :
            Posterior belief :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})`
            over the solution :math:`x`, the system matrix :math:`A`, its inverse
            :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            State of the solver at convergence.
        """
        belief = None
        solver_state = None

        for (belief, _, _, solver_state) in self.solve_iterator(problem):
            pass

        return belief, solver_state
