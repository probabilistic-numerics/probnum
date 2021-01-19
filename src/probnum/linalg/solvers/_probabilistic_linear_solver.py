"""Probabilistic Linear Solvers.

Iterative probabilistic numerical methods solving linear systems :math:`Ax = b`.
"""

import dataclasses
from typing import Generator, List, Optional, Tuple, Type, Union

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum._probabilistic_numerical_method import (
    PNMethodData,
    PNMethodState,
    ProbabilisticNumericalMethod,
)
from probnum.linalg.solvers import (
    belief_updates,
    beliefs,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

# pylint: disable="invalid-name"


@dataclasses.dataclass
class BeliefUpdateState(PNMethodData):
    r"""Quantities computed for the belief update of a linear solver.

    Parameters
    ----------
    action_obs_innerprods
        Inner product(s) :math:`(S^\top Y)_{ij} = s_i^\top y_j` of actions
        and observations. If a vector, actions and observations are assumed to be
        conjugate, i.e. :math:`s_i^\top y_j =0` for :math:`i \neq j`.
    log_rayleigh_quotients
        Log-Rayleigh quotients :math:`\ln R(A, s_i) = \ln(s_i^\top A s_i)-\ln(s_i^\top
        s_i)`.
    step_sizes
        Step sizes :math:`\alpha_i` of the solver viewed as a quadratic optimizer taking
        steps :math:`x_{i+1} = x_i + \alpha_i s_i`.
    """
    action_obs_innerprods: Optional[List[float]] = None
    log_rayleigh_quotients: Optional[List[float]] = None
    step_sizes: Optional[List[float]] = None


@dataclasses.dataclass
class LinearSolverState(PNMethodState[beliefs.LinearSystemBelief]):
    r"""State of a probabilistic linear solver.

    The solver state contains miscellaneous quantities computed during an iteration
    of a probabilistic linear solver. The solver state is passed between the
    different components of the solver and may be used by them.

    For example the residual :math:`r_i = Ax_i - b` can (depending on the prior) be
    updated more efficiently than in :math:`\mathcal{O}(n^2)` and is therefore part
    of the solver state and passed to the stopping criteria.

    Parameters
    ----------
    belief
        Belief over the quantities of the linear system.
    data
        Collected data about the linear system.
    iteration
        Current iteration :math:`i` of the solver.
    has_converged
        Has the solver converged?
    stopping_criterion
        Stopping criterion which caused termination of the solver.
    residual
        Residual :math:`r_i = Ax_i - b` of the current solution.
    belief_update_state
        State of the belief update containing precomputed quantities for efficiency.

    Examples
    --------

    """
    iteration: int = 0
    residual: Optional[Union[np.ndarray, rvs.RandomVariable]] = None
    has_converged: bool = False
    stopping_criterion: Optional[List[stop_criteria.StoppingCriterion]] = None
    belief_update_state: Optional[BeliefUpdateState] = None


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
    optimize_hyperparams :
        Whether to optimize the hyperparameters of the solver.

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
        belief_update: belief_updates.BeliefUpdate = None,
        stopping_criteria: Optional[
            List[stop_criteria.StoppingCriterion]
        ] = stop_criteria.MaxIterations(),
    ):
        # pylint: disable="too-many-arguments"
        self.policy = policy
        self.observation_op = observation_op
        self.optimize_hyperparams = optimize_hyperparams
        if belief_update is not None:
            self.belief_update = belief_update
        else:
            self.belief_update = self._init_belief_update(
                belief=prior, observation_op=observation_op
            )
        self.stopping_criteria = stopping_criteria
        super().__init__(
            prior=prior,
        )

    def _init_solver_state(self, problem: LinearSystem) -> LinearSolverState:
        """Initialize the solver state.

        Parameters
        ----------
        problem :
            Linear system to solve.
        """
        return LinearSolverState(
            belief=self.prior,
            data=PNMethodData(actions=[], observations=[]),
            iteration=0,
            residual=problem.A @ self.prior.x.mean - problem.b,
            belief_update_state=BeliefUpdateState(
                action_obs_innerprods=[], log_rayleigh_quotients=[], step_sizes=[]
            ),
            has_converged=False,
            stopping_criterion=None,
        )

    def _init_belief_update(
        self,
        belief: beliefs.LinearSystemBelief,
        observation_op: observation_ops.ObservationOperator,
    ) -> Type[belief_updates.BeliefUpdate]:
        """Choose a belief update for the provided belief and observation operator.

        Selects an appropriate belief update for the given belief and observation
        operator.

        Parameters
        ----------
        belief :
            Belief about the linear system.
        observation_op :
        """
        if isinstance(belief, beliefs.SymmetricLinearSystemBelief):
            if isinstance(observation_op, observation_ops.MatVecObservation):
                return belief_updates.SymmetricNormalLinearObsBeliefUpdate
        elif isinstance(belief, beliefs.WeakMeanCorrespondenceBelief):
            return belief_updates.WeakMeanCorrLinearObsBeliefUpdate

        raise NotImplementedError

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
                belief_class = beliefs.NoisyLinearSystemBelief
            else:
                belief_class = beliefs.WeakMeanCorrespondenceBelief
        elif "sym" in assume_linsys and "pos" not in assume_linsys:
            belief_class = beliefs.SymmetricLinearSystemBelief

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
            (beliefs.SymmetricLinearSystemBelief, beliefs.WeakMeanCorrespondenceBelief),
        ):
            policy = policies.ConjugateDirections()
            stopping_criteria.append(stop_criteria.Residual(atol=atol, rtol=rtol))
        elif isinstance(prior, beliefs.NoisyLinearSystemBelief):
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
                solver_state.has_converged = True
                solver_state.stopping_criterion = stopping_criterion.__class__.__name__
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
        if belief is None:
            if solver_state is not None:
                belief = solver_state.belief
            else:
                belief = self.prior

        # Setup
        if solver_state is None:
            solver_state = self._init_solver_state(problem)

            # Evaluate stopping criteria for the prior
            _has_converged, solver_state = self.has_converged(
                problem=problem, belief=belief, solver_state=solver_state
            )

        while True:
            # Compute action via policy
            action, solver_state = self.policy(
                problem=problem, belief=solver_state.belief, solver_state=solver_state
            )

            # Make an observation of the linear system
            observation, solver_state = self.observation_op(
                problem=problem, action=action, solver_state=solver_state
            )

            # TODO precompute quantities for the belief update potentially used in
            #  hyperparameter optimization.
            solver_state.belief_state = self.belief_update.precompute(
                problem=problem,
                actions=solver_state.actions,
                observations=solver_state.observations,
                solver_state=solver_state,
            )

            # Optimize hyperparameters
            if self.optimize_hyperparams:
                try:
                    solver_state = belief.optimize_hyperparams(
                        problem=problem,
                        actions=solver_state.data.actions,
                        observations=solver_state.data.observations,
                        solver_state=solver_state,
                    )
                except NotImplementedError:
                    pass

            # Update the belief about the system matrix, its inverse and/or the solution
            belief, solver_state = self.belief_update(
                problem=problem,
                actions=solver_state.actions,
                observations=solver_state.observations,
                solver_state=solver_state,
            )

            solver_state.iteration += 1

            # Evaluate stopping criteria and update solver state
            _has_converged, solver_state = self.has_converged(
                problem=problem,
                belief=belief,
                solver_state=solver_state,
            )

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
