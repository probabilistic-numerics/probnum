"""Probabilistic numerical methods for solving integrals."""

from typing import Callable, Optional, Tuple, Union

import numpy as np

from probnum.quad.solvers.policies import Policy, RandomPolicy
from probnum.quad.solvers.stopping_criteria import (
    BQStoppingCriterion,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)
from probnum.randprocs.kernels import ExpQuad, Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike, IntLike

from .._integration_measures import IntegrationMeasure, LebesgueMeasure
from ..kernel_embeddings import KernelEmbedding
from .belief_updates import BQBeliefUpdate, BQStandardBeliefUpdate
from .bq_state import BQState


class BayesianQuadrature:
    r"""A base class for Bayesian quadrature.

    Bayesian quadrature solves integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x).

    Parameters
    ----------
    kernel :
        The kernel used for the GP model.
    measure :
        The integration measure.
    policy :
        The policy choosing nodes at which to evaluate the integrand.
    belief_update :
        The inference method.
    stopping_criterion :
        The criterion that determines convergence.
    """
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        kernel: Kernel,
        measure: IntegrationMeasure,
        policy: Policy,
        belief_update: BQBeliefUpdate,
        stopping_criterion: BQStoppingCriterion,
    ) -> None:
        self.kernel = kernel
        self.measure = measure
        self.policy = policy
        self.belief_update = belief_update
        self.stopping_criterion = stopping_criterion

    @classmethod
    def from_problem(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        measure: Optional[IntegrationMeasure] = None,
        domain: Optional[
            Union[Tuple[FloatLike, FloatLike], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        policy: str = "bmc",
        max_evals: Optional[IntLike] = None,
        var_tol: Optional[FloatLike] = None,
        rel_tol: Optional[FloatLike] = None,
        batch_size: IntLike = 1,
        rng: np.random.Generator = None,
    ) -> "BayesianQuadrature":

        r"""Alternative way to initialize ``Bayesian_Quadrature``

        Parameters
        ----------
        input_dim :
            Input dimension.
        kernel :
            The kernel used for the GP model.
        measure :
            The integration measure.
        domain :
            The integration bounds.
        policy :
            The policy choosing nodes at which to evaluate the integrand.
        max_evals :
            Maximum number of evaluations as stopping criterion.
        var_tol :
            Variance tolerance as stopping criterion.
        rel_tol :
            Relative tolerance as stopping criterion.
        batch_size :
            Batch size used in node acquisition.
        rng :
            The random number generator.
        """
        # Set up integration measure
        if measure is None:
            measure = LebesgueMeasure(domain=domain, input_dim=input_dim)

        # Select policy and belief update
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)
        if policy == "fixed":
            policy = Policy(batch_size=batch_size)
            belief_update = BQStandardBeliefUpdate()
        elif policy == "bmc":
            policy = RandomPolicy(measure.sample, batch_size=batch_size, rng=rng)
            belief_update = BQStandardBeliefUpdate()

            if rng is None:
                errormsg = (
                    "Policy 'bmc' relies on random sampling, "
                    "thus requires a random number generator ('rng')."
                )
                raise ValueError(errormsg)

        else:
            raise NotImplementedError(
                "Policies other than random sampling are not available at the moment."
            )

        # Set stopping criteria
        # If multiple stopping criteria are given, BQ stops once the first criterion is fulfilled.
        def _stopcrit_or(sc1, sc2):
            if sc1 is None:
                return sc2
            return sc1 | sc2

        _stopping_criterion = None

        if max_evals is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, MaxNevals(max_evals)
            )
        if var_tol is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, IntegralVarianceTolerance(var_tol)
            )
        if rel_tol is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, RelativeMeanChange(rel_tol)
            )

        # If no stopping criteria are given, use some default values (these are arbitrary values)
        if _stopping_criterion is None:
            _stopping_criterion = IntegralVarianceTolerance(var_tol=1e-6) | MaxNevals(
                max_nevals=input_dim * 25
            )

        return cls(
            kernel=kernel,
            measure=measure,
            policy=policy,
            belief_update=belief_update,
            stopping_criterion=_stopping_criterion,
        )

    def has_converged(self, bq_state: BQState) -> bool:
        """Checks if the BQ method has converged.

        Parameters
        ----------
        bq_state:
            State of the Bayesian quadrature methods. Contains all necessary information about the
            problem and the computation.

        Returns
        -------
        has_converged:
            Whether or not the solver has converged.
        """

        _has_converged = self.stopping_criterion(bq_state)
        if _has_converged:
            bq_state.info.has_converged = True
            return True
        return False

    def bq_iterator(
        self,
        fun: Optional[Callable] = None,
        nodes: Optional[np.ndarray] = None,
        fun_evals: Optional[np.ndarray] = None,
        integral_belief: Optional[Normal] = None,
        bq_state: Optional[BQState] = None,
    ) -> Tuple[Normal, np.ndarray, np.ndarray, BQState]:
        """Generator that implements the iteration of the BQ method.

        This function exposes the state of the BQ method one step at a time while running the loop.

        Parameters
        ----------
        fun :
            Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
            ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
        nodes :
            *shape=(n_eval, input_dim)* -- Optional nodes at which function evaluations
            are available as ``fun_evals`` from start.
        fun_evals :
            *shape=(n_eval,)* -- Optional function evaluations at ``nodes`` available
            from the start.
        integral_belief:
            Current belief about the integral.
        bq_state:
            State of the Bayesian quadrature methods. Contains all necessary information about the
            problem and the computation.

        Returns
        -------
        integral_belief:
            Updated belief about the integral.
        new_nodes:
            *shape=(n_new_eval, input_dim)* -- The new location(s) at which
            ``new_fun_evals`` are available found during the iteration.
        new_fun_evals:
            *shape=(n_new_eval,)* -- The function evaluations at the new locations
            ``new_nodes``.
        bq_state:
            Updated state of the Bayesian quadrature methods.
        """
        # pylint: disable=missing-yield-doc

        # Setup
        if bq_state is None:
            if integral_belief is None:
                # The following is valid only when the prior is zero-mean.
                integral_belief = Normal(
                    0.0, KernelEmbedding(self.kernel, self.measure).kernel_variance()
                )

            bq_state = BQState(
                measure=self.measure,
                kernel=self.kernel,
                integral_belief=integral_belief,
                batch_size=self.policy.batch_size,
            )

        integral_belief = bq_state.integral_belief

        if nodes is not None:
            if fun_evals is None:
                fun_evals = fun(nodes)

            integral_belief, bq_state = self.belief_update(
                bq_state=bq_state,
                new_nodes=nodes,
                new_fun_evals=fun_evals,
            )

            # make sure info get the number of initial nodes
            bq_state.info.nevals = fun_evals.size

        # Evaluate stopping criteria for the initial belief
        _has_converged = self.has_converged(bq_state=bq_state)

        yield integral_belief, None, None, bq_state

        while True:
            # Have we already converged?
            if _has_converged:
                break

            # Select new nodes via policy
            new_nodes = self.policy(bq_state=bq_state)

            # Evaluate the integrand at new nodes
            new_fun_evals = fun(new_nodes)

            integral_belief, bq_state = self.belief_update(
                bq_state=bq_state,
                new_nodes=new_nodes,
                new_fun_evals=new_fun_evals,
            )

            bq_state.info.update_iteration(bq_state.batch_size)

            # Evaluate stopping criteria
            _has_converged = self.has_converged(bq_state=bq_state)

            yield integral_belief, new_nodes, new_fun_evals, bq_state

    def integrate(
        self,
        fun: Optional[Callable] = None,
        nodes: Optional[np.ndarray] = None,
        fun_evals: Optional[np.ndarray] = None,
    ) -> Tuple[Normal, BQState]:
        """Integrate the function ``fun``.

        ``fun`` may be analytically given, or numerically in terms of ``fun_evals`` at fixed nodes.
        This function calls the generator ``bq_iterator`` until the first stopping criterion is met.

        Parameters
        ----------
        fun :
            Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
            ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
        nodes :
            *shape=(n_eval, input_dim)* -- Optional nodes at which function evaluations
            are available as ``fun_evals`` from start.
        fun_evals :
            *shape=(n_eval,)* -- Optional function evaluations at ``nodes`` available
            from the start.

        Returns
        -------
        integral_belief:
            Posterior belief about the integral.
        bq_state:
            Final state of the Bayesian quadrature method.
        """
        if fun is None and fun_evals is None:
            raise ValueError("You need to provide a function to be integrated!")

        bq_state = None
        integral_belief = None

        for (integral_belief, _, _, bq_state) in self.bq_iterator(
            fun, nodes, fun_evals
        ):
            pass

        return integral_belief, bq_state
