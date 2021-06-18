"""Probabilistic numerical methods for solving integrals."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from probnum.kernels import ExpQuad, Kernel
from probnum.randvars import Normal
from probnum.type import FloatArgType, IntArgType

from .._integration_measures import IntegrationMeasure, LebesgueMeasure
from ..kernel_embeddings import KernelEmbedding
from ..policies import Policy, RandomPolicy
from ..stop_criteria import (
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
    StoppingCriterion,
)
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
        The policy for acquiring nodes for function evaluations.
    belief_update :
        The inference method.
    stopping_criteria :
        List of criteria that determine convergence.
    """

    def __init__(
        self,
        kernel: Kernel,
        measure: IntegrationMeasure,
        policy: Policy,
        belief_update: BQBeliefUpdate,
        stopping_criteria: List[StoppingCriterion],
    ) -> None:
        self.kernel = kernel
        self.measure = measure
        self.policy = policy
        self.belief_update = belief_update
        self.stopping_criteria = stopping_criteria

    @classmethod
    def from_bayesquad(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        measure: Optional[IntegrationMeasure] = None,
        domain: Optional[
            Union[Tuple[FloatArgType, FloatArgType], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        policy: str = "bmc",
        max_nevals: Optional[IntArgType] = None,
        var_tol: Optional[FloatArgType] = None,
        rel_tol: Optional[FloatArgType] = None,
        batch_size: Optional[IntArgType] = 1,
    ) -> "BayesianQuadrature":

        # Set up integration measure
        if measure is None:
            measure = LebesgueMeasure(domain=domain, input_dim=input_dim)

        # Select policy and belief update
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)
        if policy == "bmc":
            policy = RandomPolicy(measure, batch_size=batch_size)
            belief_update = BQStandardBeliefUpdate()
        else:
            raise NotImplementedError(
                "Policies other than random sampling are not available at the moment."
            )

        # Set stopping criteria
        # If multiple stopping criteria are given, BQ stops once the first criterion is fulfilled.
        _stopping_criteria = []
        if max_nevals is not None:
            _stopping_criteria.append(MaxNevals(max_nevals))
        if var_tol is not None:
            _stopping_criteria.append(IntegralVarianceTolerance(var_tol))
        if rel_tol is not None:
            _stopping_criteria.append(RelativeMeanChange(rel_tol))

        # If no stopping criteria are given, use some default values
        if not _stopping_criteria:
            _stopping_criteria.append(IntegralVarianceTolerance(var_tol=1e-6))
            _stopping_criteria.append(MaxNevals(max_nevals=input_dim * 25))

        return cls(
            kernel=kernel,
            measure=measure,
            policy=policy,
            belief_update=belief_update,
            stopping_criteria=_stopping_criteria,
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

        if bq_state.info.has_converged:
            return True

        for stopping_criterion in self.stopping_criteria:
            _has_converged = stopping_criterion(bq_state.integral_belief, bq_state)
            if _has_converged:
                bq_state.info.has_converged = True
                bq_state.info.stopping_criterion = stopping_criterion.__class__.__name__
                return True
        return False

    def bq_iterator(
        self,
        fun: Optional[Callable] = None,
        nodes: Optional[np.ndarray] = None,
        fun_evals: Optional[np.ndarray] = None,
        integral_belief: Optional[Normal] = None,
        bq_state: Optional[BQState] = None,
    ):
        """Generator that implements the iteration of the BQ method.

        This function exposes the state of the BQ method one step at a time while running the loop.

        Parameters
        ----------
        fun :
            The integrand.
        nodes :
            Optional nodes available from the start.
        fun_evals :
            Optional function evaluations available from the start.
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
            The new location(s) found during the iteration.
        new_fun_evals:
            The function evaluations at the new locations.
        bq_state:
            Updated state of the Bayesian quadrature methods.
        """

        # Setup
        if bq_state is None:
            if integral_belief is None:
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
    ):
        """Integrate the function ``fun``.

        ``fun`` may be analytically given, or numerically in terms of ``fun_evals`` at fixed nodes.
        This function calls the generator ``bq_iterator`` until the first stopping criterion is met.

        Parameters
        ----------
        fun :
            The integrand. Optional when fixed function evaluations are instead passed.
        nodes :
            Optional nodes available from the start.
        fun_evals :
            Optional function evaluations available from the start.

        Returns
        -------
        integral_belief:
            Posterior belief about the integral.
        bq_state:
            Final state of the Bayesian quadrature methods.
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
