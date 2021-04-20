"""Probabilistic numerical methods for solving integrals."""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.kernels import ExpQuad, Kernel
from probnum.randvars import Normal
from probnum.type import FloatArgType, IntArgType

from .._integration_measures import IntegrationMeasure
from ..kernel_embeddings import KernelEmbedding
from ..policies import sample_from_measure
from .belief_updates._belief_update import BQStandardBeliefUpdate
from .bq_state import BQState
from .stop_criteria._stopping_criterion import (
    IntegralVariance,
    MaxNevals,
    RelativeError,
)


class BayesianQuadrature:
    r"""A base class for Bayesian quadrature.

    Bayesian quadrature solves integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x).

    Parameters
    ----------
    kernel:
        The kernel used for the GP model.
    policy:
        The policy for acquiring nodes for function evaluations.
    """

    def __init__(
        self,
        kernel: Kernel,
        measure: IntegrationMeasure,
        policy: Callable,
        belief_update,
        stopping_criteria,
    ) -> None:
        self.kernel = kernel
        self.measure = measure
        self.policy = policy
        self.belief_update = belief_update
        self.stopping_criteria = stopping_criteria

    @classmethod
    def from_interface(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        measure: IntegrationMeasure = None,
        method: str = "vanilla",
        policy: str = "bmc",
        max_nevals: IntArgType = None,
        var_tol: FloatArgType = None,
        rel_tol: FloatArgType = None,
        batch_size: IntArgType = 1,
    ) -> "BayesianQuadrature":

        # Select policy and belief update
        # TODO: Rewrite this logic
        if method != "vanilla":
            raise NotImplementedError
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)
        if policy == "bmc":
            policy = partial(sample_from_measure, nevals=batch_size, measure=measure)
            belief_update = BQStandardBeliefUpdate()
        else:
            raise NotImplementedError(
                "Policies other than random sampling are not available at the moment."
            )

        # Set stopping criteria
        _stopping_criteria = []
        if max_nevals is not None:
            _stopping_criteria.append(MaxNevals(max_nevals))
        if var_tol is not None:
            _stopping_criteria.append(IntegralVariance(var_tol))
        if rel_tol is not None:
            _stopping_criteria.append(RelativeError(rel_tol))

        # If no stopping criteria are given, use some default values
        if not _stopping_criteria:
            _stopping_criteria.append(IntegralVariance(var_tol=1e-6))
            _stopping_criteria.append(MaxNevals(max_nevals=input_dim * 25))

        return cls(
            kernel=kernel,
            measure=measure,
            policy=policy,
            belief_update=belief_update,
            stopping_criteria=_stopping_criteria,
        )

    def has_converged(self, integral_belief, bq_state):

        if bq_state.info.has_converged:
            return True

        for stopping_criterion in self.stopping_criteria:
            _has_converged = stopping_criterion(integral_belief, bq_state)
            if _has_converged:
                bq_state.info.has_converged = True
                bq_state.info.stopping_criterion = stopping_criterion.__class__.__name__
                return True
        return False

    def bq_iterator(
        self, fun, measure, integral_belief: Optional = None, bq_state: Optional = None
    ):
        # Setup
        if integral_belief is None:
            if bq_state is not None:
                integral_belief = bq_state.integral_belief
            else:
                integral_belief = Normal(
                    0.0, KernelEmbedding(self.kernel, measure).kernel_variance()
                )

        if bq_state is None:
            bq_state = BQState(
                fun=fun,
                measure=measure,
                kernel=self.kernel,
                integral_belief=integral_belief,
            )

        # Evaluate stopping criteria for the initial belief
        _has_converged = self.has_converged(
            integral_belief=integral_belief, bq_state=bq_state
        )

        yield integral_belief, None, None, bq_state

        while True:

            # Select new nodes via policy
            new_nodes = self.policy(
                integral_belief=integral_belief,
                bq_state=bq_state,
            )

            # Evaluate the integrand at new nodes
            new_fun_evals = bq_state.fun(new_nodes)

            # Update BQ state
            bq_state = BQState.from_new_data(
                new_nodes=new_nodes,
                new_fun_evals=new_fun_evals,
                prev_state=bq_state,
            )

            # Update integral belief
            integral_belief, bq_state = self.belief_update(
                fun=fun,
                measure=measure,
                kernel=self.kernel,
                integral_belief=integral_belief,
                new_nodes=new_nodes,
                new_fun_evals=new_fun_evals,
                bq_state=bq_state,
            )

            bq_state.info.iteration += 1
            bq_state.info.nevals += bq_state.batch_size

            # Evaluate stopping criteria
            _has_converged = self.has_converged(
                integral_belief=integral_belief,
                bq_state=bq_state,
            )

            yield integral_belief, new_nodes, new_fun_evals, bq_state

            if _has_converged:
                break

    def integrate(self, fun: Callable, measure: IntegrationMeasure):

        bq_state = None
        for (integral_belief, _, _, bq_state) in self.bq_iterator(fun, measure):
            pass

        return integral_belief, bq_state

    def integrate_OLD(
        self, fun: Callable, measure: IntegrationMeasure, nevals: int
    ) -> Tuple[Normal, Dict]:
        r"""Integrate the function ``fun``.

        Parameters
        ----------
        fun:
            The integrand function :math:`f`.
        measure :
            An integration measure :math:`\mu`.
        nevals :
            Number of function evaluations.

        Returns
        -------
        F :
            The integral of ``fun`` against ``measure``.
        info :
            Information on the performance of the method.
        """

        # Acquisition policy
        nodes = self.policy(nevals, measure)
        fun_evals = fun(nodes)

        # compute integral mean and variance
        # Define kernel embedding
        kernel_embedding = KernelEmbedding(self.kernel, measure)
        gram = self.kernel(nodes, nodes)
        kernel_mean = kernel_embedding.kernel_mean(nodes)
        initial_error = kernel_embedding.kernel_variance()

        weights = self._solve_gram(gram, kernel_mean)

        integral_mean = np.squeeze(weights.T @ fun_evals)
        integral_variance = initial_error - weights.T @ kernel_mean

        integral = Normal(integral_mean, integral_variance)

        # Information on result
        info = {"model_fit_diagnostic": None}

        return integral, info

    # The following functions are here for the minimal version
    # and shall be factored out once BQ is expanded.
    # 1. acquisition policy
    # 2. GP inference

    @staticmethod
    def _solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve the linear system of the form.

        .. math:: Kx=b.

        Parameters
        ----------
        gram :
            *shape=(nevals, nevals)* -- Symmetric pos. def. kernel Gram matrix :math:`K`.
        rhs :
            *shape=(nevals, ...)* -- Right-hand-side :math:`b`, matrix or vector.

        Returns
        -------
        x:
            The solution to the linear system :math:`K x = b`.
        """
        jitter = 1.0e-6
        chol_gram = cho_factor(gram + jitter * np.eye(gram.shape[0]))
        return cho_solve(chol_gram, rhs)
