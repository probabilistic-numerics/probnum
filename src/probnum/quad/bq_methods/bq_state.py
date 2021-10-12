"""State of a Bayesian quadrature method."""

from typing import Optional, Tuple

import numpy as np

from probnum.kernels._kernel import Kernel
from probnum.quad._integration_measures import IntegrationMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.stop_criteria import StoppingCriterion
from probnum.randvars import Normal


class BQInfo:
    """Collect information about the BQ loop.

    Parameters
    ----------
    iteration :
        Iteration of the loop.
    nevals :
        Number of evaluations collected.
    has_converged :
        True if the BQ loop fulfils a stopping criterion, otherwise False.
    stopping_criterion:
        The stopping criterion used to determine convergence.
    """

    def __init__(
        self,
        iteration: int = 0,
        nevals: int = 0,
        has_converged: bool = False,
        stopping_criterion: StoppingCriterion = None,
    ):
        self.iteration = iteration
        self.nevals = nevals
        self.has_converged = has_converged
        self.stopping_criterion = stopping_criterion

    def update_iteration(self, batch_size: int) -> None:
        """Update the quantities tracking iteration info.

        Parameters
        ----------
        batch_size:
            Number of points added in each iteration.
        """
        self.iteration += 1
        self.nevals += batch_size


class BQState:
    """Container for the quantities defining the BQ problem and the BQ loop state.

    Parameters
    ----------
    measure :
        The integration measure.
    kernel :
        The kernel used for BQ.
    integral_belief :
        Normal distribution over the value of the integral.
    info:
        Information about the loop status.
    batch_size:
        Size of the batch when acquiring new nodes.
    nodes:
        All locations at which function evaluations are available.
    fun_evals:
        Function evaluations at nodes.
    """

    def __init__(
        self,
        measure: IntegrationMeasure,
        kernel: Kernel,
        integral_belief: Optional[Normal] = None,
        previous_integral_beliefs: Tuple[Normal] = (),
        info: Optional[BQInfo] = None,
        batch_size: int = 1,
        nodes: Optional[np.ndarray] = None,
        fun_evals: Optional[np.ndarray] = None,
        gram: np.ndarray = np.array([[]]),
        kernel_means: np.ndarray = np.array([]),
    ):
        self.measure = measure
        self.kernel = kernel
        self.kernel_embedding = KernelEmbedding(kernel, measure)
        self.integral_belief = integral_belief
        self.previous_integral_beliefs = previous_integral_beliefs
        self.input_dim = measure.input_dim
        self.batch_size = batch_size

        if nodes is None:
            self.nodes = np.empty((0, self.input_dim))
            self.fun_evals = np.array([])
        else:
            self.nodes = nodes
            self.fun_evals = fun_evals

        if info is None:
            info = BQInfo(nevals=self.fun_evals.size)
        self.info = info
        self.gram = gram
        self.kernel_means = kernel_means

    @classmethod
    def from_new_data(
        cls, nodes, fun_evals, integral_belief, prev_state, gram, kernel_means
    ):
        return cls(
            measure=prev_state.measure,
            kernel=prev_state.kernel,
            integral_belief=integral_belief,
            previous_integral_beliefs=prev_state.previous_integral_beliefs
            + (prev_state.integral_belief,),
            info=prev_state.info,
            batch_size=prev_state.batch_size,
            nodes=nodes,
            fun_evals=fun_evals,
            gram=gram,
            kernel_means=kernel_means,
        )
