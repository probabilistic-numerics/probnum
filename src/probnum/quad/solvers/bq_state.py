"""State of a Bayesian quadrature method."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from probnum.quad._integration_measures import IntegrationMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal

# pylint: disable=too-few-public-methods,too-many-instance-attributes


class BQState:
    """Container for the quantities defining the BQ problem and the BQ belief.

    Parameters
    ----------
    measure
        The integration measure.
    kernel
        The kernel used for BQ.
    integral_belief
        Normal distribution over the value of the integral.
    nodes
        All locations at which function evaluations are available.
    fun_evals
        Function evaluations at nodes.

    See Also
    --------
    BQIterInfo : Container for quantities concerning the BQ loop iteration.
    """

    def __init__(
        self,
        measure: IntegrationMeasure,
        kernel: Kernel,
        integral_belief: Optional[Normal] = None,
        previous_integral_beliefs: Tuple[Normal] = (),
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

        if nodes is None:
            self.nodes = np.empty((0, self.input_dim))
            self.fun_evals = np.array([])
        else:
            self.nodes = nodes
            self.fun_evals = fun_evals

        self.gram = gram
        self.kernel_means = kernel_means

    @classmethod
    def from_new_data(
        cls,
        nodes: np.ndarray,
        fun_evals: np.ndarray,
        integral_belief: Normal,
        prev_state: "BQState",
        gram: np.ndarray,
        kernel_means: np.ndarray,
    ) -> "BQState":
        r"""Initialize state from updated data.

        Parameters
        ----------
        nodes
            All locations at which function evaluations are available.
        fun_evals
            Function evaluations at nodes.
        integral_belief
            Normal distribution over the value of the integral.
        prev_state
            Previous state of the BQ loop.
        gram
            The Gram matrix of the given nodes.
        kernel_means
            The kernel means at the given nodes.

        Returns
        -------
        bq_state :
            An instance of this class.
        """
        return cls(
            measure=prev_state.measure,
            kernel=prev_state.kernel,
            integral_belief=integral_belief,
            previous_integral_beliefs=prev_state.previous_integral_beliefs
            + (prev_state.integral_belief,),
            nodes=nodes,
            fun_evals=fun_evals,
            gram=gram,
            kernel_means=kernel_means,
        )


@dataclass
class BQIterInfo:
    """Container for quantities concerning the BQ loop iteration.

    Parameters
    ----------
    iteration
        Iteration of the loop.
    nevals
        Number of evaluations collected.
    has_converged
        True if the BQ loop fulfils a stopping criterion, otherwise False.

    See Also
    --------
    BQState : Container for the quantities defining the BQ problem and the BQ belief.
    """

    iteration: int = 0
    nevals: int = 0
    has_converged: bool = False

    @classmethod
    def from_bq_state(cls, bq_state: BQState) -> "BQIterInfo":
        """Create BQIterInfo container from BQState object.

        Parameters
        ----------
        bq_state
            The initial BQ state.

        Returns
        -------
        BQIterInfo
            An instance of this class.
        """
        return cls(iteration=0, nevals=bq_state.fun_evals.size, has_converged=False)

    @classmethod
    def from_iteration(cls, info: "BQIterInfo", dnevals: int) -> "BQIterInfo":
        """Create BQIterInfo container with updated quantities from iteration.

        Parameters
        ----------
        info
            BQIterInfo from previous iteration.
        dnevals
            Number of points added.

        Returns
        -------
        BQIterInfo
            An instance of this class.
        """
        return cls(
            iteration=info.iteration + 1,
            nevals=info.nevals + dnevals,
            has_converged=info.has_converged,
        )

    @classmethod
    def from_stopping_decision(
        cls, info: "BQIterInfo", has_converged: bool
    ) -> "BQIterInfo":
        """Create BQIterInfo container with updated quantities from stopping decision.

        Parameters
        ----------
        info
            BQIterInfo from previous iteration.
        has_converged
            Whether the BQ method has converged

        Returns
        -------
        BQIterInfo
            An instance of this class.
        """
        return cls(
            iteration=info.iteration, nevals=info.nevals, has_converged=has_converged
        )
