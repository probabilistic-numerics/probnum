"""State of a Bayesian quadrature method."""

from typing import Callable, Optional

import numpy as np

from probnum.kernels._kernel import Kernel
from probnum.quad._integration_measures import IntegrationMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding


class BQInfo:
    def __init__(
        self,
        iteration: int = 0,
        nevals: int = 0,
        has_converged: bool = False,
        stopping_criterion=None,
    ):
        self.iteration = iteration
        self.nevals = nevals
        self.has_converged = has_converged
        self.stopping_criterion = stopping_criterion


class BQState:
    def __init__(
        self,
        fun: Callable,
        measure: IntegrationMeasure,
        kernel: Kernel,
        integral_belief=None,
        info: BQInfo = None,
        batch_size: Optional[int] = 1,
        nodes: np.ndarray = None,
        fun_evals=None,
    ):
        self.fun = fun
        self.measure = measure
        self.kernel = kernel
        self.kernel_embedding = KernelEmbedding(kernel, measure)
        self.integral_belief = integral_belief
        self.dim = measure.dim
        self.batch_size = batch_size
        if nodes is None:
            self.nodes = np.empty((0, self.dim))
            self.fun_evals = np.array([])
        else:
            self.nodes = nodes
            self.fun_evals = fun_evals
        if info is None:
            info = BQInfo()
        self.info = info

    @classmethod
    def from_new_data(cls, new_nodes, new_fun_evals, prev_state):
        return cls(
            fun=prev_state.fun,
            measure=prev_state.measure,
            kernel=prev_state.kernel,
            integral_belief=prev_state.integral_belief,
            info=prev_state.info,
            nodes=np.concatenate((prev_state.nodes, new_nodes), axis=0),
            fun_evals=np.append(prev_state.fun_evals, new_fun_evals),
        )
