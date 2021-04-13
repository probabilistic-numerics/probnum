"""Contains the kernel embeddings, i.e., integrals over kernels."""

from typing import Callable, Tuple

import numpy as np

from probnum.kernels import ExpQuad, Kernel
from probnum.quad._integration_measures import (
    GaussianMeasure,
    IntegrationMeasure,
    LebesgueMeasure,
)
from probnum.quad.kernel_embeddings import (
    _kernel_mean_expquad_gauss,
    _kernel_mean_expquad_lebesgue,
    _kernel_variance_expquad_gauss,
    _kernel_variance_expquad_lebesgue,
)


class KernelEmbedding:
    """Integrals over kernels against integration measures.

    Parameters
    ----------
    kernel:
        Instance of a kernel.
    measure:
        Instance of an integration measure.
    """

    def __init__(self, kernel: Kernel, measure: IntegrationMeasure) -> None:
        self.kernel = kernel
        self.measure = measure

        if self.kernel.input_dim != self.measure.dim:
            raise ValueError(
                "Input dimensions of kernel and measure need to be the same."
            )

        self.dim = self.kernel.input_dim

        # retrieve the functions for the provided combination of kernel and measure
        self._kmean, self._kvar = _get_kernel_embedding(
            kernel=self.kernel, measure=self.measure
        )

    # pylint: disable=invalid-name
    def kernel_mean(self, x: np.ndarray) -> np.ndarray:
        """Kernel mean w.r.t. its first argument against the integration measure.

        Parameters
        ----------
        x :
            *shape=(n_eval, dim)* -- n_eval locations where to evaluate the kernel mean.

        Returns
        -------
        k_mean :
            *shape=(n_eval,)* -- The kernel integrated w.r.t. its first argument, evaluated at locations x.
        """
        return self._kmean(x=x, kernel=self.kernel, measure=self.measure)

    def kernel_variance(self) -> float:
        """Kernel integrated in both arguments against the integration measure.

        Returns
        -------
        k_var :
            The kernel integrated w.r.t. both arguments.
        """
        return self._kvar(kernel=self.kernel, measure=self.measure)


def _get_kernel_embedding(
    kernel: Kernel, measure: IntegrationMeasure
) -> Tuple[Callable, Callable]:
    """Select the right kernel embedding given the kernel and integration measure.

    Parameters
    ----------
    kernel :
        Instance of a kernel.
    measure :
        Instance of an integration measure.

    Returns
    -------
        An instance of _KernelEmbedding.
    """

    # Exponentiated quadratic kernel
    if isinstance(kernel, ExpQuad):
        # pylint: disable=no-else-return
        if isinstance(measure, GaussianMeasure):
            return _kernel_mean_expquad_gauss, _kernel_variance_expquad_gauss
        elif isinstance(measure, LebesgueMeasure):
            return _kernel_mean_expquad_lebesgue, _kernel_variance_expquad_lebesgue
        raise NotImplementedError

    # other kernels
    raise NotImplementedError
