"""Contains the kernel embeddings, i.e., integrals over kernels."""

from typing import Callable, Tuple

import numpy as np

from probnum.quad.integration_measures import (
    GaussianMeasure,
    IntegrationMeasure,
    LebesgueMeasure,
)
from probnum.randprocs.kernels import ExpQuad, Kernel, Matern, ProductMatern

from ._expquad_gauss import _kernel_mean_expquad_gauss, _kernel_variance_expquad_gauss
from ._expquad_lebesgue import (
    _kernel_mean_expquad_lebesgue,
    _kernel_variance_expquad_lebesgue,
)
from ._matern_lebesgue import (
    _kernel_mean_matern_lebesgue,
    _kernel_variance_matern_lebesgue,
)


class KernelEmbedding:
    """Integrals over kernels against integration measures.

    The available kernel embeddings are:

    ============= ===============
    ExpQuad       LebesgueMeasure
    ExpQuad       GaussianMeasure
    Matern (1d)   LebesgueMeasure
    ProductMatern LebesgueMeasure
    ============= ===============

    Parameters
    ----------
    kernel:
        Instance of a kernel.
    measure:
        Instance of an integration measure.

    Raises
    ------
    ValueError
        If the input dimension of the kernel does not match the input dimension of the
        measure.
    """

    def __init__(self, kernel: Kernel, measure: IntegrationMeasure) -> None:
        self.kernel = kernel
        self.measure = measure

        if self.kernel.input_shape != (self.measure.input_dim,):
            raise ValueError(
                "Input dimensions of kernel and measure need to be the same."
            )

        (self.input_dim,) = self.kernel.input_shape

        # retrieve the functions for the provided combination of kernel and measure
        self._kmean, self._kvar = self._get_kernel_embedding(
            kernel=self.kernel, measure=self.measure
        )

    def kernel_mean(self, x: np.ndarray) -> np.ndarray:
        """Kernel mean w.r.t. its first argument against the integration measure.

        Parameters
        ----------
        x :
            *shape=(n_eval, input_dim)* -- n_eval locations where to evaluate the
            kernel mean.

        Returns
        -------
        kernel_mean :
            *shape=(n_eval,)* -- The kernel integrated w.r.t. its first argument,
            evaluated at locations ``x``.
        """
        return self._kmean(x=x, kernel=self.kernel, measure=self.measure)

    def kernel_variance(self) -> float:
        """Kernel integrated in both arguments against the integration measure.

        Returns
        -------
        kernel_variance :
            The kernel integrated w.r.t. both arguments.
        """
        return self._kvar(kernel=self.kernel, measure=self.measure)

    @staticmethod
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
        kernel_mean :
            The kernel mean function.
        kernel_variance :
            The kernel variance function.

        Raises
        ------
        NotImplementedError
            If the given kernel is unknown.
        NotImplementedError
            If the kernel embedding of the kernel-measure pair is unknown.
        """

        # Exponentiated quadratic kernel
        if isinstance(kernel, ExpQuad):
            if isinstance(measure, GaussianMeasure):
                return _kernel_mean_expquad_gauss, _kernel_variance_expquad_gauss
            if isinstance(measure, LebesgueMeasure):
                return _kernel_mean_expquad_lebesgue, _kernel_variance_expquad_lebesgue

        # Matern
        if isinstance(kernel, (Matern, ProductMatern)):
            if isinstance(measure, LebesgueMeasure):
                return _kernel_mean_matern_lebesgue, _kernel_variance_matern_lebesgue

        # other kernels
        raise NotImplementedError(
            f"The combination of kernel ({type(kernel)}) and measure ({type(measure)}) "
            f"is not available as kernel embedding."
        )
