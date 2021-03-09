"""Contains the kernel embeddings, i.e., integrals over kernels."""

import abc

import numpy as np
import scipy.linalg as slinalg

from ..kernels import ExpQuad, Kernel
from ._integration_measures import GaussianMeasure, IntegrationMeasure, LebesgueMeasure


class _KernelEmbedding(abc.ABC):
    """Abstract class for integrals over kernels.

    Child classes implement integrals for a given combination of kernel and measure.
    The naming scheme for child classes is 'K + kernel + M + measure'

    Parameters
    ----------
    kernel: Kernel
        Instance of a kernel
    measure: IntegrationMeasure
        Instance of an integration measure
    """

    def __init__(self, kernel: Kernel, measure: IntegrationMeasure):
        self.kernel = kernel
        self.measure = measure

    def kernel_mean(self, x: np.ndarray) -> np.ndarray:
        """Kernel mean w.r.t. its first argument against the integration measure.

        Parameters
        ----------
        x: np.ndarray with shape (dim, n_eval)
            n_eval locations where to evaluate the kernel mean.

        Returns
        -------
        k_mean: np.ndarray with shape (n_eval,)
            The kernel integrated w.r.t. its first argument, evaluated at locations x
        """
        raise NotImplementedError

    def kernel_variance(self) -> float:
        """Kernel integrated in both arguments against the integration measure.

        Returns
        -------
        k_var: float
            The kernel integrated w.r.t. both arguments
        """
        raise NotImplementedError


class _KExpQuadMGauss(_KernelEmbedding):
    """Kernel embedding of exponentiated quadratic kernel with Gaussian integration
    measure.

    Parameters
    ----------
    kernel: ExpQuad
        Instance of an exponentiated quadratic kernel
    measure: GaussianMeasure
        Instance of a Gaussian integration measure
    """

    def __init__(self, kernel: ExpQuad, measure: GaussianMeasure):
        super().__init__(kernel, measure)
        self.dim = self.kernel.input_dim

    def kernel_mean(self, x: np.ndarray) -> np.ndarray:

        if self.measure.diagonal_covariance:
            chol_inv_x = (x - self.measure.mean[:, None]) / (
                self.kernel.lengthscale ** 2 + np.diag(self.measure.cov)
            ).reshape(-1, 1)
            det_factor = (
                self.kernel.lengthscale ** self.dim
                / (self.kernel.lengthscale ** 2 * np.diag(self.measure.cov)).prod()
            )
        else:
            chol = slinalg.cho_factor(
                self.kernel.lengthscale * np.eye(self.dim) + self.measure.cov
            )
            chol_inv_x = slinalg.cho_solve(chol, x - self.measure.mean[:, None])

            det_factor = self.kernel.lengthscale ** self.dim / np.diag(chol[0]).prod()

        exp_factor = np.exp(-0.5 * (chol_inv_x ** 2)).sum(axis=0)
        return det_factor * exp_factor

    def kernel_variance(self) -> float:
        if self.measure.diagonal_covariance:
            denom = np.sqrt(
                (self.kernel.lengthscale ** 2 + 2.0 * np.diag(self.measure.cov)).prod()
            )
        else:
            chol, _ = slinalg.cho_factor(
                self.kernel.lengthscale * np.eye(self.dim) + 2 * self.measure.cov,
                lower=True,
            )
            denom = np.diag(chol).prod()

        return self.kernel.lengthscale ** self.dim / denom


class _KExpQuadMLebesgue(_KernelEmbedding):
    """Kernel embedding of exponentiated quadratic kernel with Lebesgue integration
    measure.

    Parameters
    ----------
    kernel: ExpQuad
        Instance of an exponentiated quadratic kernel
    measure: LebesgueMeasure
        Instance of a Lebesgue integration measure
    """

    def __init__(self, kernel: ExpQuad, measure: LebesgueMeasure):
        super().__init__(kernel, measure)
        self.dim = self.kernel.input_dim

    def kernel_mean(self, x):
        raise NotImplementedError

    def kernel_variance(self):
        raise NotImplementedError


def get_kernel_embedding(kernel: Kernel, measure: IntegrationMeasure):
    """Select the right kernel embedding given the kernel and integration measure.

    Parameters
    ----------
    kernel: Kernel
        Instance of a kernel
    measure: IntegrationMeasure
        Instance of an integration measure

    Returns
    -------
    an instance of KernelEmbedding
    """

    # Exponentiated quadratic kernel
    if isinstance(kernel, ExpQuad):
        if isinstance(measure, GaussianMeasure):
            return _KExpQuadMGauss(kernel, measure)
        elif isinstance(measure, LebesgueMeasure):
            return _KExpQuadMLebesgue(kernel, measure)
        raise NotImplementedError

    # other kernels
    raise NotImplementedError
