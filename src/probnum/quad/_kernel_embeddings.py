"""
Contains the kernel embeddings, i.e., integrals over kernels.
"""

import abc
import numpy as np
import scipy.linalg as slinalg

from ..kernels import Kernel, ExpQuad, RatQuad, Matern
from ._integration_measures import IntegrationMeasure, LebesgueMeasure, GaussianMeasure


class KernelEmbedding(abc.ABC):
    """
    Contains integrals over kernels.
    The naming scheme is 'K + kernel + M + measure'
    """

    def __init__(self, kernel: Kernel, measure: IntegrationMeasure):
        """
        Contains the kernel integrals
        """
        self.kernel = kernel
        self.measure = measure

    def qk(self, x: np.ndarray) -> np.ndarray:
        """
        Kernel mean w.r.t. its first argument against integration measure
        :param x: np.ndarray with shape (dim, n_eval)
            n_eval locations where to evaluate the kernel mean.
        """
        raise NotImplementedError

    def qkq(self) -> float:
        """
        Kernel integrated in both arguments against integration measure
        """
        raise NotImplementedError


class KExpQuadMGauss(KernelEmbedding):
    """
    Kernel embedding of exponentiated quadratic kernel with Gaussian Gaussian integration measure
    """

    def __init__(self, kernel: ExpQuad, measure: GaussianMeasure):
        super(KExpQuadMGauss, self).__init__(kernel, measure)
        # TODO: args are now child classes of args of the parent class.
        self.dim = self.kernel.input_dim

    def qk(self, x: np.ndarray) -> np.ndarray:
        """
        Kernel mean w.r.t. its first argument against integration measure
        :param x: np.ndarray with shape (dim, n_eval)
            n_eval locations where to evaluate the kernel mean.
        :returns: np.ndarray
        """
        if self.measure.diagonal_covariance:
            Linv_x = x / (
                self.kernel.lengthscale ** 2 + np.diag(self.measure.covariance)
            ).reshape(-1, 1)
            det_factor = (
                self.kernel.lengthscale ** self.dim
                / (
                    self.kernel.lengthscale ** 2 * np.diag(self.measure.covariance)
                ).prod()
            )
        else:
            L = slinalg.cho_factor(
                self.kernel.lengthscale * np.eye(self.dim) + self.measure.covariance
            )
            Linv_x = slinalg.cho_solve(L, x - self.measure.mean)

            det_factor = self.kernel.lengthscale ** self.dim / np.diag(L[0]).prod()

        exp_factor = np.exp(-0.5 * (Linv_x ** 2)).sum(axis=0)
        return det_factor * exp_factor

    def qkq(self) -> float:
        if self.measure.diagonal_covariance:
            denom = np.sqrt((self.kernel.lengthscale**2 + 2.*np.diag(self.measure.covariance)).prod())
        else:
            L = np.diag(slinalg.cholesky(
                self.kernel.lengthscale * np.eye(self.dim) + 2 * self.measure.covariance, lower=True
            ))
            denom = np.diag(L).prod()

        return self.kernel.lengthscale ** self.dim / denom


class KExpQuadMLebesgue(KernelEmbedding):
    def __init__(self, kernel: ExpQuad, measure: LebesgueMeasure):
        super(KExpQuadMGauss, self).__init__(kernel, measure)
        self.dim = self.kernel.input_dim

    def qk(self, x):
        raise NotImplementedError

    def qkq(self):
        raise NotImplementedError


def get_kernel_embedding(kernel: Kernel, measure: IntegrationMeasure):
    """
    Select the right kernel embedding given the kernel and integration measure

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
            return KExpQuadMGauss(kernel, measure)
        else:
            raise NotImplementedError

    # other kernels
    else:
        raise NotImplementedError
    # TODO: integrate all possible kernels with Monte Carlo.
