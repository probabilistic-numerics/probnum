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
    Contains integrals over kernels
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
        # TODO: should we consider diagonal lengthscales and measure covs separately?
        self.dim = self.kernel.input_dim

    def qk(self, x: np.ndarray) -> np.ndarray:
        """
        Kernel mean w.r.t. its first argument against integration measure
        :param x: np.ndarray with shape (dim, n_eval)
            n_eval locations where to evaluate the kernel mean.
        :returns: np.ndarray
        """
        L = slinalg.cho_factor(self.kernel.lengthscale * np.eye(self.dim) + self.measure.covariance)
        Lx = slinalg.cho_solve(L, x - self.measure.mean)

        exp_factor = np.exp(-0.5*(Lx**2)).sum(axis=0) #shape (N,)
        det_factor = self.kernel.lengthscale**self.dim / np.diag(L).prod()

        return det_factor * exp_factor

    def qkq(self) -> float:
        L = slinalg.cho_factor(self.kernel.lengthscale * np.eye(self.dim) + 2*self.measure.covariance)

        return self.kernel.lengthscale**self.dim / np.diag(L).prod()


class KExpQuadMLebesgue(KernelEmbedding):
    def __init__(self, kernel: ExpQuad, measure: LebesgueMeasure):
        super(KExpQuadMGauss, self).__init__(kernel, measure)
        self.dim = self.kernel.input_dim

    def qk(self, x):
        raise NotImplementedError

    def qkq(self):
        raise NotImplementedError


def get_kernel_embedding(kernel:Kernel, measure:IntegrationMeasure):
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
