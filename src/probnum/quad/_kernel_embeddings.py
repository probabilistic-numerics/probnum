"""
Contains the kernel embeddings, i.e., integrals over kernels.
"""

import numpy as np

from ..kernels import Kernel
from ._integration_measures import IntegrationMeasure


class KernelEmbedding:
    """
    Contains integrals over kernels
    """

    def __init__(self, kernel: Kernel, measure: IntegrationMeasure):
        """
        Contains the kernel integrals
        """
        self.kernel = kernel
        self.measure = measure

    def qk(self, x):
        """
        Kernel mean w.r.t. its first argument against integration measure
        """
        raise NotImplementedError

    def qkq(self):
        """
        Kernel integrated in both arguments against integration measure
        """
        raise NotImplementedError
