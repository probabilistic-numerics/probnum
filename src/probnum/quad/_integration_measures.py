"""
Contains integration measures
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class IntegrationMeasure(ABC):
    """
    An abstract class for a measure against which a target function is integrated.
    """

    def __init__(self, domain: Tuple[np.ndarray, np.ndarray], name: str):
        self.domain = domain
        self._name = name


class LebesgueMeasure(IntegrationMeasure):
    """
    A Lebesgue measure.
    """

    def __init__(self, domain: Tuple[np.ndarray, np.ndarray]):
        super().__init__(domain=domain, name="Lebesgue measure")


class GaussianMeasure(IntegrationMeasure):
    """
    A Gaussian measure.
    """

    def __init__(
        self,
        domain: Tuple[np.ndarray, np.ndarray],
        mean: np.ndarray,
        covariance: np.ndarray,
    ):
        super().__init__(domain=domain, name="Gaussian measure")
        self.mean = mean
        self.cov = covariance
