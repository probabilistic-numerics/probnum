"""
Contains integration measures
"""

import abc
from typing import Tuple, Optional, Union
from probnum.type import IntArgType, FloatArgType

import numpy as np


class IntegrationMeasure(abc.ABC):
    """
    An abstract class for a measure against which a target function is integrated.
    The integration measure is assumed normalized.
    """

    def __init__(self,
                 ndim: IntArgType,
                 domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray,
                                                                      FloatArgType]],
                 name: str
                 ):

        self._set_dimension_domain(ndim, domain)
        self._name = name

    def sample(self, n_sample):
        """
        Sample from integration measure.
        """
        raise NotImplementedError

    def _set_dimension_domain(self, ndim, domain):
        """
        Sets the integration domain and dimension. Error is thrown if the given
        dimension and domain limits do not match.

        THIS HAS NOT BEEN FULLY IMPLEMENTED YET. DOES NOT CHECK IF DIMENSIONS MATCH ETC.
        """
        if ndim >= 1:
            self.ndim = ndim
        else:
            # Error if the dimension is stupid
            pass
        if isinstance(domain[0], FloatArgType):
            # Use same domain limit in all dimensions if only one limit is given
            domain_a = np.full((self.ndim,), domain[0])
        else:
            domain_a = domain[0]
        if isinstance(domain[1], FloatArgType):
            domain_b = np.full((self.ndim,), domain[1])
        else:
            domain_b = domain[1]
        self.domain = (domain_a, domain_b)

class LebesgueMeasure(IntegrationMeasure):
    """
    A Lebesgue measure.
    """

    def __init__(self,
                 domain: Tuple[np.ndarray, np.ndarray]
                 ):

        super().__init__(domain=domain, name="Lebesgue measure")


class GaussianMeasure(IntegrationMeasure):
    """
    A Gaussian measure.
    """

    def __init__(self,
                 ndim: Optional[IntArgType],
                 mean: Optional[Union[np.ndarray, FloatArgType]] = 0.,
                 covariance: Optional[Union[np.ndarray, FloatArgType]] = 1.,
                 ):

        super().__init__(ndim=ndim, domain=(-np.Inf, np.Inf), name="Gaussian measure")
        self._set_mean_covariance(mean, covariance)

    def sample(self, n_sample):
        """
        Sample from Gaussian measure.
        """
        raise NotImplementedError

    def _set_mean_covariance(self, mean, covariance):
        """
        Sets the mean and covariance of the Gaussian integration measure. Throws
        error if their dimensions do not match with ndim.

        THIS HAS NOT BEEN FULLY IMPLEMENTED YET.
        """
        if isinstance(mean, FloatArgType):
            self.mean = np.full((self.ndim,), mean)
        else:
            self.mean = mean
        if isinstance(covariance, FloatArgType):
            self.cov = np.full((self.ndim,), covariance)
        else:
            self.cov = covariance
