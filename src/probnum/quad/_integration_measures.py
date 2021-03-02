"""Contains integration measures."""

import abc
from typing import Optional, Tuple, Union

import numpy as np

from probnum.type import FloatArgType, IntArgType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    The integration measure is assumed normalized.
    """

    def __init__(
        self,
        ndim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        name: str,
    ):

        self._set_dimension_domain(ndim, domain)
        self._name = name

    def sample(self, n_sample):
        """Sample from integration measure."""
        raise NotImplementedError

    def _set_dimension_domain(self, ndim, domain):
        """Sets the integration domain and dimension. Error is thrown if the given
        dimension and domain limits do not match.

        TODO: check that dimensions match and the domain is not empty
        """
        if ndim >= 1:
            self.ndim = ndim
        else:
            # Error if the dimension is stupid
            pass
        if np.isscalar(domain[0]):
            # Use same domain limit in all dimensions if only one limit is given
            domain_a = np.full((self.ndim, 1), domain[0])
        else:
            domain_a = domain[0]
        if np.isscalar(domain[1]):
            domain_b = np.full((self.ndim, 1), domain[1])
        else:
            domain_b = domain[1]
        self.domain = (domain_a, domain_b)


class LebesgueMeasure(IntegrationMeasure):
    """A Lebesgue measure."""

    def __init__(self, domain: Tuple[np.ndarray, np.ndarray]):

        super().__init__(domain=domain, name="Lebesgue measure")


class GaussianMeasure(IntegrationMeasure):
    """A Gaussian measure."""

    def __init__(
        self,
        ndim: Optional[IntArgType],
        mean: Optional[Union[np.ndarray, FloatArgType]] = 0.0,
        covariance: Optional[Union[np.ndarray, FloatArgType]] = 1.0,
    ):

        super().__init__(ndim=ndim, domain=(-np.Inf, np.Inf), name="Gaussian measure")
        self._set_mean_covariance(mean, covariance)

    def sample(self, n_sample):
        """Sample from Gaussian measure."""
        raise NotImplementedError

    def _set_mean_covariance(self, mean, covariance):
        """Sets the mean and covariance of the Gaussian integration measure. Throws
        error if their dimensions do not match with ndim.

        TODO: dimension or covariance positivity checks have not been implemented
        """
        if np.isscalar(mean):
            self.mean = np.full((self.ndim, 1), mean)
        else:
            self.mean = mean

        if np.isscalar(covariance):
            if covariance <= 0:
                # TODO: raise error
                pass
            self.covariance = covariance * np.eye(self.ndim)
            self.diagonal_covariance = True
        elif covariance.ndim == 1 or covariance.shape[0] != covariance.shape[1]:
            if not all(covariance > 0):
                # TODO: raise error
                pass
            self.covariance = np.diag(covariance)
            self.diagonal_covariance = True
        else:
            # TODO: raise error if covariance matrix is not positive-definite
            self.covariance = covariance
            self.diagonal_covariance = False
