"""
Contains integration measures
"""

import abc
from typing import Tuple, Optional, Union
from probnum.type import IntArgType, FloatArgType
from probnum.random_variables._normal import Normal

import numpy as np


class IntegrationMeasure(abc.ABC):
    """
    An abstract class for a measure against which a target function is integrated.
    The integration measure is assumed normalized.
    """

    def __init__(
        self,
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        name: str = "Custom measure",
    ):

        self._set_dimension_domain(dim, domain)
        self._name = name

    def sample(self, n_sample):
        """
        Sample from integration measure.
        """
        raise NotImplementedError

    def _set_dimension_domain(self, dim, domain):
        """
        Sets the integration domain and dimension. Error is thrown if the given
        dimension and domain limits do not match.

        TODO: check that dimensions match and the domain is not empty
        """
        if dim >= 1:
            self.dim = dim
        else:
            raise ValueError(f"Domain dimension dim={dim} must be positive.")
        if np.isscalar(domain[0]):
            # Use same domain limit in all dimensions if only one limit is given
            domain_a = np.full((self.dim, 1), domain[0])
        else:
            domain_a = domain[0]
        if np.isscalar(domain[1]):
            domain_b = np.full((self.dim, 1), domain[1])
        else:
            domain_b = domain[1]
        self.domain = (domain_a, domain_b)


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
        mean: Union[float, np.floating, np.ndarray],
        cov: Union[float, np.floating, np.ndarray],
    ):
        # Exploit random variables to carry out mean and covariance checks
        self.random_variable = Normal(mean, cov)
        self.mean = self.random_variable.mean
        self.cov = self.random_variable.cov
        self.diagonal_covariance = not bool(
            np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov)))
        )

        # Use the mean to set the dimension
        if len(self.mean.shape) == 0:
            self.dim = 1
        else:
            self.dim = len(self.mean)

        super().__init__(
            dim=self.dim,
            domain=(np.full((self.dim,), -np.Inf), np.full((self.dim,), np.Inf)),
            name="Gaussian measure",
        )

    def sample(self, n_sample):
        if self.dim == 1:
            return self.random_variable._univariate_sample(size=(n_sample, 1))
        else:
            return self.random_variable._dense_sample(size=n_sample)