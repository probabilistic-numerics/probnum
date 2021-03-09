"""Contains integration measures."""

import abc
from typing import Optional, Tuple, Union

import numpy as np

from probnum.random_variables._normal import Normal
from probnum.type import FloatArgType, IntArgType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    The integration measure is assumed normalized.
    """

    def __init__(
        self,
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        name: Optional[str] = "Custom measure",
    ):

        self._set_dimension_domain(dim, domain)
        self.name = name

    def __call__(self, points: Union[float, np.floating, np.ndarray]):
        return self.random_variable.pdf(points)

    def sample(self, n_sample):
        return np.squeeze(self.random_variable.sample(size=n_sample))

    def _set_dimension_domain(self, dim, domain):
        """Sets the integration domain and dimension. Error is thrown if the given
        dimension and domain limits do not match.

        TODO: check that dimensions match and the domain is not empty
        """
        if dim >= 1:
            self.dim = dim
        else:
            raise ValueError(f"Domain dimension dim={dim} must be positive.")

        if np.isscalar(domain[0]):
            # Use same domain limit in all dimensions if only one limit is given
            domain_a = np.full((dim, 1), domain[0])
        else:
            domain_a = domain[0]
        if np.isscalar(domain[1]):
            domain_b = np.full((dim, 1), domain[1])
        else:
            domain_b = domain[1]
        self.domain = (domain_a, domain_b)


class LebesgueMeasure(IntegrationMeasure):
    """A Lebesgue measure."""

    def __init__(self, domain: Tuple[np.ndarray, np.ndarray]):

        super().__init__(domain=domain, name="Lebesgue measure")

    def sample(self, n_sample):
        raise NotImplementedError


class GaussianMeasure(IntegrationMeasure):
    """A Gaussian measure."""

    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray],
        cov: Union[float, np.floating, np.ndarray],
    ):

        # Set dimension based on the mean vector
        if np.isscalar(mean):
            dim = 1
        else:
            dim = mean.size

        if dim > 1:
            if isinstance(cov, np.ndarray) and cov.size == dim:
                # cov has been given as vector of variances
                cov = np.diag(np.squeeze(cov))

        # Exploit random variables to carry out mean and covariance checks
        self.random_variable = Normal(np.squeeze(mean), np.squeeze(cov))
        self.mean = self.random_variable.mean
        self.cov = self.random_variable.cov

        # Set diagonal_covariance flag and reshape covariance to (1,1) if we are in 1d
        if dim == 1:
            self.diagonal_covariance = True
            self.cov = np.reshape(self.cov, (1, 1))
        else:
            self.diagonal_covariance = (
                np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
            )

        super().__init__(
            dim=dim,
            domain=(np.full((dim,), -np.Inf), np.full((dim,), np.Inf)),
            name="Gaussian measure",
        )
