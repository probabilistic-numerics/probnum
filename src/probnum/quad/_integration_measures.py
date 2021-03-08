"""Contains integration measures."""

import abc
<<<<<<< HEAD
from typing import Tuple, Optional, Union
from probnum.type import IntArgType, FloatArgType
from probnum.random_variables._normal import Normal
=======
from typing import Optional, Tuple, Union
>>>>>>> c17bafe99b2587386912dd86b6584107bf62cb13

import numpy as np

from probnum.type import FloatArgType, IntArgType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    The integration measure is assumed normalized.
    """

    def __init__(
        self,
<<<<<<< HEAD
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        name: str = "Custom measure",
=======
        ndim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        name: str,
>>>>>>> c17bafe99b2587386912dd86b6584107bf62cb13
    ):

        self._set_dimension_domain(dim, domain)
        self._name = name

    def sample(self, n_sample):
        """Sample from integration measure."""
        raise NotImplementedError

<<<<<<< HEAD
    def _set_dimension_domain(self, dim, domain):
        """
        Sets the integration domain and dimension. Error is thrown if the given
=======
    def _set_dimension_domain(self, ndim, domain):
        """Sets the integration domain and dimension. Error is thrown if the given
>>>>>>> c17bafe99b2587386912dd86b6584107bf62cb13
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
    """A Lebesgue measure."""

    def __init__(self, domain: Tuple[np.ndarray, np.ndarray]):

        super().__init__(domain=domain, name="Lebesgue measure")


class GaussianMeasure(IntegrationMeasure):
    """A Gaussian measure."""

    def __init__(
        self,
<<<<<<< HEAD
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
=======
        ndim: Optional[IntArgType],
        mean: Optional[Union[np.ndarray, FloatArgType]] = 0.0,
        covariance: Optional[Union[np.ndarray, FloatArgType]] = 1.0,
    ):
>>>>>>> c17bafe99b2587386912dd86b6584107bf62cb13

        super().__init__(
            dim=self.dim,
            domain=(np.full((self.dim,), -np.Inf), np.full((self.dim,), np.Inf)),
            name="Gaussian measure",
        )

    def sample(self, n_sample):
<<<<<<< HEAD
        if self.dim == 1:
            return self.random_variable._univariate_sample(size=(n_sample, 1))
        else:
            return self.random_variable._dense_sample(size=n_sample)
=======
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
>>>>>>> c17bafe99b2587386912dd86b6584107bf62cb13
