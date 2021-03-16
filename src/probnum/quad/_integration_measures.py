"""Contains integration measures."""

import abc
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats

from probnum.random_variables._normal import Normal
from probnum.type import FloatArgType, IntArgType, RandomStateArgType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated."""

    def __init__(
        self,
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        random_state: Optional[RandomStateArgType] = None,
        name: Optional[str] = "Custom measure",
    ):

        self._set_dimension_domain(dim, domain)
        self.random_state = random_state
        self.name = name

    def __call__(self, points: Union[float, np.floating, np.ndarray]):
        return self.random_variable.pdf(points)

    def sample(self, n_sample):
        return np.squeeze(self.random_variable.sample(size=n_sample))

    def _set_dimension_domain(self, dim, domain):
        """Sets the integration domain and dimension.

        Error is thrown if the given dimension and domain limits do not
        match.
        """

        domain_a_dim = np.size(domain[0])
        domain_b_dim = np.size(domain[1])

        # Check that given dimensions match and are positive
        dim_mismatch = False
        if dim is None:
            if domain_a_dim == domain_b_dim:
                dim = domain_a_dim
            elif domain_a_dim == 1 or domain_b_dim == 1:
                dim = np.max([domain_a_dim, domain_b_dim])
            else:
                dim_mismatch = True
        else:
            if (domain_a_dim > 1 or domain_b_dim > 1) and dim != np.max(
                [domain_a_dim, domain_b_dim]
            ):
                dim_mismatch = True

        if dim_mismatch:
            raise ValueError(
                "Domain limits must have the same length or at least "
                "one of them has to be one-dimensional"
            )
        if dim < 1:
            raise ValueError(f"Domain dimension dim = {dim} must be positive.")

        # Use same domain limit in all dimensions if only one limit is given
        if domain_a_dim == 1:
            domain_a = np.full((dim,), domain[0])
        else:
            domain_a = domain[0]
        if domain_b_dim == 1:
            domain_b = np.full((dim,), domain[1])
        else:
            domain_b = domain[1]

        # Check that the domain is non-empty
        if not np.all(domain_a < domain_b):
            raise ValueError(f"Domain must be non-empty.")

        self.dim = dim
        self.domain = (domain_a, domain_b)


class LebesgueMeasure(IntegrationMeasure):
    """A Lebesgue measure."""

    def __init__(
        self,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        dim: Optional[IntArgType] = None,
        normalized: Optional[bool] = False,
        random_state: Optional[RandomStateArgType] = None,
    ):
        super().__init__(
            dim=dim, domain=domain, random_state=random_state, name="Lebesgue measure"
        )

        # Set normalization constant
        self.normalized = normalized
        if self.normalized:
            self.normalization_constant = 1 / np.prod(self.domain[1] - self.domain[0])
        else:
            self.normalization_constant = 1

        if self.normalization_constant in [0, np.Inf, -np.Inf]:
            raise ValueError(
                "Normalization constant is too small or too large. "
                "Consider setting normalized = False."
            )

        # Use scipy's uniform random variable since uniform random variables are not
        # yet implemented in probnum
        self.random_variable = scipy.stats.uniform(
            loc=self.domain[0], scale=self.domain[1] - self.domain[0]
        )

    def __call__(self, points: Union[float, np.floating, np.ndarray]):
        return np.full(np.shape(np.atleast_1d(points)), self.normalization_constant)

    def sample(self, n_sample):
        return np.squeeze(
            self.random_variable.rvs(
                size=(n_sample, self.dim), random_state=self.random_state
            )
        )


class GaussianMeasure(IntegrationMeasure):
    """A Gaussian measure."""

    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray],
        cov: Union[float, np.floating, np.ndarray],
        dim: Optional[IntArgType] = None,
        random_state: Optional[RandomStateArgType] = None,
    ):

        # Extend scalar mean and covariance to higher dimensions if dim has been
        # supplied by the user
        if (
            (np.isscalar(mean) or mean.size == 1)
            and (np.isscalar(cov) or cov.size == 1)
            and dim is not None
        ):
            mean = np.full((dim,), mean)
            cov = cov * np.eye(dim)

        # Set dimension based on the mean vector
        if np.isscalar(mean):
            dim = 1
        else:
            dim = mean.size

        # If cov has been given as a vector of variances, transform to diagonal matrix
        if isinstance(cov, np.ndarray) and np.squeeze(cov).ndim == 1 and dim > 1:
            cov = np.diag(np.squeeze(cov))

        # Exploit random variables to carry out mean and covariance checks
        self.random_variable = Normal(
            mean=np.squeeze(mean), cov=np.squeeze(cov), random_state=random_state
        )
        self.mean = self.random_variable.mean
        self.cov = self.random_variable.cov

        # Set diagonal_covariance flag
        if dim == 1:
            self.diagonal_covariance = True
        else:
            self.diagonal_covariance = (
                np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
            )

        super().__init__(
            dim=dim,
            domain=(np.full((dim,), -np.Inf), np.full((dim,), np.Inf)),
            random_state=random_state,
            name="Gaussian measure",
        )
