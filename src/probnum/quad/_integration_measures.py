"""Contains integration measures."""

import abc
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats

from probnum.randvars import Normal
from probnum.typing import FloatArgType, IntArgType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    Child classes implement specific integration measures and, if available, make use
    of random variables for sampling and evaluation of the density function.

    Parameters
    ----------
    dim :
        Dimension of the integration domain.
    domain :
        Tuple which contains two arrays which define the start and end points,
        respectively, of the rectangular integration domain.
    """

    def __init__(
        self,
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
    ) -> None:

        self._set_dimension_domain(dim, domain)

    def __call__(self, points: Union[float, np.floating, np.ndarray]) -> np.ndarray:
        """Evaluate the density function of the integration measure.

        Parameters
        ----------
        points :
            *shape=(n_points,) or (n_points, dim)* -- Input locations.

        Returns
        -------
        density_evals :
            *shape=(n_points,)* -- Density evaluated at given locations.
        """
        # pylint: disable=no-member
        return self.random_variable.pdf(points).squeeze()

    def sample(
        self,
        rng: np.random.Generator,
        n_sample: IntArgType,
    ) -> np.ndarray:
        """Sample ``n_sample`` points from the integration measure.

        Parameters
        ----------
        rng :
            Random number generator
        n_sample :
            Number of points to be sampled

        Returns
        -------
        points :
            *shape=(n_sample,) or (n_sample,dim)* -- Sampled points
        """
        # pylint: disable=no-member
        return np.reshape(
            self.random_variable.sample(rng=rng, size=n_sample),
            newshape=(n_sample, self.dim),
        )

    def _set_dimension_domain(
        self,
        dim: IntArgType,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
    ) -> None:
        """Sets the integration domain and dimension.

        The following logic is used to set the domain and dimension:
            1. If ``dim`` is not given (``dim == None``):
                1a. If either ``domain[0]`` or ``domain[1]`` is a scalar, the dimension
                    is set as the maximum of their lengths and the scalar is expanded to
                    a constant vector.
                1b. Otherwise, if the ``domain[0]`` and ``domain[1]`` are not of equal
                    length, an error is raised.
            2. If ``dim`` is given:
                2a. If both ``domain[0]`` and ``domain[1]`` are scalars, they are
                    expanded to constant vectors of length ``dim``.
                2b. If only one of `domain[0]`` and ``domain[1]`` is a scalar and the
                    length of the other equals ``dim``, the scalar one is expanded to a
                    constant vector of length ``dim``.
                2c. Otherwise, if neither of ``domain[0]`` and ``domain[1]`` is a
                    scalar, error is raised if either of them has length which does not
                    equal ``dim``.
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
                "one of them has to be one-dimensional."
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
            raise ValueError("Domain must be non-empty.")

        self.dim = dim
        self.domain = (domain_a, domain_b)


class LebesgueMeasure(IntegrationMeasure):
    """Lebesgue measure on a hyper-rectangle.

    Parameters
    ----------
    dim :
        Dimension of the integration domain
    domain :
        Tuple which contains two arrays which define the start and end points,
        respectively, of the rectangular integration domain.
    normalized :
         Boolean which controls whether or not the measure is normalized (i.e.,
         integral over the domain is one).
    """

    def __init__(
        self,
        domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
        dim: Optional[IntArgType] = None,
        normalized: Optional[bool] = False,
    ) -> None:
        super().__init__(dim=dim, domain=domain)

        # Set normalization constant
        self.normalized = normalized
        if self.normalized:
            self.normalization_constant = 1.0 / np.prod(self.domain[1] - self.domain[0])
        else:
            self.normalization_constant = 1.0

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

    def __call__(self, points: Union[float, np.floating, np.ndarray]) -> np.ndarray:
        num_dat = np.atleast_1d(points).shape[0]
        return np.full(() if num_dat == 1 else (num_dat,), self.normalization_constant)

    def sample(
        self,
        rng: np.random.Generator,
        n_sample: IntArgType,
    ) -> np.ndarray:
        return self.random_variable.rvs(size=(n_sample, self.dim), random_state=rng)


# pylint: disable=too-few-public-methods
class GaussianMeasure(IntegrationMeasure):
    """Gaussian measure on Euclidean space with given mean and covariance.

    If ``mean`` and ``cov`` are scalars but ``dim`` is larger than one, ``mean`` and
    ``cov`` are extended to a constant vector and diagonal matrix, respectively,
    of appropriate dimensions.

    Parameters
    ----------
    mean :
        *shape=(dim,)* -- Mean of the Gaussian measure.
    cov :
        *shape=(dim, dim)* -- Covariance matrix of the Gaussian measure.
    dim :
        Dimension of the integration domain.
    """

    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray],
        cov: Union[float, np.floating, np.ndarray],
        dim: Optional[IntArgType] = None,
    ) -> None:

        # Extend scalar mean and covariance to higher dimensions if dim has been
        # supplied by the user
        # pylint: disable=fixme
        # TODO: This needs to be modified to account for cases where only either the
        #  mean or covariance is given in scalar form
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
        self.random_variable = Normal(mean=np.squeeze(mean), cov=np.squeeze(cov))
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
        )
