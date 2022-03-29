"""Contains integration measures."""
from __future__ import annotations

import abc
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats

from probnum.randvars import Normal
from probnum.typing import FloatLike, IntLike

from ._quad_typing import DomainLike, _DomainType


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    Child classes implement specific integration measures and, if available, make use
    of random variables for sampling and evaluation of the density function.

    Parameters
    ----------
    input_dim
        Dimension of the integration domain.
    domain
        Domain of integration. Contains lower and upper bound as a scalar or
        ``np.ndarray``. ``None`` if bounds are infinite.
    """

    def __init__(
        self,
        domain: Optional[DomainLike],
        input_dim: Optional[IntLike],
    ) -> None:

        self.domain, self.input_dim = self.as_domain(domain, input_dim)

    def __call__(self, points: Union[FloatLike, np.ndarray]) -> np.ndarray:
        """Evaluate the density function of the integration measure.

        Parameters
        ----------
        points
            *shape=(n_points, input_dim)* -- Input locations.

        Returns
        -------
        density_evals :
            *shape=(n_points,)* -- Density evaluated at given locations.
        """
        # pylint: disable=no-member
        return self.random_variable.pdf(points).reshape(-1)

    def sample(
        self,
        n_sample: IntLike,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ) -> np.ndarray:
        """Sample ``n_sample`` points from the integration measure.

        Parameters
        ----------
        n_sample
            Number of points to be sampled
        rng
            Random number generator. Optional. Default is `np.random.default_rng()`.

        Returns
        -------
        points :
            *shape=(n_sample,input_dim)* -- Sampled points
        """
        # pylint: disable=no-member
        return np.reshape(
            self.random_variable.sample(size=n_sample, rng=rng),
            newshape=(n_sample, self.input_dim),
        )

    @staticmethod
    def as_domain(
        domain: Optional[DomainLike],
        input_dim: Optional[IntLike],
    ) -> Tuple[Optional[_DomainType], Optional[int]]:
        """Static method that converts the integration domain and input dimension to
        the correct types.

        If no ``input_dim`` is given, the dimension is inferred from the lengths of
        domain limits ``domain[0]`` and ``domain[1]``. These must be either scalars
        or arrays of equal length.

        If ``input_dim`` is given, the domain limits must be either scalars or arrays.
        If they are arrays, their lengths must equal ``input_dim``. If they are scalars,
        the domain is taken to be the hypercube
        ``[domain[0], domain[1]] x .... x [domain[0], domain[1]]``
        of dimension ``input_dim``.
        """
        if input_dim is not None and input_dim < 1:
            raise ValueError(
                f"If given, input dimension must be positive. Current value "
                f"is ({input_dim})."
            )

        # Neither domain, nor input dimension is given: do nothing
        if domain is None and input_dim is None:
            return domain, input_dim

        # Only input dimension is given: convert input dimension only
        if domain is None:
            return domain, int(input_dim)

        # Domain is given
        # Domain limits must have equal dimensions
        if np.size(domain[0]) != np.size(domain[1]):
            raise ValueError(
                f"Domain limits must be given either as scalars or arrays "
                f"of equal dimension. Current sizes are ({np.size(domain[0])}) "
                f"and ({np.size(domain[1])})."
            )

        domain_dim = int(np.size(domain[0]))

        # Input dimension not given, infer it from the domain.
        if input_dim is None:
            input_dim = domain_dim
            domain_a, domain_b = domain

        # Input dimension given and bounds are given as scalars: Expand domain limits.
        elif input_dim is not None and domain_dim == 1:
            domain_a = np.full((input_dim,), domain[0])
            domain_b = np.full((input_dim,), domain[1])

        # Bounds are given as arrays
        else:
            # Size of domain and input dimension do not match
            if input_dim != domain_dim:
                raise ValueError(
                    "If domain limits are not scalars, their lengths "
                    "must match the input dimension."
                )
            domain_a = domain[0]
            domain_b = domain[1]

        # convert bounds to 1D arrays if necessary
        domain_a = np.atleast_1d(domain_a)
        domain_b = np.atleast_1d(domain_b)
        if domain_a.ndim > 1 or domain_b.ndim > 1:
            raise ValueError(
                f"Upper ({domain_b.ndim}) or lower ({domain_a.ndim}) "
                f"bounds have too many dimensions."
            )

        # Make sure the domain is non-empty
        if not np.all(domain_a < domain_b):
            raise ValueError(
                "Integration domain must be non-empty. For example, some "
                "lower bounds may be larger than their corresponding "
                "upper bounds."
            )

        domain = (domain_a, domain_b)
        return domain, input_dim


class LebesgueMeasure(IntegrationMeasure):
    """Lebesgue measure on a hyper-rectangle.

    Parameters
    ----------
    domain
        Domain of integration. Contains lower and upper bound as a scalar or
        ``np.ndarray``.
    input_dim
        Dimension of the integration domain. If not given, inferred from ``domain``.
    normalized
         Boolean which controls whether or not the measure is normalized (i.e.,
         integral over the domain is one).
    """

    def __init__(
        self,
        domain: DomainLike,
        input_dim: Optional[IntLike] = None,
        normalized: Optional[bool] = False,
    ) -> None:
        super().__init__(input_dim=input_dim, domain=domain)

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

    def __call__(self, points: np.ndarray) -> np.ndarray:
        num_dat = points.shape[0]
        return np.full((num_dat,), self.normalization_constant)

    def sample(
        self,
        n_sample: IntLike,
        rng: Optional[np.random.Generator] = np.random.default_rng(),
    ) -> np.ndarray:
        return self.random_variable.rvs(
            size=(n_sample, self.input_dim), random_state=rng
        )


# pylint: disable=too-few-public-methods
class GaussianMeasure(IntegrationMeasure):
    """Gaussian measure on Euclidean space with given mean and covariance.

    If ``mean`` and ``cov`` are scalars but ``input_dim`` is larger than one, ``mean``
    and ``cov`` are extended to a constant vector and diagonal matrix, respectively,
    of appropriate dimensions.

    Parameters
    ----------
    mean
        *shape=(input_dim,)* -- Mean of the Gaussian measure.
    cov
        *shape=(input_dim, input_dim)* -- Covariance matrix of the Gaussian measure.
    input_dim
        Dimension of the integration domain.
    """

    def __init__(
        self,
        mean: Union[float, np.floating, np.ndarray],
        cov: Union[float, np.floating, np.ndarray],
        input_dim: Optional[IntLike] = None,
    ) -> None:

        # Extend scalar mean and covariance to higher dimensions if input_dim has been
        # supplied by the user
        if (
            (np.isscalar(mean) or mean.size == 1)
            and (np.isscalar(cov) or cov.size == 1)
            and input_dim is not None
        ):
            mean = np.full((input_dim,), mean)
            cov = cov * np.eye(input_dim)

        # Set dimension based on the mean vector
        if np.isscalar(mean):
            input_dim = 1
        else:
            input_dim = mean.size

        super().__init__(input_dim=input_dim, domain=None)

        # Exploit random variables to carry out mean and covariance checks
        # squeezes are needed due to the way random variables are currently implemented
        # pylint: disable=no-member
        self.random_variable = Normal(mean=np.squeeze(mean), cov=np.squeeze(cov))
        self.mean = np.reshape(self.random_variable.mean, (self.input_dim,))
        self.cov = np.reshape(
            self.random_variable.cov, (self.input_dim, self.input_dim)
        )

        # Set diagonal_covariance flag
        if input_dim == 1:
            self.diagonal_covariance = True
        else:
            self.diagonal_covariance = (
                np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
            )
