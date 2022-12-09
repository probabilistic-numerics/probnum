"""Base class of an integration measure for Bayesian quadrature."""

from __future__ import annotations

import abc
from typing import Optional, Union

import numpy as np

from probnum.quad._utils import as_domain
from probnum.quad.typing import DomainLike
from probnum.typing import FloatLike, IntLike


class IntegrationMeasure(abc.ABC):
    """An abstract class for a measure against which a target function is integrated.

    Child classes implement specific integration measures and, if available, make use
    of random variables for sampling and evaluation of the density function.

    Parameters
    ----------
    domain
        Domain of integration. Contains lower and upper bound as a scalar or
        ``np.ndarray``.
    input_dim
        Dimension of the integration domain.
    """

    def __init__(
        self,
        domain: DomainLike,
        input_dim: Optional[IntLike],
    ) -> None:

        self.domain, self.input_dim = as_domain(domain, input_dim)

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
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ``n_sample`` points from the integration measure.

        Parameters
        ----------
        n_sample
            Number of points to be sampled
        rng
            A Random number generator.

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
