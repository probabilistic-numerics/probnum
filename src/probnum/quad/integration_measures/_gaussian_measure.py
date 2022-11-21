"""The Gaussian measure."""


from __future__ import annotations

from typing import Optional, Union

import numpy as np

from probnum.randvars import Normal
from probnum.typing import IntLike

from ._integration_measure import IntegrationMeasure


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

        # Set domain as whole R^n
        domain = (np.full((input_dim,), -np.Inf), np.full((input_dim,), np.Inf))
        super().__init__(input_dim=input_dim, domain=domain)

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
