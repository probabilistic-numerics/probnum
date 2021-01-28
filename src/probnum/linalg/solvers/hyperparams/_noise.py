"""Noise on the components of a linear system."""

import dataclasses
from typing import Optional

import probnum.random_variables as rvs

from ._hyperparameters import LinearSolverHyperparams


@dataclasses.dataclass(frozen=True)
class LinearSystemNoise(LinearSolverHyperparams):
    r"""Additive Gaussian noise on the system matrix and right hand side.

    Parameters
    ----------
    epsA_cov :
        Covariance of the noise :math:`A + E` on the system matrix.
    epsb_cov :
        Covariance of the noise :math:`b + \epsilon` on the right hand side.
    """

    epsA_cov: Optional[rvs.Normal] = None
    epsb_cov: Optional[rvs.Normal] = None
