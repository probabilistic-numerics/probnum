"""Noise on the components of a linear system."""

import dataclasses
from typing import Optional, Union

import numpy as np

import probnum.linops as linops

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

    epsA_cov: Optional[Union[np.ndarray, linops.LinearOperator]] = None
    epsb_cov: Optional[Union[np.ndarray, linops.LinearOperator]] = None
