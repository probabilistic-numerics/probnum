"""Kernel embedding of exponentiated quadratic kernel with Gaussian integration
measure."""


import numpy as np
import scipy.linalg as slinalg

from probnum.quad._integration_measures import GaussianMeasure
from probnum.randprocs.kernels import ExpQuad

# pylint: disable=invalid-name


def _kernel_mean_expquad_gauss(
    x: np.ndarray, kernel: ExpQuad, measure: GaussianMeasure
) -> np.ndarray:
    """Kernel mean of the ExpQuad kernel w.r.t. its first argument against a Gaussian
    measure.

    Parameters
    ----------
    x :
        *shape=(n_eval, input_dim)* -- n_eval locations where to evaluate the kernel mean.
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a GaussianMeasure.

    Returns
    -------
    k_mean :
        *shape (n_eval,)* -- The kernel integrated w.r.t. its first argument,
        evaluated at locations x.
    """
    input_dim = kernel.input_dim

    if measure.diagonal_covariance:
        cov_diag = np.diag(measure.cov)
        chol_inv_x = (x - measure.mean) / np.sqrt(kernel.lengthscale ** 2 + cov_diag)
        det_factor = kernel.lengthscale ** input_dim / np.sqrt(
            (kernel.lengthscale ** 2 + cov_diag).prod()
        )
        exp_factor = np.exp(-0.5 * (chol_inv_x ** 2).sum(axis=1))
    else:
        chol = slinalg.cho_factor(
            kernel.lengthscale ** 2 * np.eye(input_dim) + measure.cov,
            lower=True,
        )
        chol_inv_x = slinalg.cho_solve(chol, (x - measure.mean).T)
        exp_factor = np.exp(-0.5 * ((x - measure.mean) * chol_inv_x.T).sum(axis=1))
        det_factor = kernel.lengthscale ** input_dim / np.diag(chol[0]).prod()

    return det_factor * exp_factor


def _kernel_variance_expquad_gauss(kernel: ExpQuad, measure: GaussianMeasure) -> float:
    """Kernel variance of the ExpQuad kernel w.r.t. both arguments against a Gaussian
    measure.

    Parameters
    ----------
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a GaussianMeasure.

    Returns
    -------
    k_var :
        The kernel integrated w.r.t. both arguments.
    """
    input_dim = kernel.input_dim

    if measure.diagonal_covariance:
        denom = (kernel.lengthscale ** 2 + 2.0 * np.diag(measure.cov)).prod()

    else:
        denom = np.linalg.det(
            kernel.lengthscale ** 2 * np.eye(input_dim) + 2.0 * measure.cov
        )

    return kernel.lengthscale ** input_dim / np.sqrt(denom)
