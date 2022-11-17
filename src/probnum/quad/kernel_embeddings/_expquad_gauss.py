"""Kernel embedding of exponentiated quadratic kernel with Gaussian integration
measure."""

import numpy as np
import scipy.linalg as slinalg

from probnum.quad.integration_measures import GaussianMeasure
from probnum.randprocs.kernels import ExpQuad


def _kernel_mean_expquad_gauss(
    x: np.ndarray, kernel: ExpQuad, measure: GaussianMeasure
) -> np.ndarray:
    r"""Kernel mean of the ExpQuad kernel with lenghtscale :math:`l` w.r.t. its first
    argument against a Gaussian measure with mean vector :math:`\mu` and covariance
    matrix :math:`\Sigma`. The kernel mean is

    .. math::

        \begin{equation}
            k_P(x)
            =
            \det( I + \Sigma / l^2)^{-1/2}
            \exp\bigg(-\frac{1}{2}(x-\mu)^\maths{T} (l^2 I + \Sigma)^{-1}
                        (x-\mu) \bigg),
        \end{equation}

    where :math:`I` is the identity matrix.

    Parameters
    ----------
    x :
        *shape=(n_eval, input_dim)* -- n_eval locations where to evaluate the kernel
        mean.
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a GaussianMeasure.

    Returns
    -------
    kernel_mean :
        *shape (n_eval,)* -- The kernel integrated w.r.t. its first argument,
        evaluated at locations ``x``.
    """
    (input_dim,) = kernel.input_shape

    if measure.diagonal_covariance:
        cov_diag = np.diag(measure.cov)
        chol_inv_x = (x - measure.mean) / np.sqrt(kernel.lengthscale**2 + cov_diag)
        det_factor = kernel.lengthscale**input_dim / np.sqrt(
            (kernel.lengthscale**2 + cov_diag).prod()
        )
        exp_factor = np.exp(-0.5 * (chol_inv_x**2).sum(axis=1))
    else:
        chol = slinalg.cho_factor(
            kernel.lengthscale**2 * np.eye(input_dim) + measure.cov,
            lower=True,
        )
        chol_inv_x = slinalg.cho_solve(chol, (x - measure.mean).T)
        exp_factor = np.exp(-0.5 * ((x - measure.mean) * chol_inv_x.T).sum(axis=1))
        det_factor = kernel.lengthscale**input_dim / np.diag(chol[0]).prod()

    return det_factor * exp_factor


def _kernel_variance_expquad_gauss(kernel: ExpQuad, measure: GaussianMeasure) -> float:
    r"""Kernel variance of the ExpQuad kernel with lenghtscale :math:`l` w.r.t. both
    arguments against a :math:`D`-dimensional Gaussian measure with mean vector
    :math:`\mu` and covariance matrix :math:`\Sigma`. The kernel variance is

    .. math::

        \begin{equation}
            k_{PP}
            =
            l^D \sqrt{\frac{1}{\det(l^2 I + 2\b{\Sigma})}}
        \end{equation}

    where :math:`I` is the identity matrix.

    Parameters
    ----------
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a GaussianMeasure.

    Returns
    -------
    kernel_variance :
        The kernel integrated w.r.t. both arguments.
    """
    (input_dim,) = kernel.input_shape

    if measure.diagonal_covariance:
        denom = (kernel.lengthscale**2 + 2.0 * np.diag(measure.cov)).prod()

    else:
        denom = np.linalg.det(
            kernel.lengthscale**2 * np.eye(input_dim) + 2.0 * measure.cov
        )

    return kernel.lengthscale**input_dim / np.sqrt(denom)
