"""Kernel embedding of exponentiated quadratic kernel with Lebesgue integration
measure."""

# pylint: disable=no-name-in-module

import numpy as np
from scipy.special import erf

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.randprocs.kernels import ExpQuad


def _kernel_mean_expquad_lebesgue(
    x: np.ndarray, kernel: ExpQuad, measure: LebesgueMeasure
) -> np.ndarray:
    r"""Kernel mean of the ExpQuad kernel with lenghtscale :math:`l` w.r.t. its first
    argument against a Lebesgue measure on the hyper-rectangle
    :math:`[a_1, b_1] \times \cdots \times [a_D, b_D]`. For unnormalised Lebesgue
    measure the kernel mean is

    .. math::

        \begin{equation}
            k_P(x)
            =
            \bigg( \frac{\pi}{2} \bigg)^{D/2} l^D \prod_{i=1}^D
            \Bigg[ \mathrm{erf}\bigg( \frac{b_i-x_i}{l \sqrt{2}}\bigg)
            - \erf\bigg(\frac{a_i-x_i}{l\sqrt{2}}\bigg) \Bigg]
        \end{equation}

    where :math:`\mathrm{erf} = \frac{1}{\sqrt{\pi}} \int_{-x}^x \exp(-t^2) dt` is the
    standard error function.

    Parameters
    ----------
    x :
        *shape (n_eval, input_dim)* -- n_eval locations where to evaluate the kernel
        mean.
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a LebesgueMeasure.

    Returns
    -------
    kernel_mean :
        *shape=(n_eval,)* -- The kernel integrated w.r.t. its first argument,
        evaluated at locations ``x``.
    """
    (input_dim,) = kernel.input_shape

    ell = kernel.lengthscale
    return (
        measure.normalization_constant
        * (np.pi * ell**2 / 2) ** (input_dim / 2)
        * (
            erf((measure.domain[1] - x) / (ell * np.sqrt(2)))
            - erf((measure.domain[0] - x) / (ell * np.sqrt(2)))
        ).prod(axis=1)
    )


def _kernel_variance_expquad_lebesgue(
    kernel: ExpQuad, measure: LebesgueMeasure
) -> float:
    r"""Kernel variance of the ExpQuad kernel with lenghtscale :math:`l` w.r.t. both
    arguments against a Lebesgue measure on the hyper-rectangle
    :math:`[a_1, b_1] \times \cdots \times [a_D, b_D]`. For unnormalised Lebesgue
    measure the kernel variance is

    .. math::

        \begin{equation}
            k_{PP}
            =
            ( 2 \pi )^{D/2} l^D \prod_{i=1}^D
            \Bigg[ \frac{l\sqrt{2} }{\sqrt{\pi}}\bigg(
                \exp\bigg(-\frac{(b_i - a_i)^2}{2l^2} \bigg) - 1 \bigg) + (b_i - a_i)
                \mathrm{erf}\bigg( \frac{b_i - a_i}{l\sqrt{2}} \bigg) \Bigg]
        \end{equation}

    where :math:`\mathrm{erf} = \frac{1}{\sqrt{\pi}} \int_{-x}^x \exp(-t^2) dt` is the
    standard error function.

    Parameters
    ----------
    kernel :
        Instance of an ExpQuad kernel.
    measure :
        Instance of a LebesgueMeasure.

    Returns
    -------
    kernel_variance :
        The kernel integrated w.r.t. both arguments.
    """
    (input_dim,) = kernel.input_shape

    r = measure.domain[1] - measure.domain[0]
    ell = kernel.lengthscale
    return (
        measure.normalization_constant**2
        * (2 * np.pi * ell**2) ** (input_dim / 2)
        * (
            ell * np.sqrt(2 / np.pi) * (np.exp(-(r**2) / (2 * ell**2)) - 1)
            + r * erf(r / (ell * np.sqrt(2)))
        ).prod()
    )
