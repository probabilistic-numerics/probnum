"""Kernel embedding of Matern kernels with Lebesgue integration measure."""


from typing import Tuple, Union

import numpy as np

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.randprocs.kernels import Matern, ProductMatern


def _kernel_mean_matern_lebesgue(
    x: np.ndarray, kernel: Union[Matern, ProductMatern], measure: LebesgueMeasure
) -> np.ndarray:
    r"""Kernel mean of a ProductMatern or 1D Matern kernel w.r.t. its first argument
    against a Lebesgue measure.

    For a Matern kernel with lengthscale :math:`l` and smoothness :math:`\nu`
    and the unnormalised Lebesgue measure on :math:`[a, b]` the kernel mean is

    .. math::

        \begin{align}
            k_P^{\nu=1/2}(x)
            &=
            l \bigg[ 2 - \exp\bigg(\frac{a-x}{l}\bigg)
                        - \exp\bigg(\frac{x-b}{l}\bigg) \bigg]
            , \\
            k_P^{\nu=3/2}(x)
            &=
            \bigg[ \frac{4 l}{\sqrt{3}} - \frac{1}{3}
            \exp\bigg( \frac{\sqrt{3}(x-b)}{l} \bigg) \big(3b+2\sqrt{3}\, l-3x\big)
                -\frac{1}{3} \exp\bigg(\frac{\sqrt{3}(a-x)}{l}\bigg)\big(3x+2\sqrt{3}\,l
                -3a\big) \bigg]
            , \\
            k_P^{\nu=5/2}(x)
            &=
            \bigg[ \frac{16 l}{3\sqrt{5}} - \frac{1}{15 l}
                \exp\bigg( \frac{\sqrt{5}(x-b)}{l} \bigg) \big( 8\sqrt{5}\, l^2
                    + 25 l(b-x)+5\sqrt{5}(b-x)^2 \big)
                -\frac{1}{15 l} \exp\bigg( \frac{\sqrt{5}(a-x)}{l} \bigg)
                \big( 8\sqrt{5}\, l^2 + 25\ell(x-a) + 5\sqrt{5}(a-x)^2 \big) \bigg]
            , \\
            k_P^{\nu=7/2}(x)
            &=
            \frac{1}{105 l^2} \bigg[ 96\sqrt{7} l^3 -
                \exp\bigg( \frac{\sqrt{7}(x-b)}{l} \bigg)
                \big( 48\sqrt{7} l^3-231 l^2(x-b)
                + 63\sqrt{7} l(x-b)^2 - 49(x-b)^3\big)
                - \exp\bigg(\frac{\sqrt{7}(a-x)}{l}\bigg)
                \big( 48\sqrt{7} l^3 + 231 l^2(x-a) + 63\sqrt{7} \, l (x-a)^2
                + 49(x-a)^3\big) \bigg].
        \end{align}

    For product Materns kernel means are obtained by taking products of those for
    1D Materns above.

    Parameters
    ----------
    x :
        *shape (n_eval, input_dim)* -- n_eval locations where to evaluate the
                                       kernel mean.
    kernel :
        A product Matern kernel, or a 1D Matern kernel.
    measure :
        Instance of a LebesgueMeasure.

    Returns
    -------
    k_mean :
        *shape=(n_eval,)* -- The kernel integrated w.r.t. its first argument,
                             evaluated at locations x.
    """
    kernel = _convert_to_product_matern(kernel)
    (input_dim,) = kernel.input_shape

    # Compute kernel mean via a product of one-dimensional kernel means
    kernel_mean = np.ones((x.shape[0],))
    for dim in range(input_dim):
        kernel_mean *= _kernel_mean_matern_1d_lebesgue(
            x=x[:, dim],
            kernel=kernel.univariate_materns[dim],
            domain=(measure.domain[0][dim], measure.domain[1][dim]),
        )

    return measure.normalization_constant * kernel_mean


def _kernel_variance_matern_lebesgue(
    kernel: Union[Matern, ProductMatern], measure: LebesgueMeasure
) -> float:
    r"""Kernel variance of a ProductMatern or 1D Matern kernel w.r.t. both arguments
    against a Lebesgue measure.

    For a Matern kernel with lengthscale :math:`l` and smoothness :math:`\nu`
    and the unnormalised Lebesgue measure on :math:`[a, b]` the kernel variance is

    .. math::

        \begin{align}
            k_{PP}^{\nu=1/2}
            &=
            2l \bigg[ r + l \bigg( \exp\bigg( -\frac{r}{l} \bigg) - 1 \bigg) \bigg]
            , \\
            k_{PP}^{\nu=3/2}
            &=
            \frac{2 l}{3} \bigg[ 2\sqrt{3}\,r - 3 l
                + \exp\bigg( -\frac{\sqrt{3}\,r}{l} \bigg)
                \big( \sqrt{3}\, r + 3l \big) \bigg]
            , \\
            k_{PP}^{\nu=5/2}
            &=
            \frac{1}{15} \bigg[ 2l \big( 8\sqrt{5}\, r - 15 l \big)
                + 2\exp\bigg( -\frac{\sqrt{5}\, r}{l} \bigg)
                \big( 5a^2 -10ab +5b^2 +7\sqrt{5}\, r l + 15 l^2 \big) \bigg]
            , \\
            k_{PP}^{\nu=7/2}
            &=
            \frac{1}{105 l} \bigg[ 2\exp\bigg( -\frac{\sqrt{7}\, r}{l} \bigg)
                \Big( 7\sqrt{7}(b^3-a^3) + 84b^2 l + 57\sqrt{7}\, b l^2
                + 105 l^3 + 21a^2 \big( \sqrt{7}\, b + 4 l \big)
                - 3a\big( 7\sqrt{7}\, b^2 + 56b l + 19\sqrt{7}\, l^2 \big) \Big)
                - 6 l^2\big( 35l - 16\sqrt{7}\, r \big) \bigg],
        \end{align}

    where :math:`r = b - a`.

    For product Materns kernel variances are obtained by taking products of those for
    1D Materns above.

    Parameters
    ----------
    kernel :
        A product Matern kernel, or a 1D Matern kernel.
    measure :
        Instance of a LebesgueMeasure.

    Returns
    -------
    k_var :
        The kernel integrated w.r.t. both arguments.
    """

    kernel = _convert_to_product_matern(kernel)
    (input_dim,) = kernel.input_shape

    # Compute kernel mean via a product of one-dimensional kernel variances
    kernel_variance = 1.0
    for dim in range(input_dim):
        kernel_variance *= _kernel_variance_matern_1d_lebesgue(
            kernel=kernel.univariate_materns[dim],
            domain=(measure.domain[0][dim], measure.domain[1][dim]),
        )

    return measure.normalization_constant**2 * kernel_variance


def _convert_to_product_matern(kernel: Matern) -> ProductMatern:
    """Convert a 1D Matern kernel to a ProductMatern for unified treatment."""
    (input_dim,) = kernel.input_shape
    if isinstance(kernel, Matern):
        if input_dim > 1:
            raise NotImplementedError(
                "Kernel embeddings have been implemented only for "
                "Matérn kernels in dimension one and product Matern kernels."
            )
        kernel = ProductMatern(
            input_shape=kernel.input_shape,
            lengthscales=kernel.lengthscale,
            nus=kernel.nu,
        )
    return kernel


def _kernel_mean_matern_1d_lebesgue(
    x: np.ndarray, kernel: Matern, domain: Tuple
) -> np.ndarray:
    """Kernel means for 1D Matern kernels.

    Note that these are for unnormalized Lebesgue measure.
    """
    (a, b) = domain
    ell = kernel.lengthscale
    if kernel.nu == 0.5:
        unnormalized_mean = ell * (2.0 - np.exp((a - x) / ell) - np.exp((x - b) / ell))
    elif kernel.nu == 1.5:
        unnormalized_mean = (
            4.0 * ell / np.sqrt(3.0)
            - np.exp(np.sqrt(3.0) * (x - b) / ell)
            / 3.0
            * (3.0 * b + 2.0 * np.sqrt(3.0) * ell - 3.0 * x)
            - np.exp(np.sqrt(3.0) * (a - x) / ell)
            / 3.0
            * (3.0 * x + 2.0 * np.sqrt(3.0) * ell - 3.0 * a)
        )
    elif kernel.nu == 2.5:
        unnormalized_mean = (
            16.0 * ell / (3.0 * np.sqrt(5.0))
            - np.exp(np.sqrt(5.0) * (x - b) / ell)
            / (15.0 * ell)
            * (
                8.0 * np.sqrt(5.0) * ell**2
                + 25.0 * ell * (b - x)
                + 5.0 * np.sqrt(5.0) * (b - x) ** 2
            )
            - np.exp(np.sqrt(5.0) * (a - x) / ell)
            / (15.0 * ell)
            * (
                8.0 * np.sqrt(5.0) * ell**2
                + 25.0 * ell * (x - a)
                + 5.0 * np.sqrt(5.0) * (a - x) ** 2
            )
        )
    elif kernel.nu == 3.5:
        unnormalized_mean = (
            1.0
            / (105.0 * ell**2)
            * (
                96.0 * np.sqrt(7.0) * ell**3
                - np.exp(np.sqrt(7.0) * (x - b) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    - 231.0 * ell**2 * (x - b)
                    + 63.0 * np.sqrt(7.0) * ell * (x - b) ** 2
                    - 49.0 * (x - b) ** 3
                )
                - np.exp(np.sqrt(7.0) * (a - x) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    + 231.0 * ell**2 * (x - a)
                    + 63.0 * np.sqrt(7.0) * ell * (x - a) ** 2
                    + 49.0 * (x - a) ** 3
                )
            )
        )
    else:
        raise NotImplementedError(
            f"Kernel mean not available for kernel parameter nu={kernel.nu}"
        )
    return unnormalized_mean


def _kernel_variance_matern_1d_lebesgue(kernel: Matern, domain: Tuple):
    """Kernel variances for 1D Matern kernels.

    Note that these are for unnormalized Lebesgue measure.
    """
    (a, b) = domain
    r = b - a
    ell = kernel.lengthscale
    if kernel.nu == 0.5:
        unnormalized_variance = 2.0 * ell * (r + ell * (np.exp(-r / ell) - 1.0))
    elif kernel.nu == 1.5:
        c = np.sqrt(3.0) * r
        unnormalized_variance = (
            2.0 * ell / 3.0 * (2.0 * c - 3.0 * ell + np.exp(-c / ell) * (c + 3.0 * ell))
        )
    elif kernel.nu == 2.5:
        c = np.sqrt(5.0) * r
        unnormalized_variance = (
            1.0
            / 15.0
            * (
                2.0 * ell * (8.0 * c - 15.0 * ell)
                + 2.0
                * np.exp(-c / ell)
                * (
                    5.0 * a**2
                    - 10.0 * a * b
                    + 5.0 * b**2
                    + 7.0 * c * ell
                    + 15.0 * ell**2
                )
            )
        )
    elif kernel.nu == 3.5:
        c = np.sqrt(7.0) * r
        unnormalized_variance = (
            1.0
            / (105.0 * ell)
            * (
                2.0
                * np.exp(-c / ell)
                * (
                    7.0 * np.sqrt(7.0) * (b**3 - a**3)
                    + 84.0 * b**2 * ell
                    + 57.0 * np.sqrt(7.0) * b * ell**2
                    + 105.0 * ell**3
                    + 21.0 * a**2 * (np.sqrt(7.0) * b + 4.0 * ell)
                    - 3.0
                    * a
                    * (
                        7.0 * np.sqrt(7.0) * b**2
                        + 56.0 * b * ell
                        + 19.0 * np.sqrt(7.0) * ell**2
                    )
                )
                - 6.0 * ell**2 * (35.0 * ell - 16.0 * c)
            )
        )
    else:
        raise NotImplementedError(
            f"Kernel variance not available for kernel parameter nu={kernel.nu}"
        )
    return unnormalized_variance
