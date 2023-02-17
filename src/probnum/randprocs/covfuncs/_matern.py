"""Matérn covariance function."""

import fractions
import functools
from typing import Optional, Tuple

import numpy as np
import scipy.special

from probnum.typing import ArrayLike, ScalarLike, ScalarType, ShapeLike
import probnum.utils as _utils

from ._covariance_function import CovarianceFunction, IsotropicMixin


class Matern(CovarianceFunction, IsotropicMixin):
    r"""Matérn covariance function.

    Covariance function defined by

    .. math::
        :nowrap:

        \begin{equation}
            k_\nu(x_0, x_1)
            :=
            \frac{2^{1 - \nu}}{\Gamma(\nu)}
            \left( \sqrt{2 \nu} \lVert x_0 - x_1 \rVert_{\Lambda^{-1}} \right)^\nu
            K_\nu \left( \sqrt{2 \nu} \lVert x_0 - x_1 \rVert_{\Lambda^{-1}} \right),
        \end{equation}

    where :math:`K_\nu` is a modified Bessel function of the second kind and

    .. math::
        \lVert x_0 - x_1 \rVert_{\Lambda^{-1}}^2
        :=
        \sum_{i = 1}^d \frac{(x_{0,i} - x_{1,i})^2}{l_i}.

    The Matérn covariance function generalizes the :class:`~probnum.randprocs.covfuncs.\
    ExpQuad` covariance function via its additional parameter :math:`\nu` controlling
    the smoothness of the functions in the associated RKHS.
    For :math:`\nu \rightarrow \infty`, the Matérn covariance function converges to the
    :class:`~probnum.randprocs.covfuncs.ExpQuad` covariance function.
    A Gaussian process with Matérn covariance function is :math:`\lceil \nu \rceil - 1`
    times differentiable.

    If :math:`\nu` is a half-integer, i.e. :math:`\nu = p + \frac{1}{2}` for some
    nonnegative integer :math:`p`, then the expression for the covariance function
    simplifies to a product of an exponential and a polynomial

    .. math::
        :nowrap:

        \begin{equation}
            k_{\nu = p + \frac{1}{2}}(x_0, x_1)
            =
            \exp \left( -\sqrt{2 \nu} \lVert x_0 - x_1 \rVert_{\Lambda^{-1}} \right)
            \frac{p!}{(2p)!}
            \sum_{i = 0}^p \frac{(p + i)!}{i!(p - i)!} 2^{p - i}
            \left( \sqrt{2 \nu} \lVert x_0 - x_1 \rVert_{\Lambda^{-1}} \right)^{p - i}.
        \end{equation}

    Parameters
    ----------
    input_shape
        Shape of the covariance function's inputs.
    nu
        Hyperparameter :math:`\nu` controlling differentiability.
    lengthscales
        Lengthscales :math:`l_i` along the different input dimensions of the covariance
        function.
        Describes the input scales on which the process varies.
        The lengthscales will be broadcast to the input shape of the covariance
        function.

    See Also
    --------
    ExpQuad : Exponentiated Quadratic covariance function.
    ProductMatern : Tensor product of 1D Matérn covariance functions.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import Matern
    >>> K = Matern((), nu=2.5, lengthscales=0.1)
    >>> xs = np.linspace(0, 1, 3)
    >>> K.matrix(xs)
    array([[1.00000000e+00, 7.50933789e-04, 3.69569622e-08],
           [7.50933789e-04, 1.00000000e+00, 7.50933789e-04],
           [3.69569622e-08, 7.50933789e-04, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        nu: ScalarLike = 1.5,
        *,
        lengthscales: Optional[ArrayLike] = None,
    ):
        super().__init__(input_shape=input_shape)

        # Smoothness parameter ν
        nu = _utils.as_numpy_scalar(nu)

        if nu <= 0:
            raise ValueError(f"Hyperparameter nu={nu} must be positive.")

        self._nu = nu

        # Input lengthscales
        lengthscales = np.asarray(
            lengthscales if lengthscales is not None else 1.0,
            dtype=np.double,
        )

        if np.any(lengthscales <= 0):
            raise ValueError(f"All lengthscales l={lengthscales} must be positive.")

        np.broadcast_to(  # Check if the lengthscales broadcast to the input dimension
            lengthscales, self.input_shape
        )

        self._lengthscales = lengthscales

    @property
    def nu(self) -> ScalarType:
        r"""Smoothness parameter :math:`\nu`."""
        return self._nu

    @functools.cached_property
    def p(self) -> Optional[int]:
        r"""Degree :math:`p` of the polynomial part of a Matérn covariance function with
        half-integer smoothness parameter :math:`\nu = p + \frac{1}{2}`. If :math:`\nu`
        is not a half-integer, this is set to :data:`None`.

        Sample paths of a Gaussian process with this covariance function are
        :math:`p`-times continuously differentiable."""
        nu_minus_half = self._nu - 0.5

        # Half-integer values can be represented exactly in IEEE 754 floating point
        # numbers
        if nu_minus_half == int(nu_minus_half):
            return int(nu_minus_half)

        return None

    @property
    def is_half_integer(self) -> bool:
        r"""Indicates whether :math:`\nu` is a half-integer."""
        return self.p is not None

    @property
    def lengthscales(self) -> np.ndarray:
        r"""Input lengthscales along the different input dimensions."""
        return self._lengthscales

    @property
    def lengthscale(self) -> np.ndarray:  # pylint: disable=missing-raises-doc
        """Deprecated."""
        if self._lengthscales.shape == ():
            return self._lengthscales

        raise ValueError("There is more than one lengthscale.")

    @functools.cached_property
    def _scale_factors(self) -> np.ndarray:
        return np.sqrt(2 * self._nu) / self._lengthscales

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._scale_factors
        )

        if self.is_half_integer:
            # Evaluate the polynomial part using Horner's method
            coeffs = Matern._half_integer_coefficients_floating(self.p)

            res = coeffs[self.p]

            for i in range(self.p - 1, -1, -1):
                res *= scaled_dists
                res += coeffs[i]

            # Exponential part
            res *= np.exp(-scaled_dists)

            return res

        return _matern_bessel(scaled_dists, nu=self._nu)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def half_integer_coefficients(p: int) -> Tuple[fractions.Fraction]:
        r"""Computes the rational coefficients :math:`c_i` of the polynomial part of a
        Matérn covariance function with half-integer smoothness parameter :math:`\nu = \
        p + \frac{1}{2}`.

        We leverage the recursion

        .. math::
            :nowrap:

            \begin{align}
                c_{p - i}
                & := \frac{p!}{(2p)!} \frac{(p + i)!}{i!(p - i)!} 2^{p - i} \\
                & = \frac{2 (i + 1)}{(p + i + 1) (p - i)} c_{p - (i + 1)},
            \end{align}

        where :math:`c_0 = c_{p - p} = 1`.

        Parameters
        ----------
        p:
            Degree :math:`p` of the polynomial part.

        Returns
        -------
        coefficients:
            A tuple containing the exact rational coefficients of the polynomial part,
            where the entry at index :math:`i` contains the coefficient corresponding to
            the monomial with degree :math:`i`.
        """
        coeffs = [fractions.Fraction(1, 1)]

        for i in range(p - 1, -1, -1):
            coeffs.append(coeffs[-1] * 2 * (i + 1) / (p + i + 1) / (p - i))

        return tuple(coeffs)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _half_integer_coefficients_floating(p: int) -> np.ndarray:
        return np.asarray(Matern.half_integer_coefficients(p), dtype=np.float64)


def _matern_bessel(scaled_dists: np.ndarray, nu: ScalarType) -> np.ndarray:
    # The modified Bessel function K_nu is not defined for z=0
    scaled_dists = np.maximum(scaled_dists, np.finfo(scaled_dists.dtype).eps)

    return (
        2 ** (1.0 - nu)
        / scipy.special.gamma(nu)
        * scaled_dists**nu
        * scipy.special.kv(nu, scaled_dists)
    )
