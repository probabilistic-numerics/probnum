"""Toy integrands from Emukit."""

# The integrands are re-implementations from scratch based on the Emukit docs.
# There is no guarantee that they are identical to the Emukit implementations.

from typing import Optional

import numpy as np

from probnum.problems import QuadratureProblem
from probnum.quad.integration_measures import LebesgueMeasure
from probnum.typing import FloatLike


def hennig1d() -> QuadratureProblem:
    r"""The univariate hennig function integrated wrt the Lebesgue measure. [1]_

    The integrand is

    .. math::
        f(x) = e^{-x^2 -\sin^2(3x)}

    on the domain :math:`\Omega=[-3, 3]`.

    Returns
    -------
    quad_problem:
        The quadrature problem.

    References
    ----------
    .. [1] Emukit docs on `hennig1d <https://emukit.readthedocs.io/en/latest/api/emukit.test_functions.quadrature.html#emukit.test_functions.quadrature.hennig1D.hennig1D/>`__.

    """  # pylint: disable=line-too-long

    def fun(x):
        return np.exp(-x[:, 0] ** 2 - np.sin(3.0 * x[:, 0]) ** 2)

    measure = LebesgueMeasure(input_dim=1, domain=(-3, 3))
    return QuadratureProblem(fun=fun, measure=measure, solution=1.1433287777179366)


def hennig2d(c: Optional[np.ndarray] = None) -> QuadratureProblem:
    r"""The two-dimensional hennig function integrated wrt the Lebesgue measure. [1]_

    The integrand is

    .. math::
        f(x) = e^{-x^{\intercal}c x -\sin(3\|x\|^2)}

    on the domain :math:`\Omega=[-3, 3]^2`. Above, :math:`c` is the ``c`` parameter.

    Parameters
    ----------
    c
        A positive definite matrix of shape (2, 2). Defaults to [[1, .5], [.5, 1]].

    Returns
    -------
    quad_problem:
        The quadrature problem.

    References
    ----------
    .. [1] Emukit docs on `hennig2d <https://emukit.readthedocs.io/en/latest/api/emukit.test_functions.quadrature.html#emukit.test_functions.quadrature.hennig2D.hennig2D/>`__ .

    """  # pylint: disable=line-too-long

    solution = None
    if c is None:
        c = np.array([[1, 0.5], [0.5, 1]])
        solution = 3.525721820076955

    if c.shape != (2, 2):
        raise ValueError(f"'c' must be a (2, 2) array. Found shape is {c.shape}.")

    eigvals = np.linalg.eigvals(c)
    if np.any(eigvals <= 0):
        raise ValueError("'c' must be positive definite.")

    def fun(x):
        return np.exp(-np.sum((x @ c) * x, axis=1) - np.sin(3 * np.sum(x**2, axis=1)))

    measure = LebesgueMeasure(input_dim=2, domain=(-3, 3))
    return QuadratureProblem(fun=fun, measure=measure, solution=solution)


def sombrero2d(w: Optional[FloatLike] = None) -> QuadratureProblem:
    r"""The two-dimensional sombrero function integrated wrt the Lebesgue
    measure. [1]_

    The integrand is

    .. math::
        f(x) = \frac{\operatorname{sin}(\pi r w)}{\pi r w}

    on the domain :math:`\Omega=[-3, 3]^2`. Above, :math:`w` is the ``w``
    parameter and :math:`r=\|x\|` is the norm of the input vector :math:`x`.

    Parameters
    ----------
    w
        The positive frequency parameter. Defaults to 1.0.

    Returns
    -------
    quad_problem:
        The quadrature problem.

    References
    ----------
    .. [1] Emukit docs on `sombrero2d <https://emukit.readthedocs.io/en/latest/api/emukit.test_functions.quadrature.html#emukit.test_functions.quadrature.sombrero2D.sombrero2D/>`__ .

    """  # pylint: disable=line-too-long

    solution = None
    if w is None:
        w = 1.0
        solution = 0.85225026427372

    if w <= 0:
        raise ValueError(f"The 'w' parameter must be positive ({w}).")

    w = float(w)

    def fun(x):
        r_scaled = (np.pi * w) * np.sqrt((x * x).sum(axis=1))
        f = np.sin(r_scaled) / r_scaled
        f[np.isnan(f)] = 1.0
        return f

    measure = LebesgueMeasure(input_dim=2, domain=(-3, 3))
    return QuadratureProblem(fun=fun, measure=measure, solution=solution)


def circulargaussian2d(
    m: Optional[FloatLike] = None, v: Optional[FloatLike] = None
) -> QuadratureProblem:
    r"""The two-dimensional circular Gaussian integrated wrt the Lebesgue
    measure. [1]_

    The integrand is

    .. math::
        f(x) = (2\pi v)^{-\frac{1}{2}} r^2 e^{-\frac{(r - m)^2}{2 v}}

    on the domain :math:`\Omega=[-3, 3]^2`. Above, :math:`v` is the ``v``
    parameter, :math:`m` is the ``m`` parameter and :math:`r = \|x\|` is the
    norm of the input vector :math:`x`.

    Parameters
    ----------
    m
        The non-negative mean of the circular Gaussian in units of radius.
        Defaults to 0.0.
    v
        The positive variance of the circular Gaussian. Defaults to 1.0.

    Returns
    -------
    quad_problem:
        The quadrature problem.

    References
    ----------
    .. [1] Emukit docs on `circulargaussian2d <https://emukit.readthedocs.io/en/latest/api/emukit.test_functions.quadrature.html#emukit.test_functions.quadrature.circular_gaussian.circular_gaussian/>`__ .

    """  # pylint: disable=line-too-long

    _v = 1.0
    _m = 0.0

    solution = None
    if m is None and v is None:
        v, m = _v, _m
        solution = 4.853275495632483

    if m is None:
        m = _m
    if v is None:
        v = _v

    if m < 0:
        raise ValueError(f"'m' ({m}) must be non-negative.")

    if v <= 0:
        raise ValueError(f"'v' ({v}) must be positive.")

    m, v = float(m), float(v)

    def fun(x):
        r = np.linalg.norm(x, axis=1)
        rel_square_diff = (r - m) ** 2 / (2.0 * v)
        return r**2 * np.exp(-rel_square_diff) / np.sqrt(2.0 * np.pi * v)

    measure = LebesgueMeasure(input_dim=2, domain=(-3, 3))
    return QuadratureProblem(fun=fun, measure=measure, solution=solution)
