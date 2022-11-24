"""Special functions."""
from probnum.backend.typing import FloatLike

from .. import Array, asscalar
from ..._select_backend import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "gamma",
    "modified_bessel2",
    "ndtr",
    "ndtri",
]
__all__.sort()


def gamma(x: Array, /) -> Array:
    r"""Gamma function.

    Evaluates the gamma function defined as

    .. math::

        \Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\,dt

    for :math:`\text{Real}(x) > 0` and is extended to the rest of the complex plane by
    analytic continuation.

    The gamma function is often referred to as the generalized factorial since
    :math:`\Gamma(n+1) = n!` for natural numbers :math:`n`. More generally it satisfies
    the recurrence relation :math:`\Gamma(x + 1) = x \Gamma(x)` for complex :math:`x`,
    which, combined with the fact that :math:`\Gamma(1)=1`, implies the above.

    Parameters
    ----------
    x
        Argument(s) at which to evaluate the gamma function.
    """
    return _impl.gamma(x)


def modified_bessel2(x: Array, /, *, order: FloatLike) -> Array:
    """Modified Bessel function of the second kind of the given order.

    Parameters
    ----------
    x
        Argument(s) at which to evaluate the Bessel function.
    order
        Order of Bessel function.
    """
    return _impl.modified_bessel2(x, order)


def ndtr(x: Array, /) -> Array:
    r"""Normal distribution function.

    Returns the area under the Gaussian probability density function, integrated
    from minus infinity to x:

    .. math::

        \begin{align}
        \mathrm{ndtr}(x) =&
            \ \frac{1}{\sqrt{2 \pi}}\int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt \\
        =&\ \frac{1}{2} (1 + \mathrm{erf}(\frac{x}{\sqrt{2}})) \\
        =&\ \frac{1}{2} \mathrm{erfc}(\frac{x}{\sqrt{2}})
        \end{align}

    Parameters
    ----------
    x
        Argument(s) at which to evaluate the Normal distribution function.
    """
    return _impl.ndtr(x)


def ndtri(p: Array, /) -> Array:
    r"""The inverse of the CDF of the Normal distribution function.

    Returns `x` such that the area under the PDF from :math:`-\infty` to `x` is equal
    to `p`.

    Parameters
    ----------
    p
        Argument(s) at which to evaluate the inverse Normal distribution function.
    """
    return _impl.ndtri(p)
