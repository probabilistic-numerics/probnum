"""Elementwise functions."""

from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]


def abs(x: Array, /) -> Array:
    """Calculates the absolute value for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the absolute value of each element in ``x``.
    """
    return _impl.abs(x)


def acos(x: Array, /) -> Array:
    """Calculates an approximation of the principal value of the inverse cosine, having
    domain ``[-1, +1]`` and codomain ``[+0, +π]``, for each element ``x_i`` of the input
    array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the inverse cosine of each element in ``x``.
    """
    return _impl.acos(x)


def acosh(x: Array, /) -> Array:
    """Calculates an approximation to the inverse hyperbolic cosine, having domain
    ``[+1, infinity]`` and codomain ``[+0, infinity]``, for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.
        Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the inverse hyperbolic cosine of each element in ``x``.
    """
    return _impl.acosh(x)


def add(x1: Array, x2: Array, /) -> Array:
    """Calculates the sum for each element ``x1_i`` of the input array ``x1`` with the
    respective element ``x2_i`` of the input array ``x2``.

    .. note::

       Floating-point addition is a commutative operation, but not always associative.


    Parameters
    ----------
    x1
        first input array.
    x2
        second input array. Must be compatible with ``x1`` (according to the rules of
        broadcasting).

    Returns
    -------
    out
        an array containing the element-wise sums.
    """
    return _impl.add(x1, x2)


def asin(x: Array, /) -> Array:
    """Calculates an approximation of the principal value of the inverse sine, having
    domain ``[-1, +1]`` and codomain ``[-π/2, +π/2]`` for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the inverse sine of each element in ``x``.
    """
    return _impl.asin(x)


def asinh(x: Array, /) -> Array:
    """Calculates an approximation to the inverse hyperbolic sine, having domain
    ``[-infinity, infinity]`` and codomain ``[-infinity, infinity]``, for each element
    ``x_i`` in the input array ``x``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.

    Returns
    -------
    out
        an array containing the inverse hyperbolic sine of each element in ``x``.
    """
    return _impl.asinh(x)


def atan(x: Array, /) -> Array:
    """Calculates an approximation of the principal value of the inverse tangent, having
    domain ``[-infinity, infinity]`` and codomain ``[-π/2, +π/2]``, for each element
    ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the inverse tangent of each element in ``x``.
    """
    return _impl.atan(x)


def atan2(x1: Array, x2: Array, /) -> Array:
    """Calculates an approximation of the inverse tangent of the quotient ``x1/x2``,
    having domain ``[-infinity, infinity] x [-infinity, infinity]`` and codomain ``[-π,

    +π]``, for each pair of elements ``(x1_i, x2_i)`` of the input arrays ``x1`` and
    ``x2``, respectively.

    The mathematical signs of ``x1_i`` and ``x2_i`` determine the quadrant of each
    element-wise result. The quadrant (i.e., branch) is chosen such that each
    element-wise result is the signed angle in radians between the ray ending at the
    origin and passing through the point ``(1,0)`` and the ray ending at the origin and
    passing through the point ``(x2_i, x1_i)``.


    Parameters
    ----------
    x1
        input array corresponding to the y-coordinates.
    x2
        input array corresponding to the x-coordinates.

    Returns
    -------
    out
        an array containing the inverse tangent of the quotient ``x1/x2``.
    """
    return _impl.atan2(x1, x2)


def atanh(x: Array, /) -> Array:
    """Calculates an approximation to the inverse hyperbolic tangent, having domain
    ``[-1, +1]`` and codomain ``[-infinity, infinity]``, for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array whose elements each represent the area of a hyperbolic sector.
    Returns
    -------
    out
        an array containing the inverse hyperbolic tangent of each element in ``x``.
    """
    return _impl.atanh(x)


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """Computes the bitwise AND of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have an integer or
        boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_and(x1, x2)


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the left by
    appending ``x2_i`` (i.e., the respective element in the input array ``x2``) zeros to
    the right of ``x1_i``.

    Parameters
    ----------
    x1
        first input array. Should have an integer data type.
    x2
        second input array. Must be compatible with ``x1``. Should have an integer data
        type. Each element must be greater than or equal to ``0``.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_left_shift(x1, x2)


def bitwise_invert(x: Array, /) -> Array:
    """Inverts (flips) each bit for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have an integer or boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_invert(x)


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """Computes the bitwise OR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have an integer or
        boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_or(x1, x2)


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the right
    according to the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       This operation must be an arithmetic shift (i.e., sign-propagating) and thus
       equivalent to floor division by a power of two.

    Parameters
    ----------
    x1
        first input array. Should have an integer data type.
    x2
        second input array. Must be compatible with ``x1``. Should have an integer data
        type. Each element must be greater than or equal to ``0``.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_right_shift(x1, x2)


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    """Computes the bitwise XOR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have an integer or boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have an integer or
        boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.bitwise_xor(x1, x2)


def ceil(x: Array, /) -> Array:
    """Rounds each element ``x_i`` of the input array ``x`` to the smallest (i.e.,
    closest to ``-infinity``) integer-valued number that is not less than ``x_i``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the rounded result for each element in ``x``.
    """
    return _impl.ceil(x)


def conj(x: Array, /) -> Array:
    """Returns the complex conjugate for each element ``x_i`` of the input array ``x``.

    For complex numbers of the form

    .. math::
       a + bj

    the complex conjugate is defined as

    .. math::
       a - bj

    Hence, the returned complex conjugates must be computed by negating the imaginary
    component of each element ``x_i``.

    Parameters
    ----------
    x
        input array. Should have a complex-floating point data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.conj(x)


def cos(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the cosine for each element ``x_i`` of the input
    array ``x``.

    Each element ``x_i`` is assumed to be expressed in radians.


    For complex floating-point operands, special cases must be handled as if the
    operation is implemented as ``cosh(x*1j)``.

    .. note::
       The cosine is an entire function on the complex plane and has no branch cuts.

    .. note::
       For complex arguments, the mathematical definition of cosine is

       .. math::
          \begin{align} \operatorname{cos}(x) &= \sum_{n=0}^\infty \frac{(-1)^n}{(2n)!} x^{2n} \\ &= \frac{e^{jx} + e^{-jx}}{2} \\ &= \operatorname{cosh}(jx) \end{align}

       where :math:`\operatorname{cosh}` is the hyperbolic cosine.

    Parameters
    ----------
    x
        input array whose elements are each expressed in radians. Should have a
        floating-point data type.

    Returns
    -------
    out
        an array containing the cosine of each element in ``x``.
    """
    return _impl.cos(x)


def cosh(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the hyperbolic cosine for each element ``x_i`` in the
     input array ``x``.

    The mathematical definition of the hyperbolic cosine is

    .. math::
       \operatorname{cosh}(x) = \frac{e^x + e^{-x}}{2}

    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a
        floating-point data type.

    Returns
    -------
    out
        an array containing the hyperbolic cosine of each element in ``x``.
    """
    return _impl.cosh(x)


def divide(x1: Array, x2: Array, /) -> Array:
    """Calculates the division for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        dividend input array. Should have a real-valued data type.
    x2
        divisor input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.divide(x1, x2)


def equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with ``x1``. May have any data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.equal(x1, x2)


def exp(x: Array, /) -> Array:
    """Calculates an approximation to the exponential function for each element ``x_i``
    of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the
    base of the natural logarithm).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated exponential function result for each element
        in ``x``.
    """
    return _impl.exp(x)


def expm1(x: Array, /) -> Array:
    """Calculates an approximation to ``exp(x)-1``, having domain ``[-infinity,
    infinity]`` and codomain ``[-1, infinity]``, for each element ``x_i`` of the input
    array ``x``.

    .. note::

       The purpose of this function is to calculate ``exp(x) - 1.0`` more accurately
       when `x` is close to zero.


    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.expm1(x)


def floor(x: Array, /) -> Array:
    """Rounds each element ``x_i`` of the input array ``x`` to the greatest (i.e.,
    closest to ``infinity``) integer-valued number that is not greater than ``x_i``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the rounded result for each element in ``x``.
    """
    return _impl.floor(x)


def floor_divide(x1: Array, x2: Array, /) -> Array:
    r"""
    Rounds the result of dividing each element ``x1_i`` of the input array ``x1`` by the
     respective element ``x2_i`` of the input array ``x2`` to the greatest (i.e.,
     closest to `infinity`) integer-value number that is not greater than the division
     result.

    Parameters
    ----------
    x1
        dividend input array. Should have a real-valued data type.
    x2
        divisor input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.floor_divide(x)


def greater(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.greater(x1, x2)


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.greater_equal(x1, x2)


def imag(x: Array, /) -> Array:
    """Returns the imaginary component of a complex number for each element ``x_i`` of
    the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a complex floating-point data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.imag(x)


def isfinite(x: Array, /) -> Array:
    """Tests each element ``x_i`` of the input array ``x`` to determine if finite (i.e.,
    not ``NaN`` and not equal to positive or negative infinity).

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
         finite and ``False`` otherwise.
    """
    return _impl.isfinite(x)


def isinf(x: Array, /) -> Array:
    """Tests each element ``x_i`` of the input array ``x`` to determine if equal to
    positive or negative infinity.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i``
        is either positive or negative infinity and ``False`` otherwise.
    """
    return _impl.isinf(x)


def isnan(x: Array, /) -> Array:
    """Tests each element ``x_i`` of the input array ``x`` to determine whether the
    element is ``NaN``.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.

    Returns
    -------
    out
        An array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
        ``NaN`` and ``False`` otherwise. The returned array should have a data type of
        ``bool``.
    """
    return _impl.isnan(x)


def less(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.less(x1, x2)


def less_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i <= x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.less_equal(x1, x2)


def log(x: Array, /) -> Array:
    """Calculates an approximation to the natural (base ``e``) logarithm, having domain
    ``[0, infinity]`` and codomain ``[-infinity, infinity]``, for each element ``x_i``
    of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``infinity``, the result is ``infinity``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated natural logarithm for each element in ``x``.
    """
    return _impl.log(x)


def log1p(x: Array, /) -> Array:
    """Calculates an approximation to ``log(1+x)``, where ``log`` refers to the natural
    (base ``e``) logarithm, having domain ``[-1, infinity]`` and codomain ``[-infinity,
    infinity]``, for each element ``x_i`` of the input array ``x``.

    .. note::
       The purpose of this function is to calculate ``log(1+x)`` more accurately
       when `x` is close to zero.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.log1p(x)


def log2(x: Array, /) -> Array:
    """Calculates an approximation to the base ``2`` logarithm, having domain ``[0,
    infinity]`` and codomain ``[-infinity, infinity]``, for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated base ``2`` logarithm for each element in
        ``x``.
    """
    return _impl.log2(x)


def log10(x: Array, /) -> Array:
    """Calculates an approximation to the base ``10`` logarithm, having domain ``[0,
    infinity]`` and codomain ``[-infinity, infinity]``, for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the evaluated base ``10`` logarithm for each element in
        ``x``.
    """
    return _impl.log10(x)


def logaddexp(x1: Array, x2: Array, /) -> Array:
    """Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))``
    for each element ``x1_i`` of the input array ``x1`` with the respective element
    ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued floating-point data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        floating-point data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logaddexp(x1, x2)


def logical_and(x1: Array, x2: Array, /) -> Array:
    """Computes the logical AND for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data
        type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_and(x1, x2)


def logical_not(x: Array, /) -> Array:
    """Computes the logical NOT for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_not(x)


def logical_or(x1: Array, x2: Array, /) -> Array:
    """Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_or(x1, x2)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """Computes the logical XOR for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a boolean data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a boolean data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.logical_xor(x)


def multiply(x1: Array, x2: Array, /) -> Array:
    """Calculates the product for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise products.
    """
    return _impl.multiply(x1, x2)


def negative(x: Array, /) -> Array:
    """
    Computes the numerical negative of each element ``x_i`` (i.e., ``y_i = -x_i``) of
    the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.negative(x)


def not_equal(x1: Array, x2: Array, /) -> Array:
    """Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the
    input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. May have any data type.
    x2
        second input array. Must be compatible with ``x1``.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.not_equal(x1, x2)


def positive(x: Array, /) -> Array:
    """
    Computes the numerical positive of each element ``x_i`` (i.e., ``y_i = +x_i``) of
    the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.positive(x)


def pow(x1: Array, x2: Array, /) -> Array:
    """Calculates an approximation of exponentiation by raising each element ``x1_i``
    (the base) of the input array ``x1`` to the power of ``x2_i`` (the exponent), where
    ``x2_i`` is the corresponding element of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array whose elements correspond to the exponentiation base. Should
        have a real-valued data type.
    x2
        second input array whose elements correspond to the exponentiation exponent.
        Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.pow(x1, x2)


def real(x: Array, /) -> Array:
    """Returns the real component of a complex number for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a complex floating-point data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.real(x)


def remainder(x1: Array, x2: Array, /) -> Array:
    """Returns the remainder of division for each element ``x1_i`` of the input array
    ``x1`` and the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       This function is equivalent to the Python modulus operator ``x1_i % x2_i``.

    Parameters
    ----------
    x1
        dividend input array. Should have a real-valued data type.
    x2
        divisor input array. Must be compatible with ``x1``. Should have a real-valued
        data type.

    Returns
    -------
    out
        an array containing the element-wise results.
    """
    return _impl.remainder(x)


def round(x: Array, /) -> Array:
    """Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-
    valued number.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.

    Returns
    -------
    out
        an array containing the rounded result for each element in ``x``.
    """
    return _impl.round(x)


def sign(x: Array, /) -> Array:
    """Returns an indication of the sign of a number for each element ``x_i`` of the
    input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.sign(x)


def sin(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the sine for each element ``x_i`` of the input array
    ``x``.

    Each element ``x_i`` is assumed to be expressed in radians.

    For complex floating-point operands, special cases must be handled as if the
    operation is implemented as ``-1j * sinh(x*1j)``.

    Parameters
    ----------
    x
        input array whose elements are each expressed in radians. Should have a floating-point data type.

    Returns
    -------
    out
        an array containing the sine of each element in ``x``.
    """
    return _impl.sin(x)


def sinh(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the hyperbolic sine for each element ``x_i`` of the
    input array ``x``.

    The mathematical definition of the hyperbolic sine is

    .. math::
       \operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}

    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

    Returns
    -------
    out
        an array containing the hyperbolic sine of each element in ``x``.
    """
    return _impl.sinh(x)


def square(x: Array, /) -> Array:
    """
    Squares (``x_i * x_i``) each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the evaluated result for each element in ``x``.
    """
    return _impl.square(x)


def sqrt(x: Array, /) -> Array:
    """Calculates the square root, having domain ``[0, infinity]`` and codomain ``[0,

    infinity]``, for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x
        input array. Should have a real-valued floating-point data type.

    Returns
    -------
    out
        an array containing the square root of each element in ``x``.
    """
    return _impl.sqrt(x)


def subtract(x1: Array, x2: Array, /) -> Array:
    """Calculates the difference for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1
        first input array. Should have a real-valued data type.
    x2
        second input array. Must be compatible with ``x1``. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the element-wise differences.
    """
    return _impl.subtract(x1, x2)


def tan(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the tangent for each element ``x_i`` of the input
    array ``x``.

    Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x
        input array whose elements are expressed in radians. Should have a floating-point data type.

    Returns
    -------
    out
        an array containing the tangent of each element in ``x``.
    """
    return _impl.tan(x)


def tanh(x: Array, /) -> Array:
    r"""
    Calculates an approximation to the hyperbolic tangent for each element ``x_i`` of
    the input array ``x``.

    Parameters
    ----------
    x
        input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

    Returns
    -------
    out
        an array containing the hyperbolic tangent of each element in ``x``.
    """
    return _impl.tanh(x)


def trunc(x: Array, /) -> Array:
    """Rounds each element ``x_i`` of the input array ``x`` to the integer-valued number
    that is closest to but no greater than ``x_i``.

    Parameters
    ----------
    x
        input array. Should have a real-valued data type.

    Returns
    -------
    out
        an array containing the rounded result for each element in ``x``.
    """
    return _impl.trunc(x)
