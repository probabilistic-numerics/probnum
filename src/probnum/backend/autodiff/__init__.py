"""(Automatic) Differentiation."""

from typing import Any, Callable, Sequence, Union

from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl


__all__ = [
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "vmap",
]
__all__.sort()


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Creates a function that evaluates the gradient of ``fun``.

    Parameters
    ----------
    fun
        Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers.
        Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It
        should return a scalar (which includes arrays with shape ``()`` but not
        arrays with shape ``(1,)`` etc.)
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data.

    Returns
    -------
    grad_fun
        A function with the same arguments as ``fun``, that evaluates the gradient
        of ``fun``. If ``argnums`` is an integer then the gradient has the same
        shape and type as the positional argument indicated by that integer. If
        argnums is a tuple of integers, the gradient is a tuple of values with the
        same shapes and types as the corresponding arguments.

    Examples
    --------
    >>> from probnum.backend.autodiff import grad
    >>> grad_sin = grad(backend.sin)
    >>> grad_sin(backend.pi)
    -1.0
    """
    return _impl.grad(fun=fun, argnums=argnums, has_aux=has_aux)


def hessian(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Hessian of ``fun`` as a dense array.

    Parameters
    ----------
    fun
        Function whose Hessian is to be computed.  Its arguments at positions
        specified by ``argnums`` should be arrays, scalars, or standard Python
        containers thereof. It should return arrays, scalars, or standard Python
        containers thereof.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    hessian
        A function with the same arguments as ``fun``, that evaluates the Hessian of
        ``fun``.

    >>> from probnum.backend.autodiff import hessian
    >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
    >>> hessian(g)(backend.asarray([1., 2.])))
    [[   6.   -2.]
    [  -2. -480.]]
    """
    return _impl.hessian(fun=fun, argnums=argnums, has_aux=has_aux)


def jacfwd(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Parameters
    ----------
    fun
        Function whose Jacobian is to be computed.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    jacfun
      A function with the same arguments as ``fun``, that evaluates the Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    return _impl.jacfwd(fun, argnums, has_aux=has_aux)


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Parameters
    ----------
    fun
        Function whose Jacobian is to be computed.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    jacfun
      A function with the same arguments as ``fun``, that evaluates the Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    return _impl.jacrev(fun, argnums, has_aux=has_aux)


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
) -> Callable:
    """Vectorizing map, which creates a function which maps ``fun`` over argument axes.

    Parameters
    ----------
    fun
        Function to be mapped over additional axes.
    in_axes
        Input array axes to map over.

        If each positional argument to ``fun`` is an array, then ``in_axes`` can
        be an integer, a None, or a tuple of integers and Nones with length equal
        to the number of positional arguments to ``fun``. An integer or ``None``
        indicates which array axis to map over for all arguments (with ``None``
        indicating not to map any axis), and a tuple indicates which axis to map
        for each corresponding positional argument. Axis integers must be in the
        range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
        axes of the corresponding input array.
    out_axes
        Where the mapped axis should appear in the output.

        All outputs with a mapped axis must have a non-None
        ``out_axes`` specification. Axis integers must be in the range ``[-ndim,
        ndim)`` for each output array, where ``ndim`` is the number of dimensions
        (axes) of the array returned by the :func:`vmap`-ed function, which is one
        more than the number of dimensions (axes) of the corresponding array
        returned by ``fun``.

    Returns
    -------
    vfun
        Batched/vectorized version of ``fun`` with arguments that correspond to
        those of ``fun``, but with extra array axes at positions indicated by
        ``in_axes``, and a return value that corresponds to that of ``fun``, but
        with extra array axes at positions indicated by ``out_axes``.
    """
    return _impl.vmap(fun, in_axes, out_axes)
