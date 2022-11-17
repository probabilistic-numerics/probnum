"""Vectorization of functions."""
from typing import AbstractSet, Any, Callable, Optional, Sequence, Union

from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl


__all__ = [
    "vectorize",
    "vmap",
]
__all__.sort()


def vectorize(
    fun: Callable,
    /,
    *,
    excluded: Optional[AbstractSet[Union[int, str]]] = None,
    signature: Optional[str] = None,
) -> Callable:
    """Vectorizing map, which creates a function which maps ``fun`` over array elements.

    Define a vectorized function which takes a nested sequence of arrays as inputs
    and returns a single array or a tuple of arrays. The vectorized function
    evaluates ``fun`` over successive tuples of the input arrays like the python map
    function, except it uses broadcasting rules.

    .. note::
        The :func:`~probnum.vectorize` function is primarily provided for convenience,
        not for performance. The implementation is essentially a for loop.

    Parameters
    ----------
    fun
        Function to be mapped
    excluded
        Set of strings or integers representing the positional or keyword arguments for
        which the function will not be vectorized. These will be passed directly to
        ``fun`` unmodified.
    signature
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``fun`` will be called
        with (and expected to return) arrays with shapes given by the size of
        corresponding core dimensions. By default, ``fun`` is assumed to take scalars as
        input and output.
    """
    return _impl.vectorize(fun, excluded=excluded, signature=signature)


def vmap(
    fun: Callable,
    /,
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
