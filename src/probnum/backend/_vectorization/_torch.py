"""Vectorization in PyTorch."""
from typing import AbstractSet, Any, Callable, Optional, Sequence, Union

import functorch


def vectorize(
    fun: Callable,
    /,
    *,
    excluded: Optional[AbstractSet[Union[int, str]]] = None,
    signature: Optional[str] = None,
) -> Callable:
    raise NotImplementedError()


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
) -> Callable:
    return functorch.vmap(fun, in_dims=in_axes, out_dims=out_axes)
