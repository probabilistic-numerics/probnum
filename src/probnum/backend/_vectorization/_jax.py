"""Vectorization in JAX."""

from typing import AbstractSet, Callable, Optional, Union

from jax import vmap  # pylint: disable=unused-import
import jax.numpy as jnp


def vectorize(
    fun: Callable,
    /,
    *,
    excluded: Optional[AbstractSet[Union[int, str]]] = None,
    signature: Optional[str] = None,
) -> Callable:
    return jnp.vectorize(
        fun,
        excluded=excluded if excluded is not None else set(),
        signature=signature,
    )
