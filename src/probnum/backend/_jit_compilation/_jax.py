"""Just-In-Time Compilation in JAX."""
from typing import Callable, Iterable, Union

import jax
from jax import jit  # pylint: disable=unused-import


def jit_method(
    method: Callable,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    _static_argnums = (0,)

    if static_argnums is not None:
        _static_argnums += tuple(argnum + 1 for argnum in static_argnums)

    return jax.jit(
        method, static_argnums=_static_argnums, static_argnames=static_argnames
    )
