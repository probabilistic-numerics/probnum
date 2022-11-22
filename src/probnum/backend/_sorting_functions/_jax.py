"""Sorting functions for JAX arrays."""

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import isnan  # pylint: disable=redefined-builtin, unused-import
except ModuleNotFoundError:
    pass


def sort(
    x: "jax.Array",
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> "jax.Array":
    kind = "quicksort"
    if stable:
        kind = "stable"

    sorted_array = jnp.sort(x, axis=axis, kind=kind)

    if descending:
        return jnp.flip(sorted_array, axis=axis)

    return sorted_array


def argsort(
    x: "jax.Array",
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> "jax.Array":
    kind = "quicksort"
    if stable:
        kind = "stable"

    sort_idx = jnp.argsort(x, axis=axis, kind=kind)

    if descending:
        return jnp.flip(sort_idx, axis=axis)

    return sort_idx
