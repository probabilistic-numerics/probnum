"""Special functions in JAX."""

try:
    import jax.numpy as jnp
    from jax.scipy.special import ndtr, ndtri  # pylint: disable=unused-import
except ModuleNotFoundError:
    pass


def modified_bessel2(x: "jax.Array", order: "jax.Array") -> "jax.Array":
    return NotImplementedError


def gamma(x: "jax.Array", /) -> "jax.Array":
    raise NotImplementedError
