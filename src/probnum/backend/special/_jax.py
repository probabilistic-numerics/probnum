"""Special functions in JAX."""

try:
    from jax.scipy.special import ndtr, ndtri  # pylint: disable=unused-import
except ModuleNotFoundError:
    pass


def gamma(*args, **kwargs):
    raise NotImplementedError()


def kv(*args, **kwargs):
    raise NotImplementedError()
