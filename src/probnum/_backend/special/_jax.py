import jax


def gamma(x):
    return jax.lax.exp(jax.lax.lgamma(x))


def kv(x):
    raise NotImplementedError()
