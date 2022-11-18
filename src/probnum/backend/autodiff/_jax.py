"""(Automatic) Differentiation in JAX."""
try:
    from jax import (  # pylint: disable=unused-import
        grad,
        hessian,
        jacfwd,
        jacrev,
        value_and_grad,
    )
except ModuleNotFoundError:
    pass
