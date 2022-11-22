"""Logic functions on JAX arrays."""
try:
    from jax.numpy import (  # pylint: disable=unused-import
        all,
        any,
        equal,
        greater,
        greater_equal,
        less,
        less_equal,
        logical_and,
        logical_not,
        logical_or,
        logical_xor,
        not_equal,
    )
except ModuleNotFoundError:
    pass
