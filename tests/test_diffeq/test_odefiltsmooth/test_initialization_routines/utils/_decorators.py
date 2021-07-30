"""JAX dependency handling for tests in test_initialize."""

import pytest

# Jax dependency handling
# pylint: disable=unused-import
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config

    config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True

except ImportError:
    JAX_AVAILABLE = False


# Pytest decorators to select tests for each case
only_if_jax_available = pytest.mark.skipif(not JAX_AVAILABLE, reason="requires jax")
