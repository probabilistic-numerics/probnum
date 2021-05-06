import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo

# Jax dependency handling
# pylint: disable=unused-import
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config

    config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True

    IVPs = [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]


except ImportError:
    JAX_AVAILABLE = False
    IVPs = []


# Pytest decorators to select tests for each case
only_if_jax_available = pytest.mark.skipif(not JAX_AVAILABLE, reason="requires jax")
only_if_jax_is_not_available = pytest.mark.skipif(
    JAX_AVAILABLE,
    reason="Imports will be successful, thus catching the ImportError will fail",
)

# Tests for when JAX is available


@pytest.mark.parametrize("ivp_jax", IVPs)
@only_if_jax_available
def test_f(ivp_jax):
    ivp_jax.f(ivp_jax.t0, ivp_jax.y0)


@pytest.mark.parametrize("ivp_jax", IVPs)
@only_if_jax_available
def test_df(ivp_jax):
    ivp_jax.df(ivp_jax.t0, ivp_jax.y0)


@pytest.mark.parametrize("ivp_jax", IVPs)
@only_if_jax_available
def test_ddf(ivp_jax):
    ivp_jax.ddf(ivp_jax.t0, ivp_jax.y0)


# Tests for when JAX is not available


@only_if_jax_is_not_available
def test_threebody():
    with pytest.raises(ImportError):
        diffeq_zoo.threebody_jax()


@only_if_jax_is_not_available
def test_vanderpol():
    with pytest.raises(ImportError):
        diffeq_zoo.vanderpol_jax()
