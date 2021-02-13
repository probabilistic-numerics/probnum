import pytest

import probnum.diffeq as pnd
import probnum.problems as pnprob
import probnum.problems.zoo.diffeq as diffeq_zoo

try:
    import jax
    import jax.numpy as jnp
    from jax.config import config

    config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True

    IVPs = [diffeq_zoo.threebody_jax(), diffeq_zoo.vanderpol_jax()]
except ImportError:
    JAX_AVAILABLE = False
    IVPs = None

only_if_jax_available = pytest.mark.skipif(not JAX_AVAILABLE, reason="requires jax")
only_if_jax_is_not_available = pytest.mark.skipif(JAX_AVAILABLE)


# Tests for when JAX is available


@only_if_jax_available
@pytest.mark.parametrize("ivp_jax", IVPs)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_compute_all_derivatives_terminates_successfully(ivp_jax, order):
    """Test asserts that the examples in diffeq-zoo are compatible with
    `compute_all_derivatives`, which happens if they are implemented in jax, and jax is
    available in the current environment."""

    ivp = pnd.compute_all_derivatives(ivp_jax, order=order)
    assert isinstance(ivp, pnprob.InitialValueProblem)


@only_if_jax_available
@pytest.mark.parametrize("ivp_jax", IVPs)
def test_f(ivp_jax):
    ivp_jax.f(ivp_jax.t0, ivp_jax.y0)


@only_if_jax_available
@pytest.mark.parametrize("ivp_jax", IVPs)
def test_df(ivp_jax):
    ivp_jax.df(ivp_jax.t0, ivp_jax.y0)


@only_if_jax_available
@pytest.mark.parametrize("ivp_jax", IVPs)
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
