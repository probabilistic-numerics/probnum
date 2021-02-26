import numpy as np
import pytest

import probnum.diffeq as pnde
import probnum.problems as pnpr
import probnum.problems.zoo.diffeq as diffeq_zoo
import probnum.random_variables as pnrv

from ._known_initial_derivatives import LV_INITS, THREEBODY_INITS

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


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.skipif(not JAX_AVAILABLE, reason="requires jax")
def test_compute_all_derivatives_via_taylormode(order):
    """Test asserts that the examples in diffeq-zoo are compatible with
    `compute_all_derivatives`, which happens if they are implemented in jax, and jax is
    available in the current environment."""
    ivp = diffeq_zoo.threebody_jax()
    y0_all, errors = pnde.compute_all_derivatives_via_taylormode(
        ivp.f, ivp.y0, ivp.t0, order=order
    )
    np.testing.assert_allclose(errors, 0.0)
    print(y0_all)
    print(THREEBODY_INITS)
    assert False


@pytest.fixture
def lv():
    y0 = pnrv.Constant(np.array([20.0, 20.0]))

    # tmax is ignored anyway
    return pnde.lotkavolterra([0.0, None], y0)


def test_initialize_with_rk(lv):
    """Make sure that the values are close(ish) to the truth."""
    received, error = pnde.compute_all_derivatives_via_rk(
        lv.rhs, lv.initrv.mean, lv.t0, df=lv.jacobian, h0=1e-1, order=5, method="RK45"
    )
    ode_dim = 2
    expected = np.hstack(())
    # Extract the relevant values
    expected = np.hstack((LV_INITS[0:6], LV_INITS[15:21]))

    # The higher derivatives will have absolute difference ~8%
    # if things work out correctly
    np.testing.assert_allclose(received, expected, rtol=0.25)
