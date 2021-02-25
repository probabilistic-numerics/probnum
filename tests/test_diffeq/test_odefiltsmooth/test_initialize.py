import numpy as np
import pytest

import probnum.diffeq as pnde
import probnum.random_variables as pnrv

from ._known_initial_derivatives import LV_INITS


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

    # Extract the relevant values
    expected = np.hstack((LV_INITS[0:6], LV_INITS[15:21]))

    # The higher derivatives will have absolute difference ~8%
    # if things work out correctly
    np.testing.assert_allclose(received, expected, rtol=0.25)
