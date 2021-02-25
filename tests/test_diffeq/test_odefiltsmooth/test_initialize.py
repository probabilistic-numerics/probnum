import numpy as np
import pytest

import probnum.diffeq as pnde
from probnum.problems.zoo.diffeq import threebody

from ._known_initial_derivatives import THREEBODY_INITS


@pytest.fixture
def ivp():
    return threebody()


def test_initialize_with_rk(ivp):
    """Make sure that the values are close(ish) to the truth."""
    a, e = pnde.compute_all_derivatives_via_rk(
        ivp.f, ivp.y0, ivp.t0, h0=1e-2, order=4, method="RK45"
    )
    expected = _correct_order_of_elements(THREEBODY_INITS[:16], order=3)
    received = a
    print()
    print()
    print()
    # print(np.log(np.linalg.norm((expected - received) / (1.0 + expected))))
    print(expected)
    print(received)
    assert False


def _correct_order_of_elements(arr, order):
    """Utility function to change ordering of elements in stacked vector."""
    return arr.reshape((4, order + 1)).T.flatten()
