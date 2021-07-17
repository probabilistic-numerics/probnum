import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randprocs, randvars, statespace
from tests.test_diffeq.test_odefiltsmooth.test_initialize import (  # import LV_INITS, THREEBODY_INITS
    _known_initial_derivatives,
)


@pytest.fixture
def order():
    return 5


@pytest.fixture
def lv():
    y0 = np.array([20.0, 20.0])

    # tmax is ignored anyway
    return diffeq_zoo.lotkavolterra(t0=0.0, tmax=np.inf, y0=y0)


@pytest.fixture
def lv_inits(order):
    lv_dim = 2
    vals = _known_initial_derivatives.LV_INITS[: lv_dim * (order + 1)]
    return statespace.Integrator._convert_derivwise_to_coordwise(
        vals, ordint=order, spatialdim=lv_dim
    )
