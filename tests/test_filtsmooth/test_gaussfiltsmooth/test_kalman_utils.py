"""Test Kalman utility functions."""


import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv


@pytest.fixture
def ordint():
    return 1


@pytest.fixture
def ordint():
    return 4


@pytest.fixture
def spatialdim():
    return 1


@pytest.fixture
def spatialdim():
    return 3


@pytest.fixture
def dynamics(ordint, spatialdim):
    dynamic_model = pnfss.IBM(ordint, spatialdim)
    initrv = pnrv.Normal(
        np.zeros(dynamic_model.dimension), np.eye(dynamic_model.dimension)
    )
    return dynamic_model, initrv


def test_rts_smooth_step(dynamics):
    smooth_without = pnfs.rts_smooth_step_classic
    smooth_with = pnfs.rts_with_precon(smooth_without)

    pass
