import numpy as np
import pytest

from probnum import randprocs


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def Ah_closedform(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


@pytest.fixture
def Qh_closedform(dt):
    return np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])


@pytest.fixture
def F():
    return np.array([[0.0, 1.0], [0.0, 0.0]])


# dispersion matrices in different shapes


def L0():
    return np.array([0.0, 1.0])


def L1():
    return np.array([[0.0], [1.0]])


@pytest.mark.parametrize("L", [L0(), L1()])
def test_matrix_fraction_decomposition(F, L, dt, Ah_closedform, Qh_closedform):
    """Test MFD against a closed-form IntegratedWienerTransition( solution."""
    Ah, Qh, _ = randprocs.markov.continuous.matrix_fraction_decomposition(F, L, dt=dt)

    np.testing.assert_allclose(Ah, Ah_closedform)
    np.testing.assert_allclose(Qh, Qh_closedform)
