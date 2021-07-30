"""Test Kalman utility functions."""


import numpy as np
import pytest

from probnum import filtsmooth


@pytest.fixture
def d1():
    return 5


@pytest.fixture
def d2():
    return 4


@pytest.fixture
def maxit():
    return 10


@pytest.fixture
def stopcrit():
    return filtsmooth.optim.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)


def test_continue(stopcrit, d1, d2):
    """Iteration must not terminate if error is large and maxit is not reached."""
    y1 = np.random.rand(d1, d2)
    y2 = 2 * y1.copy() + 3
    current_iterations = stopcrit.iterations
    assert stopcrit.terminate(y1 - y2, y2) is False
    assert stopcrit.iterations == current_iterations + 1


def test_terminate_tolerance(stopcrit, d1, d2):
    """Iteration must terminate if error is small."""
    y1 = np.random.rand(d1, d2)
    y2 = y1.copy() + 1e-8
    assert stopcrit.terminate(y1 - y2, y2) is True
    assert stopcrit.iterations == 0


def test_terminate_maxit(stopcrit, d1, d2, maxit):
    """Iteration must throw an exception if maximum number of iterations is reached."""
    # error is large, which does not really matter...
    y1 = np.random.rand(d1, d2)
    y2 = 2 * y1.copy() + 3
    stopcrit.iterations = maxit + 1
    with pytest.raises(RuntimeError):
        stopcrit.terminate(y1 - y2, y2)
