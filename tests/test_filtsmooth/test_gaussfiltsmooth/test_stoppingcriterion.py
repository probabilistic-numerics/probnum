"""Test Kalman utility functions."""


import numpy as np
import pytest

import probnum.filtsmooth as pnfs


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
    return pnfs.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)


def test_continue(stopcrit, d1, d2):
    """Iteration must not terminate if error is large and maxit is not reached."""
    y1 = np.random.rand(d1, d2)
    y2 = 2 * y1.copy() + 3
    assert stopcrit.do_not_terminate_yet(y1 - y2, y2) is True


def test_terminate_tolerance(stopcrit, d1, d2):
    """Iteration must terminate if error is small."""
    y1 = np.random.rand(d1, d2)
    y2 = y1.copy() + 1e-8
    assert stopcrit.do_not_terminate_yet(y1 - y2, y2) is False


def test_terminate_maxit(stopcrit, d1, d2, maxit):
    """Iteration must terminate if error is small."""
    # error is large
    y1 = np.random.rand(d1, d2)
    y2 = 2 * y1.copy() + 3
    stopcrit.iterations = maxit + 1
    with pytest.raises(RuntimeError):
        stopcrit.do_not_terminate_yet(y1 - y2, y2)
