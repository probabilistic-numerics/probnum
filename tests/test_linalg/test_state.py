"""Tests for the state of a probabilistic linear solver."""

import pytest_cases

from probnum import problems

# pylint: disable="wildcard-import,unused-wildcard-import"
from .cases.linear_systems import *
from .test_solvers.cases.beliefs import *


@pytest_cases.parametrize_with_cases("spd_linsys", cases=case_spd_linsys)
def test_dimension_match(spd_linsys: problems.LinearSystem):
    assert spd_linsys.A.shape[0] == spd_linsys.b.shape[0]


def test_residual(state):
    pass


def test_next_step(state):
    pass
