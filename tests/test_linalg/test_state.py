"""Tests for the state of a probabilistic linear solver."""

import pytest_cases

from probnum import problems

from .cases.linear_systems import *  # pylint: disable="wildcard-import,unused-wildcard-import"


@pytest_cases.parametrize_with_cases("spd_linsys", cases=case_spd_linsys)
def test_dimension_match(spd_linsys: problems.LinearSystem):
    assert spd_linsys.A.shape[0] == spd_linsys.b.shape[0]
