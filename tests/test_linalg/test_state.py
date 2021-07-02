"""Tests for the state of a probabilistic linear solver."""

import pytest_cases

from .cases_linalg.matrix_cases import *


@pytest_cases.parametrize_with_cases("linsys", cases=case_spd_linsys)
def test_dimension_match(linsys):
    assert linsys.A.shape[0] == linsys.b.shape[0]
