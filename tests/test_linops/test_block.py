"""Tests for block structure linear operators."""

import numpy as np
import pytest

import probnum as pn


def test_no_block():
    with pytest.raises(ValueError):
        bdm = pn.linops.BlockDiagonalMatrix()


def test_property_inference():
    M = pn.linops.Matrix(np.array([[1.0, -2.0], [-2.0, 5.0]]))
    M.is_symmetric = True
    M.is_positive_definite = True
    M.is_lower_triangular = False
    M.is_upper_triangular = False
    bdm = pn.linops.BlockDiagonalMatrix(pn.linops.Identity((2, 2)), M)
    assert bdm.is_symmetric
    assert bdm.is_positive_definite
    assert bdm.is_lower_triangular is False
    assert bdm.is_upper_triangular is False
