"""Tests for SuiteSparse matrices and related functions."""

import unittest

import numpy as np

from probnum.problems.zoo.linalg import SuiteSparseMatrix, suitesparse_matrix
from tests.testing import NumpyAssertions


class SuiteSparseMatrixTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for SuiteSparse matrices."""
