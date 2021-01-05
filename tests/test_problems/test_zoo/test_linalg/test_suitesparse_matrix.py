"""Tests for SuiteSparse matrices and related functions."""

import unittest

from probnum.problems.zoo.linalg import suitesparse_matrix
from tests.testing import NumpyAssertions


class SuiteSparseMatrixTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the interface to the SuiteSparse matrix collection."""

    def test_search_by_id(self):
        m = suitesparse_matrix(42)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].matid, 42)

    def test_search_by_group(self):
        matrices = suitesparse_matrix("HB/*")
        for matrix in matrices:
            self.assertEqual(matrix.group, "HB")

    def test_search_by_name(self):
        matrices = suitesparse_matrix("c-")
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(matrix.name.startswith("c-"))

    def test_filter_by_rows(self):
        matrices = suitesparse_matrix(rowbounds=(None, 1000))
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(matrix.rows <= 1000)

    def test_filter_by_shape(self):
        rmin = 50
        rmax = 1000
        cmin = 200
        cmax = 5000
        matrices = suitesparse_matrix(rowbounds=(rmin, rmax), colbounds=(cmin, cmax))
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(
                matrix.rows >= rmin
                and matrix.rows <= rmax
                and matrix.cols >= cmin
                and matrix.cols <= cmax
            )

    def test_filter_by_spd_true(self):
        matrices = suitesparse_matrix(isspd=True)
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(matrix.isspd)

    def test_filter_by_spd_false(self):
        matrices = suitesparse_matrix(isspd=False)
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertFalse(matrix.isspd)

    def test_download_and_read_any_matrixformat(self):
        for matrixformat in ("MM", "MAT", "RB"):
            _ = suitesparse_matrix(matid=1438, download=True, matrixformat=matrixformat)
