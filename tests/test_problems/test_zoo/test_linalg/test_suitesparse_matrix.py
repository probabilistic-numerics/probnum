"""Tests for SuiteSparse matrices and related functions."""

import unittest

import scipy.sparse

from probnum.problems.zoo.linalg import suitesparse_matrix
from tests.testing import NumpyAssertions


class SuiteSparseMatrixTestCase(unittest.TestCase, NumpyAssertions):
    """Test case for the interface to the SuiteSparse matrix collection."""

    def setUp(self) -> None:
        """Test resources for the SuiteSparse interface."""
        # Download sparse matrices
        matids = [1438, 2758, 904]
        self.sparse_matrices = []
        for matid in matids:
            self.sparse_matrices.append(
                suitesparse_matrix(
                    matid=matid,
                    query_only=False,
                )
            )

    def tearDown(self) -> None:
        """Tear down created resources."""
        # TODO remove created database

    def test_search_by_id(self):
        m = suitesparse_matrix(matid=42)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].matid, 42)

    def test_search_by_group(self):
        matrices = suitesparse_matrix(group="HB/*")
        for matrix in matrices:
            self.assertEqual(matrix.group, "HB")

    def test_search_by_name(self):
        matrices = suitesparse_matrix(name="c-")
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(matrix.name.startswith("c-"))

    def test_filter_by_rows(self):
        matrices = suitesparse_matrix(rows=(None, 1000))
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertTrue(matrix.rows <= 1000)

    def test_filter_by_pattern_symmetry(self):
        psym_limits = (0.1, 1.0)
        matrices = suitesparse_matrix(psym=psym_limits)
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertGreaterEqual(matrix.psym, psym_limits[0])
            self.assertLessEqual(matrix.psym, psym_limits[1])

    def test_filter_by_numerical_symmetry(self):
        nsym_limits = (0.1, 0.5)
        matrices = suitesparse_matrix(nsym=nsym_limits)
        self.assertTrue(len(matrices) > 0)
        for matrix in matrices:
            self.assertGreaterEqual(matrix.psym, nsym_limits[0])
            self.assertLessEqual(matrix.psym, nsym_limits[1])

    def test_filter_by_shape(self):
        rmin = 50
        rmax = 1000
        cmin = 200
        cmax = 5000
        matrices = suitesparse_matrix(rows=(rmin, rmax), cols=(cmin, cmax))
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

    def test_sparse_matrix_is_spmatrix(self):
        """Test whether a sparse scipy matrix is returned."""
        for mat in self.sparse_matrices:
            with self.subTest():
                self.assertIsInstance(mat, scipy.sparse.spmatrix)
