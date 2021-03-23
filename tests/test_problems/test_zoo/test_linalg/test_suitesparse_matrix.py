"""Tests for SuiteSparse matrices and related functions."""

import numpy as np
import scipy.sparse

from probnum.problems.zoo.linalg import SuiteSparseMatrix


def test_downloaded_matrix_is_sparse(suitesparse_mat: SuiteSparseMatrix):
    """Test whether a sparse scipy matrix is returned."""
    assert isinstance(suitesparse_mat.A, scipy.sparse.spmatrix)


def test_attribute_parsing(suitesparse_mycielskian: SuiteSparseMatrix):
    """Test whether the attributes listed on the SuiteSparse Matrix Collection site are
    parsed correctly."""
    assert suitesparse_mycielskian.matid == "2759"
    assert suitesparse_mycielskian.group == "Mycielski"
    assert suitesparse_mycielskian.name == "mycielskian3"
    assert suitesparse_mycielskian.shape == (5, 5)
    assert suitesparse_mycielskian.nnz == 10
    assert not suitesparse_mycielskian.isspd
    assert suitesparse_mycielskian.psym == 1.0
    assert suitesparse_mycielskian.nsym == 1.0


def test_html_representation_returns_string(suitesparse_mat: SuiteSparseMatrix):
    """Test whether the HTML representation of a SuiteSparse Matrix returns a string."""
    assert isinstance(suitesparse_mat._repr_html_(), str)


def test_trace(suitesparse_mycielskian: SuiteSparseMatrix):
    """Test whether the SuiteSparse Matrix object computes the trace correctly."""
    assert suitesparse_mycielskian.trace() == np.trace(
        suitesparse_mycielskian.todense()
    )


def test_matrix_multiplication(suitesparse_mat: SuiteSparseMatrix):
    """Test whether the SuiteSparse matrix can be multiplied with."""
    zerovec = np.zeros(suitesparse_mat.shape[1])
    identity = np.eye(suitesparse_mat.shape[1])
    np.testing.assert_allclose(
        np.zeros(suitesparse_mat.shape[0]), suitesparse_mat @ zerovec
    )
    np.testing.assert_allclose(suitesparse_mat.todense(), suitesparse_mat @ identity)
