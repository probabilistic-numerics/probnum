"""Tests for functions generating random spd matrices."""

from typing import Union

import numpy as np
import pytest
import pytest_cases
import scipy.sparse

from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


def test_dimension(
    rnd_spd_mat: Union[np.ndarray, scipy.sparse.csr_matrix], n_cols: int
):
    """Test whether matrix dimension matches specified dimension."""
    assert rnd_spd_mat.shape == (n_cols, n_cols)


def test_symmetric(rnd_spd_mat: Union[np.ndarray, scipy.sparse.csr_matrix]):
    """Test whether the matrix is symmetric."""
    if isinstance(rnd_spd_mat, scipy.sparse.spmatrix):
        rnd_spd_mat = rnd_spd_mat.todense()
    np.testing.assert_equal(rnd_spd_mat, rnd_spd_mat.T)


def test_positive_definite(rnd_spd_mat: Union[np.ndarray, scipy.sparse.csr_matrix]):
    """Test whether the matrix is positive definite."""
    if isinstance(rnd_spd_mat, scipy.sparse.spmatrix):
        rnd_spd_mat = rnd_spd_mat.todense()
    eigvals = np.linalg.eigvals(rnd_spd_mat)
    assert np.all(eigvals > 0.0), "Eigenvalues are not all positive."


def test_spectrum_matches_given(rng: np.random.Generator):
    """Test whether the spectrum of the test problem matches the provided spectrum."""
    dim = 10
    spectrum = np.sort(rng.uniform(0.1, 1, size=dim))
    spdmat = random_spd_matrix(rng=rng, dim=dim, spectrum=spectrum)
    eigvals = np.sort(np.linalg.eigvals(spdmat))
    np.testing.assert_allclose(
        spectrum,
        eigvals,
        err_msg="Provided spectrum doesn't match actual.",
    )


def test_negative_eigenvalues_throws_error(rng: np.random.Generator):
    """Test whether a non-positive spectrum throws an error."""
    with pytest.raises(ValueError):
        random_spd_matrix(rng=rng, dim=3, spectrum=[-1, 1, 2])


def test_is_ndarray(rnd_dense_spd_mat: np.ndarray):
    """Test whether the random dense spd matrix is a `np.ndarray`."""
    assert isinstance(rnd_dense_spd_mat, np.ndarray)


def test_is_spmatrix(rnd_sparse_spd_mat: scipy.sparse.spmatrix):
    """Test whether :meth:`random_sparse_spd_matrix` returns a
    `scipy.sparse.spmatrix`."""
    assert isinstance(rnd_sparse_spd_mat, scipy.sparse.spmatrix)


@pytest_cases.parametrize(
    "spformat,sparse_matrix_class",
    [
        ("bsr", scipy.sparse.bsr_matrix),
        ("coo", scipy.sparse.coo_matrix),
        ("csc", scipy.sparse.csc_matrix),
        ("csr", scipy.sparse.csr_matrix),
        ("dia", scipy.sparse.dia_matrix),
        ("dok", scipy.sparse.dok_matrix),
        ("lil", scipy.sparse.lil_matrix),
    ],
)
def test_sparse_formats(
    spformat: str, sparse_matrix_class: scipy.sparse.spmatrix, rng: np.random.Generator
):
    """Test whether sparse matrices in different formats can be created."""

    # Scipy warns that creating DIA matrices with many diagonals is inefficient.
    # This should not dilute the test output, as the tests
    # only checks the *ability* to create large random sparse matrices.
    if spformat == "dia":
        with pytest.warns(scipy.sparse.SparseEfficiencyWarning):
            sparse_mat = random_sparse_spd_matrix(
                rng=rng, dim=1000, density=10 ** -3, format=spformat
            )
    else:
        sparse_mat = random_sparse_spd_matrix(
            rng=rng, dim=1000, density=10 ** -3, format=spformat
        )
    assert isinstance(sparse_mat, sparse_matrix_class)


def test_large_sparse_matrix(rng: np.random.Generator):
    """Test whether a large random spd matrix can be created."""
    n = 10 ** 5
    sparse_mat = random_sparse_spd_matrix(rng=rng, dim=n, density=10 ** -8)
    assert sparse_mat.shape == (n, n)
