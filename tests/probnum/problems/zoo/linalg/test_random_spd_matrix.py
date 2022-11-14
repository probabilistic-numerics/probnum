"""Tests for functions generating random spd matrices."""

from typing import Union

import scipy.sparse

from probnum import backend, compat
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

import pytest
import pytest_cases


def test_dimension(
    rnd_spd_mat: Union[backend.Array, scipy.sparse.csr_matrix], n_cols: int
):
    """Test whether matrix dimension matches specified dimension."""
    assert rnd_spd_mat.shape == (n_cols, n_cols)


def test_symmetric(rnd_spd_mat: Union[backend.Array, scipy.sparse.csr_matrix]):
    """Test whether the matrix is symmetric."""
    if isinstance(rnd_spd_mat, scipy.sparse.spmatrix):
        rnd_spd_mat = rnd_spd_mat.todense()
    compat.testing.assert_equal(rnd_spd_mat, rnd_spd_mat.T)


def test_positive_definite(rnd_spd_mat: Union[backend.Array, scipy.sparse.csr_matrix]):
    """Test whether the matrix is positive definite."""
    if isinstance(rnd_spd_mat, scipy.sparse.spmatrix):
        rnd_spd_mat = rnd_spd_mat.todense()
    eigvals = backend.linalg.eigvalsh(rnd_spd_mat)
    assert backend.all(eigvals > 0.0), "Eigenvalues are not all positive."


def test_spectrum_matches_given():
    """Test whether the spectrum of the test problem matches the provided spectrum."""
    n = 10
    rng_state_spectrum, rng_state_mat = backend.random.split(
        backend.random.rng_state(234985)
    )
    spectrum = backend.sort(
        backend.random.uniform(
            rng_state=rng_state_spectrum, minval=0.1, maxval=1.0, shape=n
        )
    )
    spdmat = random_spd_matrix(rng_state=rng_state_mat, shape=(n, n), spectrum=spectrum)
    eigvals = backend.sort(backend.linalg.eigvalsh(spdmat))
    compat.testing.assert_allclose(
        spectrum,
        eigvals,
        err_msg="Provided spectrum doesn't match actual.",
    )


def test_negative_eigenvalues_throws_error():
    """Test whether a non-positive spectrum throws an error."""
    with pytest.raises(ValueError):
        random_spd_matrix(
            rng_state=backend.random.rng_state(1), shape=(3, 3), spectrum=[-1, 1, 2]
        )


def test_is_ndarray(rnd_dense_spd_mat: backend.Array):
    """Test whether the random dense spd matrix is a `backend.Array`."""
    assert isinstance(rnd_dense_spd_mat, backend.Array)


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
    spformat: str,
    sparse_matrix_class: scipy.sparse.spmatrix,
):
    """Test whether sparse matrices in different formats can be created."""

    # Scipy warns that creating DIA matrices with many diagonals is inefficient.
    # This should not dilute the test output, as the tests
    # only checks the *ability* to create large random sparse matrices.

    rng_state = backend.random.rng_state(4378354)
    n = 1000
    if spformat == "dia":
        with pytest.warns(scipy.sparse.SparseEfficiencyWarning):
            sparse_mat = random_sparse_spd_matrix(
                rng_state=rng_state,
                shape=(n, n),
                density=10**-3,
                format=spformat,
            )
    else:
        sparse_mat = random_sparse_spd_matrix(
            rng_state=rng_state, shape=(n, n), density=10**-3, format=spformat
        )
    assert isinstance(sparse_mat, sparse_matrix_class)


def test_large_sparse_matrix():
    """Test whether a large random spd matrix can be created."""
    n = 10**5
    sparse_mat = random_sparse_spd_matrix(
        rng_state=backend.random.rng_state(345), shape=(n, n), density=10**-8
    )
    assert sparse_mat.shape == (n, n)
