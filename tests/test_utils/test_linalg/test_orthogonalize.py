"""Tests for orthogonalization functions."""

from functools import partial
from typing import Callable, Union

import numpy as np
import pytest

from probnum import linops
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.utils.linalg import (
    double_gram_schmidt,
    gram_schmidt,
    modified_gram_schmidt,
)

n = 100


@pytest.fixture(scope="module", params=[1, 10, 50])
def basis_size(request) -> int:
    """Number of basis vectors."""
    return request.param


@pytest.fixture(scope="module")
def vector() -> np.ndarray:
    rng = np.random.default_rng(526367 + n)
    return rng.standard_normal(size=(n,))


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    rng = np.random.default_rng(234 + n)
    return rng.standard_normal(size=(2, 10, n))


@pytest.fixture(
    scope="module",
    params=[
        np.eye(n),
        linops.Identity(n),
        linops.Scaling(factors=1.0, shape=(n, n)),
        np.inner,
    ],
)
def inprod(request) -> int:
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        partial(double_gram_schmidt, gram_schmidt_fn=gram_schmidt),
        partial(double_gram_schmidt, gram_schmidt_fn=modified_gram_schmidt),
    ],
)
def orthogonalization_fn(request) -> int:
    return request.param


def test_is_orthogonal(
    vector: np.ndarray,
    basis_size: int,
    inprod: Union[
        np.ndarray,
        linops.LinearOperator,
        Callable[[np.ndarray, np.ndarray], np.ndarray],
    ],
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    seed = abs(32 + hash(basis_size))
    basis = np.random.default_rng(seed).normal(size=(vector.shape[0], basis_size))
    orthogonal_basis, _ = np.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, inner_product=inprod
    )
    np.testing.assert_allclose(
        orthogonal_basis @ ortho_vector,
        np.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_is_normalized(
    vector: np.ndarray,
    basis_size: int,
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    seed = abs(9467 + hash(basis_size))
    basis = np.random.default_rng(seed).normal(size=(vector.shape[0], basis_size))
    orthogonal_basis, _ = np.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, normalize=True
    )

    assert np.inner(ortho_vector, ortho_vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "inner_product_matrix",
    [
        np.diag(np.random.default_rng(123).standard_gamma(1.0, size=(n,))),
        5 * np.eye(n),
        random_spd_matrix(rng=np.random.default_rng(46), dim=n),
    ],
)
def test_noneuclidean_innerprod(
    vector: np.ndarray,
    basis_size: int,
    inner_product_matrix: np.ndarray,
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    evals, evecs = np.linalg.eigh(inner_product_matrix)
    orthogonal_basis = evecs * 1 / np.sqrt(evals)
    orthogonal_basis = orthogonal_basis[:, 0:basis_size].T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product_matrix,
        normalize=False,
    )

    np.testing.assert_allclose(
        orthogonal_basis @ inner_product_matrix @ ortho_vector,
        np.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_broadcasting(
    vectors: np.ndarray,
    basis_size: int,
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    seed = abs(32 + hash(basis_size))
    basis = np.random.default_rng(seed).normal(size=(vectors.shape[-1], basis_size))
    orthogonal_basis, _ = np.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vectors = orthogonalization_fn(v=vectors, orthogonal_basis=orthogonal_basis)
    np.testing.assert_allclose(
        np.squeeze(orthogonal_basis @ ortho_vectors[..., None], axis=-1),
        np.zeros(vectors.shape[:-1] + (basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )
