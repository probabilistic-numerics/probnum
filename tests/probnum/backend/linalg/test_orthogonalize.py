"""Tests for orthogonalization functions."""

from functools import partial
from typing import Callable, Union

import pytest

from probnum import backend, compat, linops
from probnum.backend.linalg import (
    double_gram_schmidt,
    gram_schmidt,
    modified_gram_schmidt,
)
from probnum.problems.zoo.linalg import random_spd_matrix
import tests.utils

n = 100


@pytest.fixture(scope="module", params=[1, 10, 50])
def basis_size(request) -> int:
    """Number of basis vectors."""
    return request.param


@pytest.fixture(scope="module")
def vector() -> backend.Array:
    shape = (n,)
    return backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=526367,
            shape=shape,
        ),
        shape=shape,
    )


@pytest.fixture(scope="module")
def vectors() -> backend.Array:
    shape = (2, 10, n)
    return backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=234,
            shape=shape,
        ),
        shape=shape,
    )


@pytest.fixture(
    scope="module",
    params=[
        backend.eye(n),
        linops.Identity(n),
        linops.Scaling(factors=1.0, shape=(n, n)),
        # backend.inner,
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
    vector: backend.Array,
    basis_size: int,
    inprod: Union[
        backend.Array,
        linops.LinearOperator,
        Callable[[backend.Array, backend.Array], backend.Array],
    ],
    orthogonalization_fn: Callable[[backend.Array, backend.Array], backend.Array],
):
    # Compute orthogonal basis
    basis_shape = (vector.shape[0], basis_size)
    basis = backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=32,
            shape=basis_shape,
        ),
        shape=basis_shape,
    )
    orthogonal_basis, _ = backend.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, inner_product=inprod
    )
    compat.testing.assert_allclose(
        orthogonal_basis @ ortho_vector,
        backend.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_is_normalized(
    vector: backend.Array,
    basis_size: int,
    orthogonalization_fn: Callable[[backend.Array, backend.Array], backend.Array],
):
    # Compute orthogonal basis
    basis_shape = (vector.shape[0], basis_size)
    basis = backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=9467,
            shape=basis_shape,
        ),
        shape=basis_shape,
    )
    orthogonal_basis, _ = backend.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, normalize=True
    )

    assert backend.sum(ortho_vector**2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "inner_product_matrix",
    [
        backend.diag(backend.random.gamma(backend.random.seed(123), 1.0, shape=(n,))),
        5 * backend.eye(n),
        random_spd_matrix(seed=backend.random.seed(46), dim=n),
    ],
)
def test_noneuclidean_innerprod(
    vector: backend.Array,
    basis_size: int,
    inner_product_matrix: backend.Array,
    orthogonalization_fn: Callable[[backend.Array, backend.Array], backend.Array],
):
    evals, evecs = backend.linalg.eigh(inner_product_matrix)
    orthogonal_basis = evecs * 1 / backend.sqrt(evals)
    orthogonal_basis = orthogonal_basis[:, 0:basis_size].T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector,
        orthogonal_basis=orthogonal_basis,
        inner_product=inner_product_matrix,
        normalize=False,
    )

    compat.testing.assert_allclose(
        orthogonal_basis @ inner_product_matrix @ ortho_vector,
        backend.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )


def test_broadcasting(
    vectors: backend.Array,
    basis_size: int,
    orthogonalization_fn: Callable[[backend.Array, backend.Array], backend.Array],
):
    # Compute orthogonal basis
    basis_shape = (vectors.shape[-1], basis_size)
    basis = backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=32,
            shape=basis_shape,
        ),
        shape=basis_shape,
    )
    orthogonal_basis, _ = backend.linalg.qr(basis)
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vectors = orthogonalization_fn(v=vectors, orthogonal_basis=orthogonal_basis)
    compat.testing.assert_allclose(
        (orthogonal_basis @ ortho_vectors[..., None])[..., 0],
        backend.zeros(vectors.shape[:-1] + (basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )
