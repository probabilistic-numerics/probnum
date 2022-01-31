"""Tests for orthogonalization functions."""

from typing import Callable, Union

import numpy as np
import pytest
from pytest_cases import parametrize

from probnum import linops
from probnum.utils.linalg import (
    double_gram_schmidt,
    gram_schmidt,
    modified_gram_schmidt,
)

rng = np.random.default_rng(42)
n = 100


@parametrize("vector", rng.normal(size=(3, n)))
@parametrize("basis_size", [1, 10, 50])
@parametrize(
    "inner_product",
    [
        np.eye(n),
        linops.Identity(n),
        linops.Scaling(factors=1.0, shape=(n, n)),
        np.inner,
    ],
)
@parametrize(
    "orthogonalization_fn", [gram_schmidt, modified_gram_schmidt, double_gram_schmidt]
)
def test_is_orthogonal(
    vector: np.ndarray,
    basis_size: int,
    inner_product: Union[
        np.ndarray,
        linops.LinearOperator,
        Callable[[np.ndarray, np.ndarray], np.ndarray],
    ],
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    orthogonal_basis, _ = np.linalg.qr(rng.normal(size=(vector.shape[0], basis_size)))
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, inner_product=inner_product
    )
    np.testing.assert_allclose(
        orthogonal_basis @ ortho_vector,
        np.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )


@parametrize("vector", rng.normal(size=(3, n)))
@parametrize("basis_size", [1, 10, 50])
@parametrize(
    "orthogonalization_fn", [gram_schmidt, modified_gram_schmidt, double_gram_schmidt]
)
def test_is_normalized(
    vector: np.ndarray,
    basis_size: int,
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    orthogonal_basis, _ = np.linalg.qr(rng.normal(size=(vector.shape[0], basis_size)))
    orthogonal_basis = orthogonal_basis.T

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        v=vector, orthogonal_basis=orthogonal_basis, normalize=True
    )

    assert np.inner(ortho_vector, ortho_vector) == pytest.approx(1.0)
