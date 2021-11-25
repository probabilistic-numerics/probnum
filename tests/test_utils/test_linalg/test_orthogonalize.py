"""Tests for orthogonalization functions."""

from typing import Callable

import numpy as np
from pytest_cases import parametrize

from probnum.utils.linalg import double_gram_schmidt, gram_schmidt

rng = np.random.default_rng(42)


@parametrize("vector", rng.normal(size=(3, 100)))
@parametrize("basis_size", [1, 10, 50])
@parametrize("orthogonalization_fn", [gram_schmidt, double_gram_schmidt])
def test_is_orthogonal(
    vector: np.ndarray,
    basis_size: int,
    orthogonalization_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    # Compute orthogonal basis
    orthogonal_basis, _ = np.linalg.qr(rng.normal(size=(vector.shape[0], basis_size)))

    # Orthogonalize vector
    ortho_vector = orthogonalization_fn(
        vector=vector, orthogonal_basis=orthogonal_basis, is_orthonormal=False
    )
    np.testing.assert_allclose(
        orthogonal_basis.T @ ortho_vector,
        np.zeros((basis_size,)),
        atol=1e-12,
        rtol=1e-12,
    )
