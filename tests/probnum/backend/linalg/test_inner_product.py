"""Tests for general inner products."""

from probnum import backend
from probnum.backend.linalg import induced_norm, inner_product
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest
import tests.utils


@pytest.fixture(scope="module", params=[1, 10, 50])
def n(request) -> int:
    """Vector size."""
    return request.param


@pytest.fixture(scope="module", params=[1, 3, 5])
def m(request) -> int:
    """Number of simultaneous vectors."""
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def p(request) -> int:
    """Number of matrices."""
    return request.param


@pytest.fixture(scope="module")
def vector0(n: int) -> backend.Array:
    shape = (n,)
    return backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=86,
            shape=shape,
        ),
        shape=shape,
    )


@pytest.fixture(scope="module")
def vector1(n: int) -> backend.Array:
    shape = (n,)
    return backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=567,
            shape=shape,
        ),
        shape=shape,
    )


@pytest.fixture(scope="module")
def array0(p: int, m: int, n: int) -> backend.Array:
    shape = (p, m, n)
    return backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=86,
            shape=shape,
        ),
        shape=shape,
    )


@pytest.fixture(scope="module")
def array1(m: int, n: int) -> backend.Array:
    shape = (m, n)
    return backend.random.standard_normal(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=567,
            shape=shape,
        ),
        shape=shape,
    )


def test_inner_product_vectors(vector0: backend.Array, vector1: backend.Array):
    assert inner_product(v=vector0, w=vector1) == pytest.approx(
        backend.sum(vector0 * vector1)
    )


def test_inner_product_arrays(array0: backend.Array, array1: backend.Array):
    assert inner_product(v=array0, w=array1) == pytest.approx(
        backend.einsum("...i,...i", array0, array1)
    )


def test_euclidean_norm_vector(vector0: backend.Array):
    assert backend.sqrt(backend.sum(vector0**2)) == pytest.approx(
        induced_norm(v=vector0)
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_euclidean_norm_array(array0: backend.Array, axis: int):
    assert backend.sqrt(backend.sum(array0**2, axis=axis)) == pytest.approx(
        induced_norm(v=array0, axis=axis)
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_induced_norm_array(array0: backend.Array, axis: int):
    inprod_mat = random_spd_matrix(
        rng_state=backend.random.rng_state(254),
        dim=array0.shape[axis],
    )
    array0_moved_axis = backend.moveaxis(array0, axis, -1)
    A_array_0_moved_axis = (inprod_mat @ array0_moved_axis[..., :, None])[..., 0]

    assert backend.sqrt(
        backend.sum(array0_moved_axis * A_array_0_moved_axis, axis=-1)
    ) == pytest.approx(induced_norm(v=array0, A=inprod_mat, axis=axis))
