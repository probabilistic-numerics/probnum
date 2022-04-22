import functools
from typing import Tuple

import numpy as np

from probnum import backend, compat, randvars
from probnum.backend.typing import ArrayIndicesLike, ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix

from pytest_cases import THIS_MODULE, case, fixture, parametrize, parametrize_with_cases
import tests.utils


@case(tags=["normal"])
@parametrize(
    shape_and_getitem_arg=[
        # Indexing
        [(), ()],
        [(1,), 0],
        [(2,), -1],
        [(4, 5), 2],
        [(3, 2), (0, 1)],
        [(2,), None],
        # Slicing
        [(4,), slice(1, 4)],
        [(2, 3), (slice(1, 2), slice(0, 3, 2))],
        [(3,), slice(-1, -3, -2)],
        # Advanced Indexing
        ((3, 4), ([2, 0], [3, 0])),
        ((3, 4), ([[2, 1]], [[3], [1], [2], [0]])),
        # Masking
        (
            (2, 3),
            np.array(
                [
                    [True, True, False],
                    [False, True, False],
                ]
            ),
        ),
    ]
)
def case_normal(
    shape_and_getitem_arg: Tuple[ShapeType, ArrayIndicesLike]
) -> Tuple[randvars.Normal, ArrayIndicesLike]:
    shape, getitem_arg = shape_and_getitem_arg

    # Generate `Normal` random variable with random parameters
    mean_rng_state, cov_rng_state = backend.random.split(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=98723,
            shape=shape,
        ),
        num=2,
    )

    mean = backend.random.standard_normal(rng_state=mean_rng_state, shape=shape)
    cov = random_spd_matrix(
        rng_state=cov_rng_state, shape=() if shape == () else 2 * (mean.size,)
    )

    rv = randvars.Normal(mean, cov)

    return rv, getitem_arg


@fixture(scope="module")
@parametrize_with_cases("rv_,getitem_arg_", cases=THIS_MODULE, scope="module")
def rv_and_getitem_arg(
    rv_: randvars.Normal, getitem_arg_: ArrayIndicesLike
) -> Tuple[randvars.Normal, ArrayIndicesLike]:
    return rv_, getitem_arg_


@fixture(scope="module")
def rv(rv_and_getitem_arg: Tuple[randvars.Normal, ArrayIndicesLike]) -> randvars.Normal:
    return rv_and_getitem_arg[0]


@fixture(scope="module")
def getitem_arg(
    rv_and_getitem_arg: Tuple[randvars.Normal, ArrayIndicesLike],
) -> ArrayIndicesLike:
    return rv_and_getitem_arg[1]


@fixture(scope="module")
def getitem_rv(rv: randvars.Normal, getitem_arg: ArrayIndicesLike):
    return rv[getitem_arg]


def test_shape(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    expected_shape = backend.zeros(rv.shape)[getitem_arg].shape

    assert getitem_rv.shape == expected_shape


def test_sample_shape(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    expected_shape = backend.zeros(rv.shape)[getitem_arg].shape

    sample = getitem_rv.sample(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=123897, shape=expected_shape
        )
    )

    assert sample.shape == expected_shape


def test_mean(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    compat.testing.assert_array_equal(getitem_rv.mean, rv.mean[getitem_arg])


def test_var(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    compat.testing.assert_array_equal(getitem_rv.var, rv.var[getitem_arg])
    compat.testing.assert_array_equal(getitem_rv.mean, rv.mean[getitem_arg])


def test_std(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    compat.testing.assert_array_equal(getitem_rv.std, rv.std[getitem_arg])


def test_cov(
    rv: randvars.Normal,
    getitem_arg: ArrayIndicesLike,
    getitem_rv: randvars.RandomVariable,
):
    # Create tensor, wich contains indices as elements
    if rv.ndim > 0:
        index_array = np.stack(
            np.meshgrid(
                *(np.arange(0, dim) for dim in rv.shape),
                indexing="ij",
            ),
            axis=-1,
        )

        @functools.partial(np.vectorize, otypes=[np.object_], signature="(d)->()")
        def _make_index_objects(idcs: np.ndarray):
            return list(int(idx) for idx in idcs)

        index_array = _make_index_objects(index_array)
    else:
        index_array = np.empty(shape=(), dtype=np.object_)
        index_array[()] = []

    # Select indices according to `getitem_arg`
    getitem_idx_to_original_idx = index_array[getitem_arg]

    # "Unravel" original covariance
    cov_unraveled = rv.cov.reshape(rv.shape + rv.shape, order="C")

    if isinstance(getitem_idx_to_original_idx, list):
        # __getitem__ returned a scalar random variable
        assert getitem_rv.cov.shape == ()

        cov_unraveled_idx = tuple(
            getitem_idx_to_original_idx + getitem_idx_to_original_idx
        )

        assert getitem_rv.cov[()] == cov_unraveled[cov_unraveled_idx]
    else:
        # __getitem__ returned a multi-dimensional random variable

        # Row-vectorization of indices
        raveled_getitem_idx_to_original_idx = getitem_idx_to_original_idx.reshape(
            -1, order="C"
        )

        for i in range(getitem_rv.cov.shape[0]):
            for j in range(getitem_rv.cov.shape[1]):
                cov_unraveled_idx = tuple(
                    raveled_getitem_idx_to_original_idx[i]
                    + raveled_getitem_idx_to_original_idx[j]
                )

                assert getitem_rv.cov[i, j] == cov_unraveled[cov_unraveled_idx]
