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
        # [(), ()],  # This is broken
        [(4,), slice(1, 4)],
        [(2, 3), (slice(1, 2), slice(0, 3, 2))],
    ]
)
def case_normal(
    shape_and_getitem_arg: Tuple[ShapeType, ArrayIndicesLike]
) -> Tuple[randvars.Normal, ArrayIndicesLike]:
    shape, getitem_arg = shape_and_getitem_arg

    # Generate `Normal` random variable with random parameters
    mean_seed, cov_seed = backend.random.split(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=98723,
            shape=shape,
        ),
        num=2,
    )

    mean = backend.random.standard_normal(seed=mean_seed, shape=shape)
    cov = random_spd_matrix(seed=cov_seed, dim=mean.size)

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
        seed=tests.utils.random.seed_from_sampling_args(
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
    index_tensor = np.stack(
        np.meshgrid(
            *(np.arange(0, dim) for dim in rv.shape),
            indexing="ij",
        ),
        axis=-1,
    )

    @functools.partial(np.vectorize, otypes=[np.object_], signature="(d)->()")
    def _make_index_objects(idcs: np.ndarray):
        return list(int(idx) for idx in idcs)

    index_tensor = _make_index_objects(index_tensor)

    # Select indices according to `getitem_arg`
    getitem_idx_to_original_idx = index_tensor[getitem_arg]

    # Row-vectorization of indices
    raveled_getitem_idx_to_original_idx = getitem_idx_to_original_idx.reshape(
        -1, order="C"
    )

    # "Unravel" original covariance
    cov_unraveled = rv.cov.reshape(rv.shape + rv.shape, order="C")

    for i in range(getitem_rv.cov.shape[0]):
        for j in range(getitem_rv.cov.shape[1]):
            cov_unraveled_idx = tuple(
                raveled_getitem_idx_to_original_idx[i]
                + raveled_getitem_idx_to_original_idx[j]
            )

            assert getitem_rv.cov[i, j] == cov_unraveled[cov_unraveled_idx]
