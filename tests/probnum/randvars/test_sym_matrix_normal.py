from probnum import backend, compat, linops, randvars
from probnum.backend.typing import ShapeLike, ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix

from pytest_cases import THIS_MODULE, case, fixture, parametrize, parametrize_with_cases
import tests.utils


@case(tags=["symmetric-matrix"])
@parametrize("shape", [(1, 1), (2, 2), (3, 3), (5, 5)])
def case_symmetric_matrix(shape: ShapeType) -> randvars.SymmetricMatrixNormal:
    rng_state_mean, rng_state_cov = backend.random.split(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=453987,
            shape=shape,
        ),
        num=2,
    )

    assert shape[0] == shape[1]

    return randvars.SymmetricMatrixNormal(
        mean=random_spd_matrix(rng_state_mean, shape[0]),
        cov=linops.SymmetricKronecker(random_spd_matrix(rng_state_cov, shape[0])),
    )


@fixture(scope="module")
@parametrize(shape=[(), 3, (1,), (1, 1), (2, 1, 3)])
def sample_shape_arg(shape: ShapeLike) -> ShapeLike:
    return shape


@fixture(scope="module")
def sample_shape(sample_shape_arg: ShapeLike) -> ShapeType:
    return backend.asshape(sample_shape_arg)


@fixture(scope="module")
@parametrize_with_cases("rv_", cases=THIS_MODULE, scope="module")
def rv(rv_: randvars.Normal) -> randvars.Normal:
    return rv_


@fixture(scope="module")
def samples(
    rv: randvars.Normal, sample_shape_arg: ShapeLike, sample_shape: ShapeType
) -> backend.Array:
    return rv.sample(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=355231,
            shape=sample_shape + rv.shape,
        ),
        sample_shape=sample_shape_arg,
    )


def test_sample_shape(
    samples: backend.Array, rv: randvars.Normal, sample_shape: ShapeType
):
    assert samples.shape == sample_shape + rv.shape


def test_samples_symmetric(samples: backend.Array):
    compat.testing.assert_array_equal(
        backend.swapaxes(samples, -2, -1),
        samples,
    )
