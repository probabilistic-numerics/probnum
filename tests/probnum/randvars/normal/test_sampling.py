from pytest_cases import fixture, parametrize, parametrize_with_cases

from probnum import backend, randvars
from probnum.backend.typing import ShapeLike, ShapeType
import tests.utils


@fixture(scope="module")
@parametrize(shape=[(), 3, (1,), (1, 1), (2, 3, 2)])
def sample_shape_arg(shape: ShapeLike) -> ShapeLike:
    return shape


@fixture(scope="module")
def sample_shape(sample_shape_arg: ShapeLike) -> ShapeType:
    return backend.as_shape(sample_shape_arg)


@fixture(scope="module")
@parametrize_with_cases("rv_", cases=".cases", scope="module")
def rv(rv_: randvars.Normal) -> randvars.Normal:
    return rv_


@fixture(scope="module")
def samples(rv: randvars.Normal, sample_shape_arg: ShapeLike) -> backend.Array:
    return rv.sample(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=9879,
            shape=sample_shape_arg,
        ),
        sample_shape=sample_shape_arg,
    )


def test_sample_shape(
    samples: backend.Array, rv: randvars.Normal, sample_shape: ShapeType
):
    assert samples.shape == sample_shape + rv.shape
