from probnum import backend, compat, randvars
from probnum.backend.typing import ShapeLike, ShapeType

from pytest_cases import fixture, parametrize, parametrize_with_cases
import tests.utils


@fixture(scope="module")
@parametrize(shape=[(), 3, (1,), (1, 1), (2, 3, 2)])
def sample_shape_arg(shape: ShapeLike) -> ShapeLike:
    return shape


@fixture(scope="module")
def sample_shape(sample_shape_arg: ShapeLike) -> ShapeType:
    return backend.asshape(sample_shape_arg)


@fixture(scope="module")
@parametrize_with_cases("rv_", cases=".cases", scope="module")
def rv(rv_: randvars.Normal) -> randvars.Normal:
    return rv_


@fixture(scope="module")
def samples(
    rv: randvars.Normal, sample_shape_arg: ShapeLike, sample_shape: ShapeType
) -> backend.Array:
    return rv.sample(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=9879,
            shape=sample_shape + rv.shape,
        ),
        sample_shape=sample_shape_arg,
    )


def test_sample_shape(
    samples: backend.Array, rv: randvars.Normal, sample_shape: ShapeType
):
    assert samples.shape == sample_shape + rv.shape


@parametrize_with_cases("rv_constant", cases=".cases", has_tag=["constant"])
def test_sample_constant(rv_constant: randvars.Normal):
    sample = rv_constant.sample(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=2346,
            shape=rv_constant.shape,
        )
    )

    compat.testing.assert_allclose(sample, rv_constant.mean)
