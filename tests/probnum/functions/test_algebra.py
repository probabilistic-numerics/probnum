from probnum import backend, compat, functions
from probnum.backend.typing import ShapeType

import pytest
from pytest_cases import param_fixture, param_fixtures
from tests.utils.random import rng_state_from_sampling_args

lambda_fn_0 = functions.LambdaFunction(
    lambda xs: (
        backend.sin(
            backend.linspace(0.5, 2.0, 6).reshape((3, 2))
            * backend.sum(xs**2, axis=-1)[..., None, None]
        )
    ),
    input_shape=(2,),
    output_shape=(3, 2),
)

lambda_fn_1 = functions.LambdaFunction(
    lambda xs: (
        backend.linspace(0.5, 2.0, 6).reshape((3, 2))
        * backend.exp(-0.5 * backend.sum(xs**2, axis=-1))[..., None, None]
    ),
    input_shape=(2,),
    output_shape=(3, 2),
)

op0, op1 = param_fixtures(
    "op0, op1",
    (
        pytest.param(
            lambda_fn_0,
            lambda_fn_1,
            id="LambdaFunction-LambdaFunction",
        ),
        pytest.param(
            lambda_fn_0,
            functions.Zero(lambda_fn_0.input_shape, lambda_fn_1.output_shape),
            id="LambdaFunction-Zero",
        ),
        pytest.param(
            functions.Zero(lambda_fn_0.input_shape, lambda_fn_1.output_shape),
            lambda_fn_0,
            id="Zero-LambdaFunction",
        ),
        pytest.param(
            functions.Zero((3, 3), ()),
            functions.Zero((3, 3), ()),
            id="Zero-Zero",
        ),
    ),
)

batch_shape = param_fixture("batch_shape", ((), (3,), (2, 1, 2)))


def test_add_evaluation(
    op0: functions.Function, op1: functions.Function, batch_shape: ShapeType
):
    fn_add = op0 + op1

    rng_state = rng_state_from_sampling_args(base_seed=2457, shape=batch_shape)
    xs = backend.random.uniform(
        rng_state=rng_state,
        minval=-1.0,
        maxval=1.0,
        shape=batch_shape + op0.input_shape,
    )

    compat.testing.assert_array_equal(
        fn_add(xs),
        op0(xs) + op1(xs),
    )


def test_sub_evaluation(
    op0: functions.Function, op1: functions.Function, batch_shape: ShapeType
):
    fn_sub = op0 - op1

    rng_state = rng_state_from_sampling_args(base_seed=27545, shape=batch_shape)
    xs = backend.random.uniform(
        rng_state=rng_state,
        minval=-1.0,
        maxval=1.0,
        shape=batch_shape + op0.input_shape,
    )

    compat.testing.assert_array_equal(
        fn_sub(xs),
        op0(xs) - op1(xs),
    )


@pytest.mark.parametrize("scalar", [1.0, 3, 1000.0])
def test_mul_scalar_evaluation(
    op0: functions.Function,
    scalar: backend.Scalar,
    batch_shape: ShapeType,
):
    fn_scaled = op0 * scalar

    rng_state = rng_state_from_sampling_args(base_seed=2527, shape=batch_shape)
    xs = backend.random.uniform(
        rng_state=rng_state,
        minval=-1.0,
        maxval=1.0,
        shape=batch_shape + op0.input_shape,
    )

    compat.testing.assert_array_equal(
        fn_scaled(xs),
        op0(xs) * scalar,
    )


@pytest.mark.parametrize("scalar", [1.0, 3, 1000.0])
def test_rmul_scalar_evaluation(
    op0: functions.Function,
    scalar: backend.Scalar,
    batch_shape: ShapeType,
):
    fn_scaled = scalar * op0

    rng_state = rng_state_from_sampling_args(base_seed=83664, shape=batch_shape)
    xs = backend.random.uniform(
        rng_state=rng_state,
        minval=-1.0,
        maxval=1.0,
        shape=batch_shape + op0.input_shape,
    )

    compat.testing.assert_array_equal(
        fn_scaled(xs),
        scalar * op0(xs),
    )
