import numpy as np

from probnum import functions

import pytest


@pytest.fixture(scope="module")
def fn0() -> functions.LambdaFunction:
    return functions.LambdaFunction(
        lambda xs: ((np.linspace(0.5, 2.0, 6).reshape(3, 2) @ xs[..., None])[..., 0]),
        input_shape=2,
        output_shape=3,
    )


@pytest.fixture(scope="module")
def fn1() -> functions.LambdaFunction:
    return functions.LambdaFunction(
        lambda xs: np.zeros(shape=xs.shape[:-1]),
        input_shape=2,
        output_shape=3,
    )


def test_scaling_lambda_raises_error():
    with pytest.raises(TypeError):
        functions.ScaledFunction(lambda x: 2.0 * x, scalar=2.0)


def test_sum_lambda_raises_error(fn1: functions.Function):
    with pytest.raises(TypeError):
        functions.SumFunction(lambda x: 2.0 * x, fn1)


def test_sum_function_contracts(fn0: functions.Function, fn1: functions.Function):
    sum_fn = (fn0 + (fn1 + fn0)) - fn1 + fn0 + (fn0 + fn1)

    assert isinstance(sum_fn, functions.SumFunction)
    assert len(sum_fn.summands) == 7
    assert sum_fn.summands[0] is fn0
    assert sum_fn.summands[1] is fn1
    assert sum_fn.summands[2] is fn0
    assert (
        isinstance(sum_fn.summands[3], functions.ScaledFunction)
        and sum_fn.summands[3].function is fn1
        and sum_fn.summands[3].scalar == -1
    )
    assert sum_fn.summands[4] is fn0
    assert sum_fn.summands[5] is fn0
    assert sum_fn.summands[6] is fn1


def test_sum_function_input_shape_mismatch_raises_error(fn0: functions.Function):
    fn_err = functions.LambdaFunction(
        lambda x: np.zeros(fn0.output_shape),
        input_shape=(),
        output_shape=fn0.output_shape,
    )

    with pytest.raises(ValueError):
        fn0 + fn_err  # pylint: disable=pointless-statement


def test_sum_function_output_shape_mismatch_raises_error(fn0: functions.Function):
    fn_err = functions.LambdaFunction(
        lambda x: np.zeros(()),
        input_shape=fn0.input_shape,
        output_shape=(),
    )

    with pytest.raises(ValueError):
        fn0 + fn_err  # pylint: disable=pointless-statement


def test_scaled_function_contracts(fn0: functions.Function):
    scaled_fn_mul = -fn0 * 2.0

    assert isinstance(scaled_fn_mul, functions.ScaledFunction)
    assert scaled_fn_mul.function is fn0
    assert scaled_fn_mul.scalar == -2.0

    scaled_fn_rmul = 2.0 * -fn0

    assert isinstance(scaled_fn_rmul, functions.ScaledFunction)
    assert scaled_fn_rmul.function is fn0
    assert scaled_fn_rmul.scalar == -2.0
