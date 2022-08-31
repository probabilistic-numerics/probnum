import numpy as np
import pytest

import probnum as pn
from probnum import functions


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
    sum_fn = (fn0 + (fn1 + fn0)) - fn1

    assert isinstance(sum_fn, functions.SumFunction)
    assert len(sum_fn.summands) == 4
    assert sum_fn.summands[0] is fn0
    assert sum_fn.summands[1] is fn1
    assert sum_fn.summands[2] is fn0
    assert (
        isinstance(sum_fn.summands[3], functions.ScaledFunction)
        and sum_fn.summands[3].function is fn1
        and sum_fn.summands[3].scalar == -1
    )


def test_scaled_function_contracts(fn0: functions.Function):
    scaled_fn = 2.0 * -fn0

    assert isinstance(scaled_fn, functions.ScaledFunction)
    assert scaled_fn.function is fn0
    assert scaled_fn.scalar == -2.0
