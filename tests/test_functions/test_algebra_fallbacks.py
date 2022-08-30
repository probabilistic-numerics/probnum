from typing import Union

import numpy as np
import pytest

import probnum as pn
from probnum import _function
from probnum.typing import ShapeType, ScalarLike


@pytest.fixture(scope="module")
def seed() -> int:
    return 234


@pytest.fixture(scope="module")
def input_shape() -> ShapeType:
    return (2,)


@pytest.fixture(scope="module")
def output_shape() -> ShapeType:
    return (3, 2)


@pytest.fixture(scope="module")
def fn0(input_shape: ShapeType, output_shape: ShapeType) -> pn.Function:
    return pn.LambdaFunction(
        lambda xs: (
            np.sin(np.linspace(0.5, 2.0, 6).reshape(output_shape) * np.sum(xs**2))
        ),
        input_shape=input_shape,
        output_shape=output_shape,
    )


@pytest.fixture(scope="module")
def fn1(input_shape: ShapeType, output_shape: ShapeType) -> pn.Function:
    return pn.LambdaFunction(
        lambda xs: (
            np.linspace(0.5, 2.0, 6).reshape(output_shape)
            * np.exp(-0.5 * np.sum(xs**2))
        ),
        input_shape=input_shape,
        output_shape=output_shape,
    )


@pytest.mark.parametrize("scalar", [1.0, 3, 1000.0])
def test_scaled_function_rmul_evaluation(
    seed: int, fn0: pn.Function, scalar: ScalarLike
):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, (2,))

    fn_rmul = scalar * fn0

    np.testing.assert_array_equal(
        fn_rmul(xs),
        scalar * fn0(xs),
    )


def test_sum_function_add_evaluation(seed: int, fn0: pn.Function, fn1: pn.Function):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, (2,))

    fn_add = fn0 + fn1

    np.testing.assert_array_equal(
        fn_add(xs),
        fn0(xs) + fn1(xs),
    )


def test_sum_function_sub_evaluation(seed: int, fn0: pn.Function, fn1: pn.Function):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, (2,))

    fn_sub = fn0 - fn1

    np.testing.assert_array_equal(
        fn_sub(xs),
        fn0(xs) - fn1(xs),
    )


def test_sum_function_contracts(fn0: pn.Function, fn1: pn.Function):
    sum_fn = (fn0 + (fn1 + fn0)) - fn1

    assert isinstance(sum_fn, _function.SumFunction)
    assert len(sum_fn.summands) == 4
    assert sum_fn.summands[0] is fn0
    assert sum_fn.summands[1] is fn1
    assert sum_fn.summands[2] is fn0
    assert (
        isinstance(sum_fn.summands[3], _function.ScaledFunction)
        and sum_fn.summands[3].function is fn1
    )
