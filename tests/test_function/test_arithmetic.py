import numpy as np
import pytest

import probnum as pn
from probnum import _function


@pytest.fixture(scope="module")
def fn0() -> pn.Function:
    return pn.LambdaFunction(
        lambda xs: np.sin(np.linspace(0.5, 2.0, 6).reshape(3, 2) * np.sum(xs**2)),
        input_shape=(2,),
        output_shape=(3, 2),
    )


@pytest.fixture(scope="module")
def fn1() -> pn.Function:
    return pn.LambdaFunction(
        lambda xs: (
            np.linspace(0.5, 2.0, 6).reshape(3, 2) * np.exp(-0.5 * np.sum(xs**2))
        ),
        input_shape=(2,),
        output_shape=(3, 2),
    )


def test_add(fn0: pn.Function, fn1: pn.Function):
    xs = np.random.default_rng(234).uniform(-1.0, 1.0, (2,))

    np.testing.assert_array_equal(
        (fn0 + fn1)(xs),
        fn0(xs) + fn1(xs)
    )

def test_sum_function_contracts(fn0: pn.Function, fn1: pn.Function):
    sum_fn = (fn0 + (fn1 + fn0)) + fn1

    assert isinstance(sum_fn, _function.SumFunction)
    assert len(sum_fn.summands) == 4
