"""Tests for random variable arithmetic for constants."""
import operator
from typing import Callable

import numpy as np
import pytest

from probnum import randvars


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.floordiv,
        operator.pow,
    ],
)
@pytest.mark.parametrize("shape_const", [2, (3,), (2, 3)])
def test_constant_constant_entrywise_op(op: Callable, constant: randvars.Constant):
    rv = op(constant, constant)
    np.testing.assert_allclose(rv.support, op(constant.support, constant.support))


@pytest.mark.parametrize("shape_const", [(3,), (2, 2)])
def test_constant_constant_matmul(constant: randvars.Constant):
    rv = constant @ constant
    np.testing.assert_allclose(rv.support, constant.support @ constant.support)
