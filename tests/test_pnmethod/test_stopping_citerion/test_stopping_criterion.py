"""Tests for stopping criteria of probabilistic linear solvers."""

import operator
from typing import Callable

import pytest

from probnum import LambdaStoppingCriterion, StoppingCriterion


@pytest.fixture
def stopcrit():
    return LambdaStoppingCriterion(stopcrit=lambda: True)


def test_invert_stopcrit(stopcrit: StoppingCriterion):
    assert (not stopcrit()) == (~stopcrit)()


@pytest.mark.parametrize("binary_op", [operator.and_, operator.or_])
def test_boolean_binary_arithmetic(binary_op: Callable, stopcrit: StoppingCriterion):
    assert binary_op(stopcrit(), stopcrit()) == binary_op(stopcrit, stopcrit)()


def test_lambda_stopping_criterion():
    """Test whether a stopping criterion can be created from an anonymous function."""
    stopcrit = LambdaStoppingCriterion(stopcrit=lambda tol: tol < 1e-6)
    assert stopcrit(1e-12)
