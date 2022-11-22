"""Tests for functions with fixed in- and output shape."""

import numpy as np

from probnum.functions import LambdaFunction

import pytest


def test_input_shape_mismatch_raises_error():
    fn = LambdaFunction(fn=lambda x: 2 * x, input_shape=(1,), output_shape=(1,))
    with pytest.raises(ValueError):
        fn(np.ones(shape=(3, 2)))
