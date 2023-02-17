"""Test cases for `CovarianceFunction`"""

import numpy as np

from probnum.randprocs import covfuncs


def test_input_ndim(k: covfuncs.CovarianceFunction):
    assert k.input_ndim == np.empty(k.input_shape).ndim


def test_input_size(k: covfuncs.CovarianceFunction):
    assert k.input_size == np.empty(k.input_shape).size
