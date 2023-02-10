"""Test cases for `Kernel`"""

import numpy as np

from probnum.randprocs import kernels


def test_input_ndim(kernel: kernels.Kernel):
    assert kernel.input_ndim == np.empty(kernel.input_shape).ndim


def test_input_size(kernel: kernels.Kernel):
    assert kernel.input_size == np.empty(kernel.input_shape).size
