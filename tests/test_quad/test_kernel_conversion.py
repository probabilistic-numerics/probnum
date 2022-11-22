"""Test cases for converting kernels to product kernels in quad."""

from probnum.quad.kernel_embeddings._matern_lebesgue import _convert_to_product_matern
from probnum.randprocs.kernels import Matern

import pytest


def test_product_kernel_conversion_matern():
    kernel = Matern(input_shape=(1,))
    product_kernel = _convert_to_product_matern(kernel)

    # shapes
    assert product_kernel.lengthscales.shape == (1,)
    assert product_kernel.nus.shape == (1,)
    assert product_kernel.input_shape == (1,)

    # values
    assert product_kernel.lengthscales[0] == kernel.lengthscale
    assert product_kernel.nus[0] == kernel.nu

    # raises
    with pytest.raises(NotImplementedError):
        _convert_to_product_matern(Matern(input_shape=(2,)))
