"""Tests for kernel arithmetic."""
from probnum.randprocs import kernels
from probnum.randprocs.kernels._arithmetic_fallbacks import (
    ProductKernel,
    ScaledKernel,
    SumKernel,
)


def test_scalar_mul(kernel: kernels.Kernel):
    scalar = 3.14
    kernel_mul = kernel * scalar
    kernel_rmul = scalar * kernel
    assert isinstance(kernel_mul, ScaledKernel)
    assert isinstance(kernel_rmul, ScaledKernel)
    assert kernel_mul.input_shape == kernel.input_shape
    assert kernel_rmul.input_shape == kernel.input_shape
    assert kernel_mul.output_shape_0 == kernel.output_shape_0
    assert kernel_mul.output_shape_1 == kernel.output_shape_1
    assert kernel_rmul.output_shape_0 == kernel.output_shape_0
    assert kernel_rmul.output_shape_1 == kernel.output_shape_1


def test_add(kernel: kernels.Kernel):
    k_whitenoise = kernels.WhiteNoise(input_shape=kernel.input_shape)
    kernel_sum = kernel + k_whitenoise
    assert isinstance(kernel_sum, SumKernel)
    assert kernel_sum.input_shape == kernel.input_shape
    assert kernel_sum.output_shape_0 == kernel.output_shape_0
    assert kernel_sum.output_shape_1 == kernel.output_shape_1


def test_mul(kernel: kernels.Kernel):
    k_poly = kernels.Polynomial(input_shape=kernel.input_shape)
    kernel_prod = kernel * k_poly
    assert isinstance(kernel_prod, ProductKernel)
    assert kernel_prod.input_shape == kernel.input_shape
    assert kernel_prod.output_shape_0 == kernel.output_shape_0
    assert kernel_prod.output_shape_1 == kernel.output_shape_1
