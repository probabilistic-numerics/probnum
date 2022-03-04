"""Tests for fall-back implementations of kernel arithmetic."""

import numpy as np
import pytest
from pytest_cases import parametrize

from probnum.randprocs import kernels
from probnum.randprocs.kernels._arithmetic_fallbacks import (
    ProductKernel,
    ScaledKernel,
    SumKernel,
)
from probnum.typing import ScalarType


@parametrize("scalar", [1.0, 3, 1000.0])
def test_scaled_kernel_evaluation(
    kernel: kernels.Kernel, scalar: ScalarType, x0: np.ndarray
):
    k_scaled = ScaledKernel(kernel=kernel, scalar=scalar)
    np.testing.assert_allclose(k_scaled.matrix(x0), scalar * kernel.matrix(x0))


def test_non_positive_scalar_raises_error():
    with pytest.raises(ValueError):
        ScaledKernel(kernel=kernels.WhiteNoise(input_shape=()), scalar=0.0)

    with pytest.raises(ValueError):
        ScaledKernel(kernel=kernels.WhiteNoise(input_shape=()), scalar=-1.0)


def test_non_scalar_raises_error():
    with pytest.raises(TypeError):
        ScaledKernel(kernel=kernels.WhiteNoise(input_shape=()), scalar=np.array([0, 1]))


def test_non_kernel_raises_error():
    with pytest.raises(TypeError):
        ScaledKernel(kernel=np.eye(5), scalar=1.0)


def test_sum_kernel_evaluation(kernel: kernels.Kernel, x0: np.ndarray):
    k_whitenoise = kernels.WhiteNoise(input_shape=kernel.input_shape)
    k_sum = SumKernel(kernel, k_whitenoise)
    np.testing.assert_allclose(
        k_sum.matrix(x0), kernel.matrix(x0) + k_whitenoise.matrix(x0)
    )


def test_sum_kernel_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        SumKernel(
            kernels.WhiteNoise(input_shape=()), kernels.WhiteNoise(input_shape=(1,))
        )


def test_sum_kernel_contracts():
    input_shape = ()
    k = kernels.ExpQuad(input_shape=input_shape)
    k_sum = SumKernel(k, SumKernel(k, k))
    assert all(not isinstance(summand, SumKernel) for summand in k_sum._summands)


def test_product_kernel_evaluation(kernel: kernels.Kernel, x0: np.ndarray):
    k_poly = kernels.Polynomial(input_shape=kernel.input_shape)
    k_sum = ProductKernel(kernel, k_poly)
    np.testing.assert_allclose(k_sum.matrix(x0), kernel.matrix(x0) * k_poly.matrix(x0))


def test_product_kernel_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        ProductKernel(
            kernels.WhiteNoise(input_shape=()), kernels.WhiteNoise(input_shape=(1,))
        )


def test_product_kernel_contracts():
    input_shape = ()
    k = kernels.ExpQuad(input_shape=input_shape)
    k_prod = ProductKernel(k, ProductKernel(k, k))
    assert all(not isinstance(factor, ProductKernel) for factor in k_prod._factors)
