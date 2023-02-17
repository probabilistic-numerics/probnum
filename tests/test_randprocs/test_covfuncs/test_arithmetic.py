"""Tests for covariance function arithmetic."""
from probnum.randprocs import covfuncs
from probnum.randprocs.covfuncs._arithmetic_fallbacks import (
    ProductCovarianceFunction,
    ScaledCovarianceFunction,
    SumCovarianceFunction,
)


def test_scalar_mul(k: covfuncs.CovarianceFunction):
    scalar = 3.14
    k_mul = k * scalar
    k_rmul = scalar * k
    assert isinstance(k_mul, ScaledCovarianceFunction)
    assert isinstance(k_rmul, ScaledCovarianceFunction)
    assert k_mul.input_shape == k.input_shape
    assert k_rmul.input_shape == k.input_shape
    assert k_mul.output_shape_0 == k.output_shape_0
    assert k_mul.output_shape_1 == k.output_shape_1
    assert k_rmul.output_shape_0 == k.output_shape_0
    assert k_rmul.output_shape_1 == k.output_shape_1


def test_add(k: covfuncs.CovarianceFunction):
    k_whitenoise = covfuncs.WhiteNoise(input_shape=k.input_shape)
    k_sum = k + k_whitenoise
    assert isinstance(k_sum, SumCovarianceFunction)
    assert k_sum.input_shape == k.input_shape
    assert k_sum.output_shape_0 == k.output_shape_0
    assert k_sum.output_shape_1 == k.output_shape_1


def test_mul(k: covfuncs.CovarianceFunction):
    k_poly = covfuncs.Polynomial(input_shape=k.input_shape)
    k_prod = k * k_poly
    assert isinstance(k_prod, ProductCovarianceFunction)
    assert k_prod.input_shape == k.input_shape
    assert k_prod.output_shape_0 == k.output_shape_0
    assert k_prod.output_shape_1 == k.output_shape_1
