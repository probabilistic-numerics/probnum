"""Tests for fall-back implementations of covariance function arithmetic."""

import numpy as np
import pytest
from pytest_cases import parametrize

from probnum.randprocs import covfuncs
from probnum.randprocs.covfuncs._arithmetic_fallbacks import (
    ProductCovarianceFunction,
    ScaledCovarianceFunction,
    SumCovarianceFunction,
)
from probnum.typing import ScalarType


@parametrize("scalar", [1.0, 3, 1000.0])
def test_scaled_covfunc_evaluation(
    k: covfuncs.CovarianceFunction, scalar: ScalarType, x0: np.ndarray
):
    k_scaled = ScaledCovarianceFunction(k, scalar=scalar)
    np.testing.assert_allclose(k_scaled.matrix(x0), scalar * k.matrix(x0))


def test_non_scalar_raises_error():
    with pytest.raises(TypeError):
        ScaledCovarianceFunction(
            covfuncs.WhiteNoise(input_shape=()), scalar=np.array([0, 1])
        )


def test_non_covfunc_raises_error():
    with pytest.raises(TypeError):
        ScaledCovarianceFunction(np.eye(5), scalar=1.0)


def test_sum_covfunc_evaluation(k: covfuncs.CovarianceFunction, x0: np.ndarray):
    k_whitenoise = covfuncs.WhiteNoise(input_shape=k.input_shape)
    k_sum = SumCovarianceFunction(k, k_whitenoise)
    np.testing.assert_allclose(k_sum.matrix(x0), k.matrix(x0) + k_whitenoise.matrix(x0))


def test_sum_covfunc_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        SumCovarianceFunction(
            covfuncs.WhiteNoise(input_shape=()), covfuncs.WhiteNoise(input_shape=(1,))
        )


def test_sum_covfunc_contracts():
    input_shape = ()
    k = covfuncs.ExpQuad(input_shape=input_shape)
    k_sum = SumCovarianceFunction(k, SumCovarianceFunction(k, k))
    assert all(
        not isinstance(summand, SumCovarianceFunction) for summand in k_sum._summands
    )


def test_product_covfunc_evaluation(k: covfuncs.CovarianceFunction, x0: np.ndarray):
    k_poly = covfuncs.Polynomial(input_shape=k.input_shape)
    k_sum = ProductCovarianceFunction(k, k_poly)
    np.testing.assert_allclose(k_sum.matrix(x0), k.matrix(x0) * k_poly.matrix(x0))


def test_product_covfunc_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        ProductCovarianceFunction(
            covfuncs.WhiteNoise(input_shape=()), covfuncs.WhiteNoise(input_shape=(1,))
        )


def test_product_covfunc_contracts():
    input_shape = ()
    k = covfuncs.ExpQuad(input_shape=input_shape)
    k_prod = ProductCovarianceFunction(k, ProductCovarianceFunction(k, k))
    assert all(
        not isinstance(factor, ProductCovarianceFunction) for factor in k_prod._factors
    )
