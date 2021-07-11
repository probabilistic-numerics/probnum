"""Tests for random variable configurations."""

import numpy as np
import pytest

from probnum import config, randvars


@pytest.fixture
def zero_cov_normal():
    return randvars.Normal(np.random.rand(5), np.zeros((5, 5)))


def test_randvars_config(zero_cov_normal):
    # Check default
    assert config.covariance_inversion_damping == 1e-12

    chol_1 = zero_cov_normal.dense_cov_cholesky()
    np.testing.assert_allclose(
        np.diag(chol_1), np.full(shape=(chol_1.shape[0],), fill_value=np.sqrt(1e-12))
    )

    with config(covariance_inversion_damping=1e-3):
        chol_2 = zero_cov_normal.dense_cov_cholesky()
        np.testing.assert_allclose(
            np.diag(chol_2),
            np.full(shape=(chol_1.shape[0],), fill_value=np.sqrt(1e-3)),
        )
