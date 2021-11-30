"""Test properties of normal random variables."""
import numpy as np
import scipy.stats
from pytest_cases import parametrize_with_cases

from probnum import backend


@parametrize_with_cases("rv", cases=".cases", has_tag=["univariate"])
def test_entropy(rv):
    scipy_entropy = scipy.stats.norm.entropy(
        loc=backend.to_numpy(rv.mean),
        scale=backend.to_numpy(rv.std),
    )

    np.testing.assert_allclose(backend.to_numpy(rv.entropy), scipy_entropy)
