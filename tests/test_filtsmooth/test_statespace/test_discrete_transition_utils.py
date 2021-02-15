import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss


def test_condition_state_on_rv(some_normal_rv1, some_normal_rv2):
    """If rv_attained == rv_forwarded, the conditioned rv is the prior rv.

    This function is indirectly tested so many times, we really don't
    need to be fancy here.
    """
    gain = np.random.rand(len(some_normal_rv1.mean), len(some_normal_rv1.mean))

    out = pnfss.condition_state_on_rv(
        some_normal_rv2, some_normal_rv2, some_normal_rv1, gain
    )
    np.testing.assert_allclose(out.mean, some_normal_rv1.mean)
    np.testing.assert_allclose(out.cov, some_normal_rv1.cov)


def test_condition_state_on_measurement(some_normal_rv1, some_normal_rv2):
    """If rv_attained == rv_forwarded, the conditioned rv is the prior rv.

    This function is indirectly tested so many times, we really don't
    need to be fancy here.
    """
    gain = np.random.rand(len(some_normal_rv1.mean), len(some_normal_rv1.mean))

    out = pnfss.condition_state_on_measurement(
        some_normal_rv2.mean, some_normal_rv2, some_normal_rv1, gain
    )

    # Only shape tests
    np.testing.assert_allclose(out.mean.shape, some_normal_rv1.mean.shape)
    np.testing.assert_allclose(out.cov.shape, some_normal_rv1.cov.shape)


def test_cholesky_update(spdmat1, spdmat2):
    expected = np.linalg.cholesky(spdmat1 + spdmat2)

    S1 = np.linalg.cholesky(spdmat1)
    S2 = np.linalg.cholesky(spdmat2)
    received = pnfss.cholesky_update(S1, S2)
    np.testing.assert_allclose(expected, received)


def test_cholesky_optional(spdmat1, test_ndim):
    H = np.random.rand(test_ndim, test_ndim)
    expected = np.linalg.cholesky(H @ spdmat1 @ H.T)
    S1 = np.linalg.cholesky(spdmat1)
    received = pnfss.cholesky_update(H @ S1)
    np.testing.assert_allclose(expected, received)
